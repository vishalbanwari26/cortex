"""The orchestrator: the main conversational agent.

It owns the end-to-end cognitive loop and coordinates the sub-agents:

    perceive (VLM)  ->  plan  ->  execute step-by-step  ->  critique on failure
                                       ^                          |
                                       |__________ replan ________|

Every stage emits events so a consumer sees reasoning and action in real time.
The loop is bounded by `max_replans` so a pathological scene cannot spin
forever. This is the piece a hiring manager should read first: it is small on
purpose, because the intelligence lives in the agents and the loop just
sequences them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents import ActorAgent, CriticAgent, PerceptionAgent, PlannerAgent, PlanStep
from .events import Event, EventBus, EventType
from .llm import ImageInput, LLMClient
from .scene import Scene, WorldState
from .skills import SkillRegistry, default_registry


@dataclass
class RunResult:
    success: bool
    goal: str
    world: WorldState
    steps_executed: int
    replans: int
    transcript: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        llm: LLMClient,
        registry: SkillRegistry | None = None,
        bus: EventBus | None = None,
        max_replans: int = 2,
    ) -> None:
        self.bus = bus or EventBus()
        self.registry = registry or default_registry()
        self.max_replans = max_replans

        self.perception = PerceptionAgent(llm)
        self.planner = PlannerAgent(llm, self.registry)
        self.actor = ActorAgent(self.registry)
        self.critic = CriticAgent(llm)

    def _emit(self, type_: EventType, message: str, **payload) -> None:
        self.bus.emit(Event(type=type_, message=message, payload=payload))

    def run(
        self,
        goal: str,
        note: str = "",
        image: ImageInput | None = None,
        scene: Scene | None = None,
    ) -> RunResult:
        world = WorldState()
        transcript: list[str] = []

        # 1. Perceive (skipped if a scene is supplied directly).
        if scene is None:
            self._emit(EventType.PERCEIVING, "Observing the scene...")
            scene = self.perception.perceive(note=note, image=image)
        world.scene = scene
        self._emit(EventType.SCENE_READY, scene.summary, objects=len(scene.objects))
        transcript.append(f"scene: {scene.summary}")

        # 2. Plan.
        self._emit(EventType.PLANNING, f"Planning to: {goal}")
        plan = self.planner.plan(goal, scene, world)
        self._emit(EventType.PLAN_READY, self._render_plan(plan), steps=len(plan))

        replans = 0
        steps_executed = 0
        feedback: str | None = None

        # 3. Execute / critique / replan loop.
        while True:
            failed_step: PlanStep | None = None
            failure_obs = ""

            for step in plan:
                self._emit(
                    EventType.STEP_STARTED,
                    f"[{step.skill}] {step.action} -> {step.target}",
                    step=step.id,
                )
                result = self.actor.execute(step, world)
                steps_executed += 1
                self._emit(
                    EventType.STEP_RESULT,
                    result.observation,
                    ok=result.ok,
                    step=step.id,
                )
                transcript.append(f"{'ok ' if result.ok else 'err'} {step.action}: {result.observation}")
                if not result.ok:
                    failed_step = step
                    failure_obs = result.observation
                    break

            if failed_step is None:
                self._emit(EventType.DONE, "Goal achieved.")
                return RunResult(True, goal, world, steps_executed, replans, transcript)

            # A step failed: ask the critic what to do.
            critique = self.critic.assess(
                goal, f"{failed_step.action} -> {failed_step.target}", failure_obs, world
            )
            self._emit(EventType.CRITIQUE, critique.reason, verdict=critique.verdict, hint=critique.hint)
            transcript.append(f"critique: {critique.verdict} ({critique.reason})")

            if not critique.should_replan or replans >= self.max_replans:
                self._emit(EventType.FAILED, f"Aborting: {critique.reason}")
                return RunResult(False, goal, world, steps_executed, replans, transcript)

            replans += 1
            feedback = f"{failure_obs} Hint: {critique.hint}"
            self._emit(EventType.REPLANNING, f"Replanning (attempt {replans})...")
            plan = self.planner.plan(goal, scene, world, feedback=feedback)
            self._emit(EventType.PLAN_READY, self._render_plan(plan), steps=len(plan))

    @staticmethod
    def _render_plan(plan: list[PlanStep]) -> str:
        return " | ".join(f"{s.id}. {s.action} {s.target}" for s in plan)
