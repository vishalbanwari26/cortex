"""Simulated robot skills.

These stand in for real actuator calls. Each skill reads and mutates the
`WorldState` so the loop has real consequences to reason over: navigation
changes the robot's location, grasp only succeeds when the robot is co-located
with the target, and so on. Swapping these for a ROS bridge or a real robot SDK
would not touch the orchestrator or the planning logic.
"""

from __future__ import annotations

from ..scene import WorldState
from .registry import Skill, SkillRegistry, SkillResult


def _same_place(a: str, b: str) -> bool:
    """Tolerant location match.

    A real VLM emits free text ("on the wooden table", "kitchen counter"), so
    exact equality is too strict. Normalize and allow containment either way.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _navigate(target: str, world: WorldState) -> SkillResult:
    world.robot_location = target
    world.log(f"navigated to {target}")
    return SkillResult(ok=True, observation=f"Arrived at {target}.")


def _grasp(target: str, world: WorldState) -> SkillResult:
    obj = world.scene.object_named(target) if world.scene else None
    if obj is None:
        return SkillResult(ok=False, observation=f"No object matching '{target}' in scene.")
    if not _same_place(world.robot_location, obj.location):
        return SkillResult(
            ok=False,
            observation=(
                f"'{target}' is at {obj.location} but the robot is at "
                f"{world.robot_location}; out of reach."
            ),
        )
    if world.holding is not None:
        return SkillResult(ok=False, observation=f"Already holding {world.holding}.")
    world.holding = obj.name
    world.log(f"grasped {obj.name}")
    return SkillResult(ok=True, observation=f"Grasped {obj.name}.")


def _place(target: str, world: WorldState) -> SkillResult:
    if world.holding is None:
        return SkillResult(ok=False, observation="Nothing in gripper to place.")
    held = world.holding
    world.holding = None
    world.log(f"placed {held} at {target}")
    return SkillResult(ok=True, observation=f"Placed {held} at {target}.")


def _scan(target: str, world: WorldState) -> SkillResult:
    world.log(f"scanned {target}")
    summary = world.scene.summary if world.scene else "no scene loaded"
    return SkillResult(ok=True, observation=f"Scan of {target}: {summary}")


def default_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(Skill("navigate", "Move the robot to a named location.", _navigate))
    reg.register(Skill("grasp", "Pick up an object the robot is co-located with.", _grasp))
    reg.register(Skill("place", "Put down the currently held object at a location.", _place))
    reg.register(Skill("scan", "Re-observe a location and refresh the scene summary.", _scan))
    return reg
