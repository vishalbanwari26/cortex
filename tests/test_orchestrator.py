"""End-to-end and unit tests, all on the offline MockClient (no key, no network).

The headline test asserts the adaptive loop: the first plan omits navigation,
grasp fails, the critic triggers a replan, and the corrected plan succeeds.
"""

from cortex import Orchestrator, EventType
from cortex.llm import MockClient
from cortex.scene import Scene, WorldState
from cortex.skills import default_registry
from cortex.skills.simulated import _grasp, _navigate


def test_full_loop_recovers_via_replan():
    events = []
    orch = Orchestrator(MockClient())
    orch.bus.subscribe(events.append)

    result = orch.run("Put the red mug in the cupboard.")

    assert result.success is True
    assert result.replans == 1                      # exactly one correction
    assert result.world.holding is None             # mug was placed, not still held
    types = [e.type for e in events]
    assert EventType.REPLANNING in types            # the loop actually adapted
    assert types[-1] == EventType.DONE


def test_emits_events_in_expected_order():
    orch = Orchestrator(MockClient())
    seen = []
    orch.bus.subscribe(lambda e: seen.append(e.type))
    orch.run("Put the red mug in the cupboard.")
    assert seen[0] == EventType.PERCEIVING
    assert EventType.SCENE_READY in seen
    assert EventType.PLAN_READY in seen


def test_grasp_fails_when_not_co_located():
    scene = Scene.from_dict(
        {"summary": "x", "objects": [{"name": "red mug", "location": "table", "state": "upright"}]}
    )
    world = WorldState(robot_location="dock", scene=scene)
    result = _grasp("red mug", world)
    assert result.ok is False
    assert "out of reach" in result.observation


def test_grasp_succeeds_after_navigate():
    scene = Scene.from_dict(
        {"summary": "x", "objects": [{"name": "red mug", "location": "table", "state": "upright"}]}
    )
    world = WorldState(robot_location="dock", scene=scene)
    _navigate("table", world)
    result = _grasp("red mug", world)
    assert result.ok is True
    assert world.holding == "red mug"


def test_registry_catalog_lists_skills():
    catalog = default_registry().catalog()
    for name in ("navigate", "grasp", "place", "scan"):
        assert name in catalog
