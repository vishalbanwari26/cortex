"""Scene and world-state representations.

`Scene` is the perceived snapshot produced by the perception agent from a
multimodal input. `WorldState` is the mutable belief the orchestrator updates
as skills execute, so later steps reason over the consequences of earlier ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SceneObject:
    name: str
    location: str
    state: str = "unknown"


@dataclass
class Scene:
    summary: str
    objects: list[SceneObject] = field(default_factory=list)

    def object_named(self, name: str) -> SceneObject | None:
        target = name.lower().strip()
        for obj in self.objects:
            if obj.name.lower() == target or target in obj.name.lower():
                return obj
        return None

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        objects = [
            SceneObject(
                name=o["name"],
                location=o.get("location", "unknown"),
                state=o.get("state", "unknown"),
            )
            for o in data.get("objects", [])
        ]
        return cls(summary=data.get("summary", ""), objects=objects)


@dataclass
class WorldState:
    """The robot's running belief about itself and the scene."""

    robot_location: str = "dock"
    holding: str | None = None
    scene: Scene | None = None
    history: list[str] = field(default_factory=list)

    def log(self, line: str) -> None:
        self.history.append(line)
