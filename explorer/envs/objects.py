from __future__ import annotations

import numpy as np
from minigrid.core.constants import (
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_circle

# Register custom object types in MiniGrid's global dictionaries if not present
if "poi" not in OBJECT_TO_IDX:
    poi_idx = max(OBJECT_TO_IDX.values()) + 1
    OBJECT_TO_IDX["poi"] = poi_idx
    IDX_TO_OBJECT[poi_idx] = "poi"

if "monster" not in OBJECT_TO_IDX:
    monster_idx = max(OBJECT_TO_IDX.values()) + 1
    OBJECT_TO_IDX["monster"] = monster_idx
    IDX_TO_OBJECT[monster_idx] = "monster"


class PointOfInterest(WorldObj):
    """Point of Interest (POI) object.
    
    POIs are stationary targets located on walkable floor tiles.
    The agent can overlap with them, but cannot pick them up.
    Discovery occurs as soon as the POI tile enters the agent's visible field of view.
    """

    def __init__(self, color: str = "purple"):
        super().__init__("poi", color)

    def can_overlap(self) -> bool:
        return True

    def can_pickup(self) -> bool:
        return False

    def see_behind(self) -> bool:
        return True

    def render(self, img: np.ndarray) -> None:
        """Render POI as a distinct circular jewel."""
        # Use bright purple for POI
        fill_coords(img, point_in_circle(0.5, 0.5, 0.35), (180, 0, 255))
        # Inner shiny center
        fill_coords(img, point_in_circle(0.5, 0.5, 0.15), (230, 180, 255))
