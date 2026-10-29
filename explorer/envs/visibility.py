from __future__ import annotations

import numpy as np
from minigrid.core.grid import Grid


def compute_raycast_vis_mask(grid: Grid, agent_pos: tuple[int, int]) -> np.ndarray:
    """Compute true Line-of-Sight visibility mask using corner-aware Bresenham raycasting.

    Guarantees that:
    1. Direct walls facing the agent are visible on their front surface.
    2. All tiles located behind opaque objects (Wall) are completely occluded (False).
    3. Diagonal rays cannot slip through wall corners (corner-cutting occlusion leak).
    4. Occluded objects are hidden (set to None) in the partially observable sub-grid.
    """
    w, h = grid.width, grid.height
    ax, ay = agent_pos

    vis_mask = np.zeros((w, h), dtype=bool)
    vis_mask[ax, ay] = True

    for tx in range(w):
        for ty in range(h):
            if (tx, ty) == (ax, ay):
                continue

            dx = abs(tx - ax)
            dy = abs(ty - ay)
            x, y = ax, ay
            sx = 1 if ax < tx else -1
            sy = 1 if ay < ty else -1

            blocked = False

            if dx > dy:
                err = dx / 2.0
                while x != tx:
                    err -= dy
                    prev_x, prev_y = x, y
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx

                    if (x, y) == (tx, ty):
                        # Reached target cell: check diagonal corner pass to target
                        if x != prev_x and y != prev_y:
                            c1 = grid.get(prev_x, y)
                            c2 = grid.get(x, prev_y)
                            if (c1 and not c1.see_behind()) and (c2 and not c2.see_behind()):
                                blocked = True
                        break

                    # Intermediate cell along ray
                    cell = grid.get(x, y)
                    if cell and not cell.see_behind():
                        blocked = True
                        break

                    # Check diagonal step corner obstruction
                    if x != prev_x and y != prev_y:
                        c1 = grid.get(prev_x, y)
                        c2 = grid.get(x, prev_y)
                        if (c1 and not c1.see_behind()) or (c2 and not c2.see_behind()):
                            blocked = True
                            break
            else:
                err = dy / 2.0
                while y != ty:
                    err -= dx
                    prev_x, prev_y = x, y
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy

                    if (x, y) == (tx, ty):
                        if x != prev_x and y != prev_y:
                            c1 = grid.get(prev_x, y)
                            c2 = grid.get(x, prev_y)
                            if (c1 and not c1.see_behind()) and (c2 and not c2.see_behind()):
                                blocked = True
                        break

                    cell = grid.get(x, y)
                    if cell and not cell.see_behind():
                        blocked = True
                        break

                    if x != prev_x and y != prev_y:
                        c1 = grid.get(prev_x, y)
                        c2 = grid.get(x, prev_y)
                        if (c1 and not c1.see_behind()) or (c2 and not c2.see_behind()):
                            blocked = True
                            break

            if not blocked:
                vis_mask[tx, ty] = True

    # Obscure occluded objects in the sub-grid (aligning with MiniGrid behavior)
    for j in range(h):
        for i in range(w):
            if not vis_mask[i, j]:
                grid.set(i, j, None)

    return vis_mask
