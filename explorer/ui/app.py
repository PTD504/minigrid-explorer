from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pygame

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from explorer.agents import (
    BaseAgent,
    FrontierExplorerAgent,
    OraclePlannerAgent,
    RandomAgent,
    WallFollowerAgent,
)
from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper

# --- Color Palette (Cyber Dark Theme) ---
COLOR_BG = (11, 15, 25)
COLOR_PANEL_BG = (22, 30, 46)
COLOR_PANEL_BORDER = (45, 60, 90)
COLOR_TEXT_PRIMARY = (248, 250, 252)
COLOR_TEXT_SECONDARY = (148, 163, 184)
COLOR_TEXT_MUTED = (100, 116, 139)

COLOR_CYAN = (6, 182, 212)
COLOR_CYAN_LIGHT = (56, 189, 248)
COLOR_GREEN = (34, 197, 94)
COLOR_AMBER = (245, 158, 11)
COLOR_RED = (239, 68, 68)
COLOR_PURPLE = (168, 85, 247)

# Map element colors
COLOR_UNREVEALED = (18, 24, 38)
COLOR_WALL = (71, 85, 105)
COLOR_WALL_OUTLINE = (51, 65, 85)
COLOR_FLOOR_REVEALED = (120, 138, 163)
COLOR_FLOOR_VISITED = (160, 180, 205)
COLOR_AGENT = (6, 182, 212)
COLOR_MONSTER = (239, 68, 68)
COLOR_POI = (168, 85, 247)
COLOR_PATH = (56, 189, 248)

SPEED_LEVELS = [2, 6, 15, 30]


class Button:
    """Clickable UI button with active/inactive states."""

    def __init__(
        self,
        rect: tuple[int, int, int, int],
        text: str,
        tag: str,
        active: bool = False,
        bg_color: tuple[int, int, int] | None = None,
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.tag = tag
        self.active = active
        self.custom_bg = bg_color

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if self.active:
            bg = self.custom_bg or COLOR_CYAN
            border = (255, 255, 255)
            text_color = (15, 23, 42)
        else:
            bg = self.custom_bg or (30, 41, 59)
            border = COLOR_PANEL_BORDER
            text_color = COLOR_TEXT_PRIMARY

        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=6)

        text_surf = font.render(self.text, True, text_color)
        t_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, t_rect)

    def is_clicked(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


class ExplorerUIApp:
    """Interactive Graphical UI Viewer for MiniGrid Explorer.

    Provides real-time interactive testing, debugging, and visualization
    supporting manual and automated exploration baselines, difficulty scaling,
    seed randomization, and live diagnostic telemetry.
    """

    def __init__(
        self,
        difficulty: str = "easy",
        seed: int = 1000000,
        agent_type: str = "frontier",
        width: int = 1260,
        height: int = 840,
        headless: bool = False,
    ):
        self.difficulty = difficulty
        self.seed = seed
        self.agent_type = agent_type
        self.window_w = width
        self.window_h = height
        self.headless = headless

        pygame.init()
        pygame.font.init()

        if self.headless:
            self.screen = pygame.Surface((self.window_w, self.window_h))
        else:
            self.screen = pygame.display.set_mode((self.window_w, self.window_h))
            pygame.display.set_caption("MiniGrid Explorer - Interactive Research Viewer")

        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("DejaVu Sans, Arial, Helvetica", 20, bold=True)
        self.font_header = pygame.font.SysFont("DejaVu Sans, Arial, Helvetica", 14, bold=True)
        self.font_body = pygame.font.SysFont("DejaVu Sans, Arial, Helvetica", 12)
        self.font_mono = pygame.font.SysFont("DejaVu Sans Mono, monospace", 11)
        self.font_btn = pygame.font.SysFont("DejaVu Sans, Arial, Helvetica", 11, bold=True)

        # Simulation states
        self.is_playing = False
        self.speed_idx = 1  # 6 FPS default
        self.fps = SPEED_LEVELS[self.speed_idx]
        self.show_fog = True

        # Environment & Agent
        self.raw_env: ExploreEnv | None = None
        self.env: ObservationMemoryWrapper | None = None
        self.agent: BaseAgent | None = None
        self.obs: dict[str, Any] = {}
        self.info: dict[str, Any] = {}
        self.trajectory: list[tuple[int, int]] = []
        self.last_reward = 0.0
        self.last_breakdown: dict[str, float] = {}

        self.buttons: list[Button] = []
        self._init_buttons()
        self.reset_env(self.seed, self.difficulty)

    def _init_buttons(self) -> None:
        """Construct interactive UI button widgets."""
        self.buttons.clear()
        sx = 800  # Sidebar X offset

        # Difficulty Buttons
        diffs = [("easy", "Easy (17x17)"), ("medium", "Med (25x25)"), ("hard", "Hard (33x33)")]
        bw = 135
        for i, (d_key, label) in enumerate(diffs):
            btn = Button(
                rect=(sx + i * (bw + 8), 100, bw, 28),
                text=label,
                tag=f"diff_{d_key}",
                active=(d_key == self.difficulty),
            )
            self.buttons.append(btn)

        # Seed Buttons
        self.buttons.append(Button(rect=(sx, 168, 90, 26), text="◀ Prev", tag="seed_prev"))
        self.buttons.append(Button(rect=(sx + 100, 168, 130, 26), text="Randomize", tag="seed_rand", bg_color=(45, 55, 72)))
        self.buttons.append(Button(rect=(sx + 240, 168, 90, 26), text="Next ▶", tag="seed_next"))
        self.buttons.append(Button(rect=(sx + 340, 168, 80, 26), text="Reset", tag="reset_ep", bg_color=(71, 85, 105)))

        # Agent Select Buttons
        agents = [
            ("manual", "Manual (WASD)"),
            ("frontier", "Frontier (BFS)"),
            ("wall_follower", "Wall Follower"),
            ("random", "Random"),
            ("oracle", "Oracle (Upper)"),
        ]
        aw = 205
        for i, (a_key, label) in enumerate(agents):
            col = i % 2
            row = i // 2
            btn = Button(
                rect=(sx + col * (aw + 10), 242 + row * 34, aw, 28),
                text=label,
                tag=f"agent_{a_key}",
                active=(a_key == self.agent_type),
            )
            self.buttons.append(btn)

        # Playback Control Buttons
        pw = 100
        self.btn_play = Button(rect=(sx, 380, pw, 32), text="▶ Play", tag="ctrl_play", bg_color=(16, 185, 129))
        self.buttons.append(self.btn_play)
        self.buttons.append(Button(rect=(sx + pw + 10, 380, pw, 32), text="⏭ Step", tag="ctrl_step", bg_color=(56, 189, 248)))
        self.buttons.append(Button(rect=(sx + 2 * (pw + 10), 380, pw, 32), text="↺ Restart", tag="ctrl_reset"))

        # Speed buttons
        for i, spd in enumerate(["1x", "2x", "5x", "Max"]):
            btn = Button(
                rect=(sx + 340 + i * 21, 380, 19, 32),
                text=str(spd[0]),
                tag=f"speed_{i}",
                active=(i == self.speed_idx),
            )
            self.buttons.append(btn)

    def _sync_button_states(self) -> None:
        """Update active states for radio buttons."""
        for btn in self.buttons:
            if btn.tag.startswith("diff_"):
                btn.active = btn.tag == f"diff_{self.difficulty}"
            elif btn.tag.startswith("agent_"):
                btn.active = btn.tag == f"agent_{self.agent_type}"
            elif btn.tag.startswith("speed_"):
                btn.active = btn.tag == f"speed_{self.speed_idx}"

        self.btn_play.text = "⏸ Pause" if self.is_playing else "▶ Play"
        self.btn_play.custom_bg = (245, 158, 11) if self.is_playing else (16, 185, 129)

    def reset_env(self, seed: int | None = None, difficulty: str | None = None) -> None:
        """Initialize or reset the environment with the specified configuration."""
        if difficulty is not None:
            self.difficulty = difficulty
        if seed is not None:
            self.seed = seed

        self.raw_env = ExploreEnv(difficulty=self.difficulty, terminate_on_coverage=True)
        self.env = ObservationMemoryWrapper(self.raw_env)
        self.obs, self.info = self.env.reset(seed=self.seed)

        self._init_agent()
        self.trajectory = [tuple(self.raw_env.agent_pos)]
        self.last_reward = 0.0
        self.last_breakdown = {
            "reward_reveal": 0.0,
            "reward_poi": 0.0,
            "penalty_step": 0.0,
            "penalty_bump": 0.0,
            "penalty_loop": 0.0,
            "penalty_death": 0.0,
            "bonus_done": 0.0,
            "total_reward": 0.0,
        }
        self._sync_button_states()

    def _init_agent(self) -> None:
        """Instantiate the active exploration agent."""
        if self.agent_type == "random":
            self.agent = RandomAgent(seed=self.seed)
        elif self.agent_type == "wall_follower":
            self.agent = WallFollowerAgent()
        elif self.agent_type == "frontier":
            self.agent = FrontierExplorerAgent(self.env)
        elif self.agent_type == "oracle":
            self.agent = OraclePlannerAgent(self.env)
        else:
            self.agent = None  # Manual WASD control

    def step(self, action: int | None = None) -> None:
        """Advance simulation by one environment step."""
        if self.raw_env is None or self.env is None:
            return

        # If manual action is not specified, query autonomous agent
        if action is None:
            if self.agent is not None:
                action = self.agent.act(self.obs, self.info)
            else:
                return  # Waiting for manual key press

        self.obs, reward, term, trunc, self.info = self.env.step(action)
        self.last_reward = float(reward)
        self.last_breakdown = self.info.get("reward_breakdown", {})
        self.trajectory.append(tuple(self.raw_env.agent_pos))

        if term or trunc:
            self.is_playing = False
            self._sync_button_states()

    def handle_events(self) -> bool:
        """Process Pygame user input events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                for btn in self.buttons:
                    if btn.is_clicked(pos):
                        self._handle_button_click(btn.tag)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.is_playing = not self.is_playing
                    self._sync_button_states()
                elif event.key in (pygame.K_s, pygame.K_RIGHT):
                    self.is_playing = False
                    self.step()
                elif event.key == pygame.K_r:
                    self.reset_env(self.seed)
                elif event.key == pygame.K_n:
                    self.reset_env(int(np.random.randint(0, 10_000_000)))
                elif event.key == pygame.K_1:
                    self.reset_env(self.seed, "easy")
                elif event.key == pygame.K_2:
                    self.reset_env(self.seed, "medium")
                elif event.key == pygame.K_3:
                    self.reset_env(self.seed, "hard")
                elif event.key == pygame.K_TAB:
                    # Cycle agent
                    agent_keys = ["manual", "frontier", "wall_follower", "random", "oracle"]
                    nxt_idx = (agent_keys.index(self.agent_type) + 1) % len(agent_keys)
                    self.agent_type = agent_keys[nxt_idx]
                    self._init_agent()
                    self._sync_button_states()
                elif event.key == pygame.K_f:
                    self.show_fog = not self.show_fog
                # Manual WASD controls
                elif self.agent_type == "manual":
                    if event.key in (pygame.K_a, pygame.K_LEFT):
                        self.step(0)  # Left
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        self.step(1)  # Right
                    elif event.key in (pygame.K_w, pygame.K_UP):
                        self.step(2)  # Forward

        return True

    def _handle_button_click(self, tag: str) -> None:
        """Handle click actions from GUI buttons."""
        if tag.startswith("diff_"):
            d = tag.replace("diff_", "")
            self.reset_env(self.seed, d)
        elif tag.startswith("agent_"):
            self.agent_type = tag.replace("agent_", "")
            self._init_agent()
            self._sync_button_states()
        elif tag == "ctrl_play":
            self.is_playing = not self.is_playing
            self._sync_button_states()
        elif tag == "ctrl_step":
            self.is_playing = False
            self.step()
            self._sync_button_states()
        elif tag in ("ctrl_reset", "reset_ep"):
            self.reset_env(self.seed)
        elif tag == "seed_rand":
            self.reset_env(int(np.random.randint(0, 10_000_000)))
        elif tag == "seed_prev":
            self.reset_env(max(0, self.seed - 1))
        elif tag == "seed_next":
            self.reset_env(self.seed + 1)
        elif tag.startswith("speed_"):
            self.speed_idx = int(tag.replace("speed_", ""))
            self.fps = SPEED_LEVELS[self.speed_idx]
            self._sync_button_states()

    def render(self) -> None:
        """Draw complete UI scene to surface."""
        self.screen.fill(COLOR_BG)
        self._render_map_viewport()
        self._render_sidebar()

        if not self.headless:
            pygame.display.flip()

    def _render_map_viewport(self) -> None:
        """Render the 2D dungeon grid, fog of war, trajectory, and entities."""
        if self.raw_env is None:
            return

        viewport_w = 760
        viewport_h = 800
        vx = 20
        vy = 20

        # Background panel for map
        panel_rect = pygame.Rect(vx, vy, viewport_w, viewport_h)
        pygame.draw.rect(self.screen, (15, 20, 32), panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=12)

        # Scale tiles to fit viewport
        grid_sz = self.raw_env.width
        tile_sz = min((viewport_w - 40) // grid_sz, (viewport_h - 40) // grid_sz)
        map_w = grid_sz * tile_sz
        map_h = grid_sz * tile_sz
        ox = vx + (viewport_w - map_w) // 2
        oy = vy + (viewport_h - map_h) // 2

        revealed_tiles = self.raw_env.tracker.revealed_tiles
        walkable_tiles = self.raw_env.walkable_tiles
        discovered_pois = self.raw_env.tracker.discovered_pois

        # 1. Draw tiles (Floor and Walls)
        for x in range(grid_sz):
            for y in range(grid_sz):
                tx = ox + x * tile_sz
                ty = oy + y * tile_sz
                cell_rect = pygame.Rect(tx, ty, tile_sz, tile_sz)

                pos = (x, y)
                is_revealed = pos in revealed_tiles

                if self.show_fog and not is_revealed:
                    # Unrevealed fog of war
                    pygame.draw.rect(self.screen, COLOR_UNREVEALED, cell_rect)
                    pygame.draw.rect(self.screen, (25, 33, 50), cell_rect, width=1)
                else:
                    if pos in walkable_tiles:
                        color = COLOR_FLOOR_VISITED if pos in self.raw_env.tracker.visited_tiles else COLOR_FLOOR_REVEALED
                        pygame.draw.rect(self.screen, color, cell_rect)
                        pygame.draw.rect(self.screen, (100, 116, 139), cell_rect, width=1)
                    else:
                        # Wall
                        pygame.draw.rect(self.screen, COLOR_WALL, cell_rect)
                        pygame.draw.rect(self.screen, COLOR_WALL_OUTLINE, cell_rect, width=1)

        # 2. Draw Discovered POIs (Purple Diamond with star)
        for px, py in self.raw_env.poi_positions:
            if px < grid_sz and py < grid_sz and ((not self.show_fog) or ((px, py) in revealed_tiles)):
                cx = ox + px * tile_sz + tile_sz // 2
                cy = oy + py * tile_sz + tile_sz // 2
                r = tile_sz // 3
                pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
                pygame.draw.polygon(self.screen, COLOR_PURPLE, pts)
                if (px, py) in discovered_pois:
                    pygame.draw.circle(self.screen, (253, 224, 71), (cx, cy), max(2, r // 3))

        # 3. Draw Visited Trajectory line
        if len(self.trajectory) > 1:
            points = [
                (ox + x * tile_sz + tile_sz // 2, oy + y * tile_sz + tile_sz // 2)
                for x, y in self.trajectory
            ]
            pygame.draw.lines(self.screen, COLOR_PATH, False, points, width=max(2, tile_sz // 10))

        # 4. Draw Monsters
        for m in self.raw_env.monsters:
            mx, my = m.pos
            if (not self.show_fog) or ((mx, my) in revealed_tiles):
                mcx = ox + mx * tile_sz + tile_sz // 2
                mcy = oy + my * tile_sz + tile_sz // 2
                mr = tile_sz // 3
                # Draw monster circle with warning outline
                pygame.draw.circle(self.screen, COLOR_MONSTER, (mcx, mcy), mr)
                pygame.draw.circle(self.screen, (255, 255, 255), (mcx, mcy), mr, width=1)
                # State letter badge (P=Patrol, C=Chase)
                badge = self.font_mono.render(m.state[0].upper(), True, (255, 255, 255))
                self.screen.blit(badge, badge.get_rect(center=(mcx, mcy)))

        # 5. Draw Agent (Directional Triangle with cyan halo)
        ax, ay = self.raw_env.agent_pos
        acx = ox + ax * tile_sz + tile_sz // 2
        acy = oy + ay * tile_sz + tile_sz // 2
        ar = tile_sz // 2.5
        adir = self.raw_env.agent_dir

        # Triangles for 4 directions: 0=Right, 1=Down, 2=Left, 3=Up
        if adir == 0:  # Right
            tri = [(acx + ar, acy), (acx - ar, acy - ar * 0.7), (acx - ar, acy + ar * 0.7)]
        elif adir == 1:  # Down
            tri = [(acx, acy + ar), (acx - ar * 0.7, acy - ar), (acx + ar * 0.7, acy - ar)]
        elif adir == 2:  # Left
            tri = [(acx - ar, acy), (acx + ar, acy - ar * 0.7), (acx + ar, acy + ar * 0.7)]
        else:  # Up
            tri = [(acx, acy - ar), (acx - ar * 0.7, acy + ar), (acx + ar * 0.7, acy + ar)]

        pygame.draw.circle(self.screen, (6, 182, 212, 100), (acx, acy), int(ar * 1.3), width=2)
        pygame.draw.polygon(self.screen, COLOR_CYAN, tri)
        pygame.draw.polygon(self.screen, (255, 255, 255), tri, width=1)

        # Footer badge on map viewport
        mode_text = f"Difficulty: {self.difficulty.upper()} | Seed: {self.seed} | Grid: {grid_sz}x{grid_sz} | Fog: {'ON' if self.show_fog else 'OFF'} (Press F)"
        lbl = self.font_mono.render(mode_text, True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl, (vx + 15, vy + viewport_h - 22))

    def _render_sidebar(self) -> None:
        """Render the right dashboard with cards, buttons, and telemetry."""
        sx = 800
        sy = 20
        sw = 440
        sh = 800

        # Sidebar container panel
        panel_rect = pygame.Rect(sx - 10, sy, sw, sh)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=12)

        # Card 1: Title & Episode Status
        title_surf = self.font_title.render("MiniGrid Explorer", True, COLOR_CYAN_LIGHT)
        self.screen.blit(title_surf, (sx, sy + 12))

        # Status Badge
        status_msg = "EXPLORING"
        status_color = COLOR_CYAN
        end_reason = self.info.get("end_reason")
        if end_reason == "coverage":
            status_msg = "SUCCESS (TARGET REACHED)"
            status_color = COLOR_GREEN
        elif end_reason == "death":
            status_msg = "DEAD (SLAIN BY MONSTER)"
            status_color = COLOR_RED
        elif end_reason == "timeout":
            status_msg = "TIMEOUT (MAX STEPS)"
            status_color = COLOR_AMBER

        s_badge = self.font_header.render(f"● {status_msg}", True, status_color)
        self.screen.blit(s_badge, (sx, sy + 40))

        # Card 2: Environment Settings
        self._draw_section_header("1. Environment & Dungeon Layout", sx, 76)
        # Buttons are drawn automatically from self.buttons

        seed_lbl = self.font_mono.render(f"Current Seed: {self.seed}", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(seed_lbl, (sx, 148))

        # Card 3: Agent Paradigm Selection
        self._draw_section_header("2. Exploration Agent", sx, 218)

        # Card 4: Playback Controls
        self._draw_section_header("3. Simulation Playback", sx, 354)

        # Card 5: Real-time Telemetry Dashboard
        self._draw_section_header("4. Live Episode Telemetry", sx, 430)

        # Telemetry metrics
        cov_pct = self.info.get("revealed_coverage", 0.0) * 100.0
        vis_pct = self.info.get("visited_coverage", 0.0) * 100.0
        step_cur = self.raw_env.step_count if self.raw_env else 0
        step_max = self.raw_env.max_steps if self.raw_env else 250
        pois_found = self.info.get("discovered_pois", 0)
        pois_tot = self.info.get("total_pois", 0)

        # Step Progress
        self._draw_metric_bar(
            label=f"Step Progress: {step_cur}/{step_max}",
            ratio=step_cur / max(1, step_max),
            x=sx,
            y=460,
            width=sw - 30,
            fill_color=COLOR_AMBER,
        )

        # Revealed Coverage Progress (Target 95%)
        self._draw_metric_bar(
            label=f"Revealed Coverage: {cov_pct:.1f}% (Target: 95%)",
            ratio=cov_pct / 100.0,
            x=sx,
            y=505,
            width=sw - 30,
            fill_color=COLOR_GREEN,
            target_mark=0.95,
        )

        # Compact Metric Grid
        my = 550
        self._draw_stat_box("Visited Path", f"{vis_pct:.1f}%", sx, my, 125)
        self._draw_stat_box("POI Discovery", f"{pois_found}/{pois_tot}", sx + 140, my, 125)
        self._draw_stat_box("Total Reward", f"{self.last_reward:+.3f}", sx + 280, my, 130)

        # Reward Breakdown - split into 2 lines so no metrics are clipped
        my2 = 604
        bd = self.last_breakdown
        gains_text = (
            f"Step Reward: rev={bd.get('reward_reveal', 0):+.2f} | "
            f"poi={bd.get('reward_poi', 0):+.2f} | "
            f"step={bd.get('penalty_step', 0):+.3f}"
        )
        penalties_text = (
            f"Penalties:   bump={bd.get('penalty_bump', 0):+.2f} | "
            f"loop={bd.get('penalty_loop', 0):+.2f} | "
            f"death={bd.get('penalty_death', 0):+.2f}"
        )
        g_surf = self.font_mono.render(gains_text, True, COLOR_TEXT_MUTED)
        p_surf = self.font_mono.render(penalties_text, True, COLOR_TEXT_MUTED)
        self.screen.blit(g_surf, (sx, my2))
        self.screen.blit(p_surf, (sx, my2 + 18))

        # Monster Status
        my3 = 644
        m_info = []
        if self.raw_env:
            for m in self.raw_env.monsters:
                m_info.append(f"M{m.id}@{m.pos} [{m.state.upper()}]")
        m_str = " | ".join(m_info) if m_info else "None active"
        m_surf = self.font_mono.render(f"Monsters: {m_str}", True, (248, 113, 113))
        self.screen.blit(m_surf, (sx, my3))

        # Card 6: Keyboard Shortcuts Reference - 2 columns to prevent text overlap
        self._draw_section_header("Hotkeys Reference", sx, 672)
        hotkeys = [
            ("Space", "Play / Pause"),
            ("Tab", "Cycle agent"),
            ("S / Right", "Step 1 action"),
            ("R", "Reset episode"),
            ("W, A, D", "Manual move"),
            ("F", "Toggle Fog"),
            ("1, 2, 3", "Easy/Med/Hard"),
            ("N", "Next seed"),
            ("P", "Save Screenshot"),
            ("Esc / Q", "Quit viewer"),
        ]
        hy = 698
        col_w = 205
        for i, (key, desc) in enumerate(hotkeys):
            col = i % 2
            row = i // 2
            kx = sx + col * col_w
            ky = hy + row * 20
            k_tag = f"[{key}]"
            k_surf = self.font_mono.render(k_tag, True, COLOR_CYAN_LIGHT)
            d_surf = self.font_body.render(desc, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(k_surf, (kx, ky))
            self.screen.blit(d_surf, (kx + k_surf.get_width() + 6, ky - 1))

        # Draw all button widgets
        for btn in self.buttons:
            btn.draw(self.screen, self.font_btn)

    def _draw_section_header(self, title: str, x: int, y: int) -> None:
        """Render card sub-header."""
        lbl = self.font_header.render(title, True, (203, 213, 225))
        self.screen.blit(lbl, (x, y))
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (x, y + 18), (x + 410, y + 18), 1)

    def _draw_metric_bar(
        self,
        label: str,
        ratio: float,
        x: int,
        y: int,
        width: int,
        fill_color: tuple[int, int, int],
        target_mark: float | None = None,
    ) -> None:
        """Draw an animated-style percentage progress bar."""
        lbl = self.font_body.render(label, True, COLOR_TEXT_SECONDARY)
        self.screen.blit(lbl, (x, y))

        bar_rect = pygame.Rect(x, y + 18, width, 14)
        pygame.draw.rect(self.screen, (15, 23, 42), bar_rect, border_radius=4)

        fill_w = int(max(0.0, min(1.0, ratio)) * width)
        if fill_w > 0:
            fill_rect = pygame.Rect(x, y + 18, fill_w, 14)
            pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=4)

        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, bar_rect, width=1, border_radius=4)

        if target_mark is not None:
            tx = x + int(target_mark * width)
            pygame.draw.line(self.screen, (255, 255, 255), (tx, y + 16), (tx, y + 34), 2)

    def _draw_stat_box(self, label: str, value: str, x: int, y: int, width: int) -> None:
        """Render small statistical metric box."""
        box = pygame.Rect(x, y, width, 48)
        pygame.draw.rect(self.screen, (30, 41, 59), box, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, box, width=1, border_radius=6)

        lbl = self.font_mono.render(label, True, COLOR_TEXT_MUTED)
        val = self.font_title.render(value, True, COLOR_TEXT_PRIMARY)
        self.screen.blit(lbl, (x + 8, y + 5))
        self.screen.blit(val, (x + 8, y + 22))

    def save_screenshot(self, filepath: str) -> None:
        """Save high-resolution rendering of current UI state to disk."""
        self.render()
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(out_p))
        print(f"UI Screenshot saved to: {out_p.resolve()}")

    def run(self, max_steps: int | None = None) -> None:
        """Main application loop."""
        running = True
        step_count = 0

        while running:
            running = self.handle_events()

            if self.is_playing:
                self.step()
                step_count += 1
                if max_steps is not None and step_count >= max_steps:
                    self.is_playing = False

            self.render()
            self.clock.tick(self.fps)

        if not self.headless:
            pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Python UI Viewer for MiniGrid Explorer.")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--agent", type=str, default="frontier", choices=["manual", "frontier", "wall_follower", "random", "oracle"])
    parser.add_argument("--steps", type=int, default=None, help="Auto-run N steps then hold or exit")
    parser.add_argument("--headless", action="store_true", help="Run with dummy display driver")
    parser.add_argument("--screenshot-output", type=str, default=None, help="Save screenshot and exit")

    args = parser.parse_args()

    # Enable headless driver if requested
    if args.headless or args.screenshot_output:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    app = ExplorerUIApp(
        difficulty=args.difficulty,
        seed=args.seed,
        agent_type=args.agent,
        headless=(args.headless or args.screenshot_output is not None),
    )

    if args.screenshot_output:
        # Step a few actions to show interesting trajectory and fog-of-war
        num_preview_steps = args.steps if args.steps is not None else 35
        for _ in range(num_preview_steps):
            app.step()
        app.render()
        app.save_screenshot(args.screenshot_output)
    else:
        app.run(max_steps=args.steps)


if __name__ == "__main__":
    main()
