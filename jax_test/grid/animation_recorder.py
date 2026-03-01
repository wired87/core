"""
Grid animation recorder: saves plot per time step, generates GIF at sim end,
stores animation path in envs table.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class GridAnimationRecorder:
    """
    Saves a plot per time step, generates GIF at sim end, saves path to envs table.
    """

    def __init__(
        self,
        env_id: str,
        user_id: str,
        env_cfg: dict[str, Any],
        cfg: dict[str, Any],
        env_manager,
        *,
        out_dir: str | None = None,
        fps: int = 10,
    ):
        self.env_id = env_id
        self.user_id = user_id
        self.env_cfg = env_cfg
        self.cfg = cfg
        self.env_manager = env_manager
        self.fps = fps
        self._out_dir = Path(out_dir or tempfile.mkdtemp(prefix="grid_anim_"))
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._frame_paths: list[str] = []
        self._visualizer = None

    def _get_visualizer(self):
        if self._visualizer is None:
            from grid.visualizer import ModularVisualizer

            self._visualizer = ModularVisualizer(self.env_cfg, self.cfg)
        return self._visualizer

    def save_frame(self, step: int, data: np.ndarray) -> None:
        """Save plot for this time step as PNG."""
        if not _HAS_PIL:
            return
        try:
            viz = self._get_visualizer()
            fig = viz.plot_time_step([data], time_idx=0)
            path = self._out_dir / f"frame_{step:06d}.png"
            fig.figure.savefig(str(path), facecolor="white", bbox_inches="tight")
            fig.figure.clf()
            import matplotlib.pyplot as plt

            plt.close(fig.figure)
            self._frame_paths.append(str(path))
        except Exception as e:
            print(f"[GridAnimationRecorder] save_frame error: {e}")

    def finish(self) -> str | None:
        """
        Create GIF from saved frames, save to envs table, return path.
        Returns None if no frames or error.
        """
        if not _HAS_PIL or not self._frame_paths:
            return None
        try:
            frames = [Image.open(p) for p in self._frame_paths]
            if not frames:
                return None
            duration_ms = int(1000 / self.fps) if self.fps > 0 else 100
            animation_path = self._out_dir / f"{self.env_id}_animation.gif"
            frames[0].save(
                str(animation_path),
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                disposal=0,
            )
            for f in frames:
                f.close()
            path_str = str(animation_path)
            self._save_to_envs(path_str)
            return path_str
        except Exception as e:
            print(f"[GridAnimationRecorder] finish error: {e}")
            return None
        finally:
            for p in self._frame_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

    def _save_to_envs(self, animation_path: str) -> None:
        """Update envs table row with animation_path."""
        try:
            if hasattr(self.env_manager.qb, "insert_col"):
                self.env_manager.qb.insert_col(
                    self.env_manager.TABLE_ID,
                    "animation_path",
                    "STRING",
                )
        except Exception:
            pass
        try:
            self.env_manager.qb.set_item(
                self.env_manager.TABLE_ID,
                {"animation_path": animation_path},
                keys={"id": self.env_id, "user_id": self.user_id},
            )
            print(f"[GridAnimationRecorder] saved animation_path to envs: {animation_path}")
        except Exception as e:
            print(f"[GridAnimationRecorder] _save_to_envs error: {e}")
