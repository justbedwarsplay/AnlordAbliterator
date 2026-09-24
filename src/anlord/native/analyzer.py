# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Residual space analysis for the native abliteration pipeline.

Provides:
- `print_residual_geometry()`: per-layer table of cosine similarities, vector
  norms, and silhouette coefficients for the good/bad residual clusters
  (using means and geometric medians). The full table is also saved next to
  the residual plots as text and JSON.
- `plot_residuals()`: per-layer PaCMAP projections of the residual vectors,
  rendered to PNG frames and combined into an animated GIF.

Both features require the optional research dependencies (geom-median,
scikit-learn, pacmap, imageio, matplotlib), installable via the `research`
extra of this package.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.linalg as LA
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.console import Console
from rich.progress import track
from rich.table import Table
from torch import Tensor

from .config import NativeConfig
from .model import Model
from .utils import print

_GEOMETRY_LEGEND = [
    "[bold]g[/] = mean of residual vectors for good prompts",
    "[bold]g*[/] = geometric median of residual vectors for good prompts",
    "[bold]b[/] = mean of residual vectors for bad prompts",
    "[bold]b*[/] = geometric median of residual vectors for bad prompts",
    "[bold]r[/] = residual direction for means (i.e., [bold]b - g[/])",
    "[bold]r*[/] = residual direction for geometric medians (i.e., [bold]b* - g*[/])",
    "[bold]S(x,y)[/] = cosine similarity of [bold]x[/] and [bold]y[/]",
    "[bold]|x|[/] = L2 norm of [bold]x[/]",
    "[bold]Silh[/] = Mean silhouette coefficient of residuals for good/bad clusters",
]


class ResidualAnalyzer:
    def __init__(
        self,
        settings: NativeConfig,
        model: Model,
        good_residuals: Tensor,
        bad_residuals: Tensor,
    ):
        self.settings = settings
        self.model = model
        self.good_residuals = good_residuals
        self.bad_residuals = bad_residuals

    def print_residual_geometry(self):
        try:
            from geom_median.torch import compute_geometric_median
            from sklearn.metrics import silhouette_score
        except ImportError:
            print()
            print(
                (
                    "[red]Research dependencies not found. Printing residual geometry requires "
                    "installing Anlord Abliterator with the optional research feature, i.e., "
                    "using \"pip install -U 'anlord\[research]'\".[/]"
                )
            )
            return

        print()
        print("Computing residual geometry...")

        table = Table()
        table.add_column("Layer", justify="right")
        table.add_column("S(g,b)", justify="right")
        table.add_column("S(g*,b*)", justify="right")
        table.add_column("S(g,r)", justify="right")
        table.add_column("S(g*,r*)", justify="right")
        table.add_column("S(b,r)", justify="right")
        table.add_column("S(b*,r*)", justify="right")
        table.add_column("|g|", justify="right")
        table.add_column("|g*|", justify="right")
        table.add_column("|b|", justify="right")
        table.add_column("|b*|", justify="right")
        table.add_column("|r|", justify="right")
        table.add_column("|r*|", justify="right")
        table.add_column("Silh", justify="right")

        g = self.good_residuals.mean(dim=0)
        g_star = torch.stack(
            [
                compute_geometric_median(
                    self.good_residuals[:, layer_index, :].detach().cpu()
                ).median
                for layer_index in range(len(self.model.get_layers()) + 1)
            ]
        )
        b = self.bad_residuals.mean(dim=0)
        b_star = torch.stack(
            [
                compute_geometric_median(
                    self.bad_residuals[:, layer_index, :].detach().cpu()
                ).median
                for layer_index in range(len(self.model.get_layers()) + 1)
            ]
        )
        r = b - g
        r_star = b_star - g_star

        g_b_similarities = F.cosine_similarity(g, b, dim=-1)
        g_star_b_star_similarities = F.cosine_similarity(g_star, b_star, dim=-1)
        g_r_similarities = F.cosine_similarity(g, r, dim=-1)
        g_star_r_star_similarities = F.cosine_similarity(g_star, r_star, dim=-1)
        b_r_similarities = F.cosine_similarity(b, r, dim=-1)
        b_star_r_star_similarities = F.cosine_similarity(b_star, r_star, dim=-1)

        g_norms = LA.vector_norm(g, dim=-1)
        g_star_norms = LA.vector_norm(g_star, dim=-1)
        b_norms = LA.vector_norm(b, dim=-1)
        b_star_norms = LA.vector_norm(b_star, dim=-1)
        r_norms = LA.vector_norm(r, dim=-1)
        r_star_norms = LA.vector_norm(r_star, dim=-1)

        residuals = (
            torch.cat(
                [
                    self.good_residuals,
                    self.bad_residuals,
                ],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        labels = [0] * len(self.good_residuals) + [1] * len(self.bad_residuals)
        silhouettes = [
            silhouette_score(residuals[:, layer_index, :], labels)
            for layer_index in range(len(self.model.get_layers()) + 1)
        ]

        for layer_index in range(1, len(self.model.get_layers()) + 1):
            table.add_row(
                f"{layer_index}",
                f"{g_b_similarities[layer_index].item():.4f}",
                f"{g_star_b_star_similarities[layer_index].item():.4f}",
                f"{g_r_similarities[layer_index].item():.4f}",
                f"{g_star_r_star_similarities[layer_index].item():.4f}",
                f"{b_r_similarities[layer_index].item():.4f}",
                f"{b_star_r_star_similarities[layer_index].item():.4f}",
                f"{g_norms[layer_index].item():.2f}",
                f"{g_star_norms[layer_index].item():.2f}",
                f"{b_norms[layer_index].item():.2f}",
                f"{b_star_norms[layer_index].item():.2f}",
                f"{r_norms[layer_index].item():.2f}",
                f"{r_star_norms[layer_index].item():.2f}",
                f"{silhouettes[layer_index]:.4f}",
            )

        print()
        print("[bold]Residual Geometry[/]")
        print(table)
        for line in _GEOMETRY_LEGEND:
            print(line)

        # The 14-column table is unreadable in a narrow terminal, so save the
        # full rendering next to the residual plots, plus a machine-readable
        # copy of the numbers.
        try:
            base_path = self._residual_output_base_path()

            file_console = Console(file=io.StringIO(), width=220, highlight=False)
            file_console.print("[bold]Residual Geometry[/]")
            file_console.print(table)
            for line in _GEOMETRY_LEGEND:
                file_console.print(line)
            text_path = base_path / "residual_geometry.txt"
            text_path.write_text(file_console.file.getvalue(), encoding="utf-8")

            geometry = {
                "model": self.settings.model,
                "layers": [
                    {
                        "layer": layer_index,
                        "S(g,b)": g_b_similarities[layer_index].item(),
                        "S(g*,b*)": g_star_b_star_similarities[layer_index].item(),
                        "S(g,r)": g_r_similarities[layer_index].item(),
                        "S(g*,r*)": g_star_r_star_similarities[layer_index].item(),
                        "S(b,r)": b_r_similarities[layer_index].item(),
                        "S(b*,r*)": b_star_r_star_similarities[layer_index].item(),
                        "|g|": g_norms[layer_index].item(),
                        "|g*|": g_star_norms[layer_index].item(),
                        "|b|": b_norms[layer_index].item(),
                        "|b*|": b_star_norms[layer_index].item(),
                        "|r|": r_norms[layer_index].item(),
                        "|r*|": r_star_norms[layer_index].item(),
                        "Silh": float(silhouettes[layer_index]),
                    }
                    for layer_index in range(1, len(self.model.get_layers()) + 1)
                ],
            }
            json_path = base_path / "residual_geometry.json"
            json_path.write_text(json.dumps(geometry, indent=2), encoding="utf-8")

            print(f"* Residual geometry saved to [bold]{text_path}[/]")
        except Exception as e:
            print(f"[yellow]Could not save residual geometry: {e}[/]")

    def _residual_output_base_path(self) -> Path:
        """Base directory for residual plots and analysis artifacts."""
        base_path = Path(
            self.settings.residual_plot_path
        ) / self.settings.model.replace(
            "/",
            "_",
        ).replace(
            "\\",
            "_",
        )
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    def plot_residuals(self):
        try:
            import imageio.v3 as iio
            import matplotlib.pyplot as plt
            from geom_median.numpy import compute_geometric_median
            from pacmap import PaCMAP
        except ImportError:
            print()
            print(
                (
                    "[red]Research dependencies not found. Plotting residuals requires "
                    "installing Anlord Abliterator with the optional research feature, i.e., "
                    "using \"pip install -U 'anlord\[research]'\".[/]"
                )
            )
            return

        LAYER_FRAME_DURATION = 1000
        N_TRANSITION_FRAMES = 20
        TRANSITION_FRAME_DURATION = 50

        print()
        print("Plotting residual vectors...")

        layer_residuals_2d = []
        pacmap_init = None

        for layer_index in track(
            range(1, len(self.model.get_layers()) + 1),
            description="* Computing PaCMAP projections...",
        ):
            good_residuals = (
                self.good_residuals[:, layer_index, :].detach().cpu().numpy()
            )
            bad_residuals = self.bad_residuals[:, layer_index, :].detach().cpu().numpy()

            residuals = np.vstack((good_residuals, bad_residuals))
            embedding = PaCMAP(n_components=2, n_neighbors=min(30, len(residuals) - 1))
            residuals_2d = embedding.fit_transform(residuals, init=pacmap_init)
            pacmap_init = residuals_2d

            n_good_residuals = good_residuals.shape[0]
            good_residuals_2d = residuals_2d[:n_good_residuals]
            bad_residuals_2d = residuals_2d[n_good_residuals:]

            # Important: These are the medians of the 2D-projected residuals,
            #            not the projections of the medians of the residuals.
            #            Their only purpose is to rotate the individual plots
            #            into a consistent orientation. They are not suitable
            #            for being plotted themselves.
            good_anchor = compute_geometric_median(good_residuals_2d).median
            bad_anchor = compute_geometric_median(bad_residuals_2d).median

            # Rotate points to make the line connecting the medians horizontal,
            # with the median of the good residuals on the left.
            direction = bad_anchor - good_anchor
            angle = -np.arctan2(direction[1], direction[0])
            cosine = np.cos(angle)
            sine = np.sin(angle)
            rotation_matrix = np.array([[cosine, -sine], [sine, cosine]])
            residuals_2d = residuals_2d @ rotation_matrix.T

            good_residuals_2d = residuals_2d[:n_good_residuals]
            bad_residuals_2d = residuals_2d[n_good_residuals:]

            layer_residuals_2d.append((good_residuals_2d, bad_residuals_2d))

        plt.style.use(self.settings.residual_plot_style)

        def plot(
            image_path: Path,
            layer_index: int,
            good_residuals_2d: NDArray,
            bad_residuals_2d: NDArray,
        ):
            fig, ax = plt.subplots(figsize=(8, 6))

            ax.scatter(
                good_residuals_2d[:, 0],
                good_residuals_2d[:, 1],
                s=10,
                c=self.settings.good_prompts.residual_plot_color,
                alpha=0.5,
                label=self.settings.good_prompts.residual_plot_label,
            )
            ax.scatter(
                bad_residuals_2d[:, 0],
                bad_residuals_2d[:, 1],
                s=10,
                c=self.settings.bad_prompts.residual_plot_color,
                alpha=0.5,
                label=self.settings.bad_prompts.residual_plot_label,
            )

            ax.set_title(self.settings.residual_plot_title, pad=11)
            ax.legend(loc="upper right")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

            fig.text(
                0.018,
                0.02,
                self.settings.model,
                ha="left",
                va="bottom",
                fontsize=12,
            )
            fig.text(
                0.982,
                0.02,
                f"Layer {layer_index:03}",
                ha="right",
                va="bottom",
                fontsize=12,
            )

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.08)

            fig.savefig(image_path, dpi=100)
            plt.close(fig)

        base_path = self._residual_output_base_path()

        images = []
        durations = []

        for layer_index, (
            good_residuals_2d,
            bad_residuals_2d,
        ) in enumerate(
            track(
                layer_residuals_2d,
                description="* Generating plots...",
            ),
            1,
        ):
            image_path = base_path / f"layer_{layer_index:03}.png"

            plot(image_path, layer_index, good_residuals_2d, bad_residuals_2d)

            images.append(iio.imread(image_path))
            durations.append(LAYER_FRAME_DURATION)

            if layer_index < len(layer_residuals_2d):
                # The first frame of the transition is the layer frame created above.
                # The last frame is the next layer frame, created in the next iteration
                # of the outer loop. The following are the intermediate frames.
                # There are a total of N_TRANSITION_FRAMES frame changes in the transition.
                for frame_index in range(1, N_TRANSITION_FRAMES):
                    image_path = (
                        base_path / f"layer_{layer_index:03}_frame_{frame_index:03}.png"
                    )

                    progress = frame_index / N_TRANSITION_FRAMES

                    good_residuals_2d_interpolated = good_residuals_2d + progress * (
                        layer_residuals_2d[layer_index][0] - good_residuals_2d
                    )
                    bad_residuals_2d_interpolated = bad_residuals_2d + progress * (
                        layer_residuals_2d[layer_index][1] - bad_residuals_2d
                    )

                    plot(
                        image_path,
                        layer_index,
                        good_residuals_2d_interpolated,
                        bad_residuals_2d_interpolated,
                    )

                    images.append(iio.imread(image_path))
                    durations.append(TRANSITION_FRAME_DURATION)

                    # Delete the image file containing the animation frame.
                    # We have already read its contents and it serves no purpose
                    # other than building the animation.
                    image_path.unlink()

        print("* Generating animation...")

        iio.imwrite(
            base_path / "animation.gif",
            images,
            duration=durations,
            loop=0,
        )

        print(f"* Plots saved to [bold]{base_path.resolve()}[/].")
