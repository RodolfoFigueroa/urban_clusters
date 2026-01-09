import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from photutils.utils import make_random_cmap
import matplotlib.colors as mcol

from urban_clusters.constants import (
    SECTOR_NAME_MAP,
    SECTOR_NAME_MAP_SHORT,
    SUBSECTOR_NAME_MAP,
    SUPERSECTOR_NAME_MAP,
)
from urban_clusters.economic import (
    get_centroid_df,
    get_distribution_df,
    get_location_quotient_df,
)

LEVEL_NAME_MAP = {
    1: SUPERSECTOR_NAME_MAP,
    2: SECTOR_NAME_MAP,
    3: SUBSECTOR_NAME_MAP,
}


def plot_heatmap(
    df: pd.DataFrame,
    *,
    ax: Axes,
    heatmap_kws: dict | None = None,
    cbar_title: str = "",
) -> ColorbarBase:
    if heatmap_kws is None:
        heatmap_kws = {}

    heatmap = sns.heatmap(
        df,
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": cbar_title},
        **heatmap_kws,
    )
    cbar = heatmap.collections[0].colorbar

    if cbar is None:
        err = "No colorbar found in the heatmap."
        raise TypeError(err)

    ax.set_xlabel("Cluster económico")
    ax.set_ylabel("")

    return cbar


def plot_cluster_distribution(
    points: gpd.GeoDataFrame,
    *,
    ax: Axes,
    level: int = 2,
    heatmap_kws: dict | None = None,
) -> None:
    out = get_distribution_df(
        points,
        normalize=True,
        level=level,
        cluster_col="economic_cluster",
    ).transpose()

    out.index = out.index.map(LEVEL_NAME_MAP[level])
    cbar = plot_heatmap(
        out,
        ax=ax,
        cbar_title="Fracción de empleos dentro del cluster",
        heatmap_kws=heatmap_kws,
    )
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))


def plot_location_quotient(
    points: gpd.GeoDataFrame,
    *,
    ax: Axes,
    level: int = 2,
    heatmap_kws: dict | None = None,
) -> None:
    df_lq = get_location_quotient_df(points, level=level).transpose()
    df_lq.index = df_lq.index.map(LEVEL_NAME_MAP[level])
    plot_heatmap(
        df_lq,
        ax=ax,
        cbar_title="Cociente de ubicación",
        heatmap_kws=heatmap_kws,
    )


def plot_num_jobs(
    points: gpd.GeoDataFrame,
    *,
    ax: Axes,
    bar_kws: dict | None = None,
) -> None:
    if bar_kws is None:
        bar_kws = {}

    temp = (
        points.dropna(subset=["economic_cluster"])
        .assign(economic_cluster=lambda df: df["economic_cluster"].astype(int))
        .groupby("economic_cluster")["jobs"]
        .sum()
    )
    temp.plot.bar(ax=ax, **bar_kws)
    ax.bar_label(ax.containers[0], fmt="{:,.0f}")  # type: ignore[reportArgumentType]
    ax.set_xlabel("Cluster económico")
    ax.set_ylabel("Número de empleos esperados")
    ax.yaxis.set_major_formatter("{x:,.0f}")


def plot_circular_hist(
    row: pd.Series,
    *,
    ax: PolarAxes,
    annotate_top: bool = False,
) -> None:
    ax.set_theta_zero_location("N")
    spoke_labels = row.index.tolist()
    data = row.to_numpy()

    n = len(spoke_labels)
    bottom = 0.05

    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    theta_map = dict(zip(spoke_labels, theta, strict=True))
    width = 2 * np.pi / n - 0.01

    ax.set_xticks(theta)
    ax.set_xticklabels([])

    bars = ax.bar(
        theta,
        data,
        width=width,
        bottom=bottom,
        fc="C0",
        alpha=0.8,
        ec="k",
        lw=1.5,
    )

    if annotate_top:
        top_values = row.sort_values(ascending=False).head(3)
        texts = []
        for idx, value in top_values.items():
            if not isinstance(idx, int):
                err = "Index must be an integer."
                raise TypeError(err)

            text = ax.text(
                theta_map[idx],
                value,
                SECTOR_NAME_MAP_SHORT[idx],
                fontsize=10,
                weight="bold",
            )
            texts.append(text)
        adjust_text(
            texts,
            ax=ax,
            objects=bars,
            ensure_inside_axes=False,
            arrowprops={"arrowstyle": "->", "color": "red"},
        )


def plot_centroids(
    points: gpd.GeoDataFrame,
    *,
    level: int = 2,
    figsize: tuple[float, float] | None = None,
    annotate_top: bool = False,
) -> tuple[Figure, np.ndarray]:
    df_centroids = get_centroid_df(points, level=level)

    num_rows = (len(df_centroids) + 1) // 2
    if figsize is None:
        figsize = (8, 10 / 3 * num_rows)

    fig, axes = plt.subplots(num_rows, 2, figsize=figsize, subplot_kw={"polar": True})

    # Fix if number of rows is one
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for centroid_idx, row in df_centroids.iterrows():
        if not isinstance(centroid_idx, int):
            err = "Centroid index must be an integer."
            raise TypeError(err)

        plot_circular_hist(
            row,
            ax=axes[centroid_idx // 2, centroid_idx % 2],
            annotate_top=annotate_top,
        )
        axes[centroid_idx // 2, centroid_idx % 2].set_title(
            f"Cluster {centroid_idx}",
            fontsize=16,
            weight="bold",
        )

    if len(df_centroids) % 2 != 0:
        axes[-1, -1].set_visible(False)
    return fig, axes


def generate_unique_colors(labels: pd.Series) -> pd.Series:
    cmap = make_random_cmap(labels.nunique(dropna=True), seed=42)
    norm = mcol.Normalize(vmin=labels.min(), vmax=labels.max())
    return labels.map(lambda x: mcol.rgb2hex(cmap(norm(x))) if pd.notna(x) else "#a9a9a9")