import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np

from urban_clusters.spatial import (
    cluster_points,
    cluster_points_weighted,
    get_hotspot_mask,
)

HASH_LENGTH = 8


def hash_dict(h: dict, *, length: int = 8) -> str:
    """Create a hash from a dictionary."""
    h_str = json.dumps(h, sort_keys=True)
    out = str(abs(hash(h_str)))
    return out[:length]


def hash_point_gdf(points: gpd.GeoDataFrame, *, precision: float = 10) -> str:
    """Create a hash from a GeoSeries of points."""
    points = points.assign(
        x_rounded=lambda df: df["geometry"].x.div(precision).round().astype(int),
        y_rounded=lambda df: df["geometry"].y.div(precision).round().astype(int),
        x_hash=lambda df: df["x_rounded"].apply(lambda x: str(abs(hash(x)))),
        y_hash=lambda df: df["y_rounded"].apply(lambda y: str(abs(hash(y)))),
        hash=lambda df: df["x_hash"] + df["y_hash"],
    ).sort_values(by=["x_rounded", "y_rounded"])
    return hashlib.md5("".join(points["hash"].tolist()).encode()).hexdigest()[
        :HASH_LENGTH
    ]


def get_spatial_hash(args: argparse.Namespace, points_hash: str) -> str:
    return hashlib.md5(
        (
            str(abs(hash(args.method)))
            + str(abs(hash(args.min_spatial_size)))
            + str(abs(hash(args.epsilon)))
            + points_hash
        ).encode(),
    ).hexdigest()[:HASH_LENGTH]


def get_or_load_hotspot_mask(
    points: gpd.GeoDataFrame,
    hotspot_cache_path: Path,
) -> np.ndarray:
    if hotspot_cache_path.exists():
        hotspot_mask = np.load(hotspot_cache_path)
    else:
        hotspot_mask = get_hotspot_mask(points, distance_band=500)
        np.save(hotspot_cache_path, hotspot_mask)
    return hotspot_mask


def get_hashes(points: gpd.GeoDataFrame, args: argparse.Namespace) -> dict[str, str]:
    hashes = {"hotspots": hash_point_gdf(points)[:HASH_LENGTH]}
    hashes["spatial"] = get_spatial_hash(args, hashes["hotspots"])
    return hashes


def get_or_load_point_clusters(
    points: gpd.GeoDataFrame,
    args: argparse.Namespace,
    hotspot_mask: np.ndarray,
    cluster_cache_path: Path,
) -> gpd.GeoDataFrame:
    if cluster_cache_path.exists():
        points = gpd.read_file(cluster_cache_path)
    else:
        if args.method == 1:
            clusters = cluster_points(
                points,
                hdbscan_params={
                    "min_cluster_size": args.min_spatial_size,
                    "cluster_selection_epsilon": args.epsilon,
                },
            )
        elif args.method == 2:
            clusters = cluster_points_weighted(
                points,
                hotspot_mask=hotspot_mask,
                hdbscan_params={
                    "min_cluster_size": args.min_spatial_size,
                    "cluster_selection_epsilon": args.epsilon,
                },
            )
        else:
            err = f"Unknown method: {args.method}"
            raise ValueError(err)

        points = points.assign(spatial_cluster=clusters)
        points.to_file(cluster_cache_path)
    return points


def update_metadata(
    metadata_path: Path,
    args: argparse.Namespace,
    hashes: dict[str, str],
    points: gpd.GeoDataFrame
) -> None:
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata[hashes["spatial"]] = {
        "method": args.method,
        "min_spatial_size": args.min_spatial_size,
        "epsilon": args.epsilon,
        "points_path": args.POINTS,
        "num_spatial_clusters": int(points["spatial_cluster"].dropna().nunique()),
    }

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=4)


def cluster_spatial():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "POINTS",
        type=str,
        help="Path to a geometry file containing the employment points.",
    )
    parser.add_argument(
        "--method",
        type=int,
        choices=[1, 2],
        help="Clustering method to use: 1 (unweighted points), 2 (points weighted by number of jobs).",
        required=True,
    )
    parser.add_argument(
        "--min-spatial-size",
        type=int,
        help="Minimum number of points the spatial clusters must have.",
        required=True,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Epsilon parameter for the DBSCAN algorithm (in meters).",
        required=True,
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="./cache",
        help="Path to cache directory.",
    )

    args = parser.parse_args()
    cache_path = Path(args.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    df_points = gpd.read_file(args.POINTS)

    hashes = get_hashes(df_points, args)

    hotspot_cache_path = cache_path / f"{hashes['hotspots']}.npy"
    cluster_cache_path = cache_path / f"{hashes['spatial']}.gpkg"

    hotspot_mask = get_or_load_hotspot_mask(df_points, hotspot_cache_path)
    df_points = get_or_load_point_clusters(
        df_points,
        args,
        hotspot_mask,
        cluster_cache_path,
    )

    update_metadata(cache_path / "metadata.json", args, hashes, df_points)