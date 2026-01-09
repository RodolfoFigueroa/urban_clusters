import argparse
import json
from pathlib import Path

import geopandas as gpd

from urban_clusters.cache import (
    get_hashes,
    get_or_load_all_jobs,
    get_or_load_hotspot_mask,
    get_or_load_point_clusters,
)
from urban_clusters.spatial import (
    generate_hulls,
)


def update_metadata(
    metadata_path: Path,
    args: argparse.Namespace,
    hashes: dict[str, str],
    points: gpd.GeoDataFrame,
) -> None:
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata[hashes["spatial"]] = {
        "method": args.method,
        "min_spatial_size": args.min_spatial_size,
        "epsilon": args.epsilon,
        "bounds_path": args.BOUNDS,
        "num_spatial_clusters": int(points["spatial_cluster"].dropna().nunique()),
    }

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=4)


def cluster_spatial() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "BOUNDS",
        type=str,
        help=(
            "Path to a geometry file (in GeoPackage or Shapefile format) "
            "containing the bounds to filter the employment points by."
        ),
    )
    parser.add_argument(
        "--method",
        type=int,
        choices=[1, 2],
        help=(
            "Clustering method to use: 1 (unweighted points), 2 (points "
            "weighted by number of jobs)."
        ),
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
    parser.add_argument(
        "--generate-hulls",
        action="store_true",
        help=(
            "If set, generate concave hulls for the spatial clusters after clustering."
        ),
    )
    parser.add_argument(
        "--hull-alpha",
        type=float,
        default=0.2,
        help=(
            "Alpha parameter for the alpha shape algorithm when "
            "generating concave hulls."
        ),
    )
    parser.add_argument(
        "--hull-buffer",
        type=float,
        default=20,
        help="Buffer distance to apply to the hulls when generating concave hulls.",
    )
    parser.add_argument(
        "--hull-group-by",
        type=str,
        default="spatial_cluster",
        help="Column name to group points by when generating concave hulls.",
    )

    args = parser.parse_args()
    cache_path = Path(args.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    all_jobs = get_or_load_all_jobs(cache_path)
    df_bounds = gpd.read_file(args.BOUNDS).to_crs("EPSG:6372")
    df_points = gpd.sjoin(
        all_jobs,
        df_bounds,
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])

    hashes = get_hashes(df_points, vars(args))

    hotspot_cache_path = cache_path / f"{hashes['hotspots']}.npy"
    cluster_cache_path = cache_path / hashes["spatial"] / "points.gpkg"
    cluster_cache_path.parent.mkdir(parents=True, exist_ok=True)

    hotspot_mask = get_or_load_hotspot_mask(df_points, hotspot_cache_path)
    df_points = get_or_load_point_clusters(
        df_points,
        vars(args),
        hotspot_mask,
        cluster_cache_path,
    )

    if args.generate_hulls:
        hulls = generate_hulls(
            df_points,
            cluster_col=args.hull_group_by,
            alpha=args.hull_alpha,
            buffer=args.hull_buffer,
        )
        hulls.to_file(cache_path / hashes["spatial"] / "hulls.gpkg")

    update_metadata(cache_path / "metadata.json", args, hashes, df_points)
