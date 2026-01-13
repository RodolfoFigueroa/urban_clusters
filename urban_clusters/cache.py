import hashlib
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS

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
    return hashlib.sha256("".join(points["hash"].tolist()).encode()).hexdigest()[
        :HASH_LENGTH
    ]


def hash_polygon(
    polygon: shapely.geometry.Polygon,
    *,
    crs: CRS,
    precision: float = 10,
) -> str:
    """Create a hash from a Polygon geometry."""
    x, y = polygon.exterior.coords.xy
    df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=crs)
    return hash_point_gdf(df, precision=precision)


def hash_polygon_gdf(polygon: gpd.GeoDataFrame, *, precision: float = 10) -> str:
    if len(polygon) != 1:
        err = "hash_polygon_gdf only supports GeoDataFrames with a single polygon."
        raise ValueError(err)

    crs = polygon.crs
    if crs is None:
        err = "Input GeoDataFrame must have a defined CRS."
        raise ValueError(err)

    return hash_polygon(polygon.iloc[0].geometry, crs=crs, precision=precision)


def hash_spatial_args(args: dict, points_hash: str) -> str:
    return hashlib.sha256(
        (
            str(abs(hash(args["method"])))
            + str(abs(hash(args["min_spatial_size"])))
            + str(abs(hash(args["epsilon"])))
            + points_hash
        ).encode(),
    ).hexdigest()[:HASH_LENGTH]


def get_or_load_filtered_points(
    bounds: gpd.GeoDataFrame,
    *,
    cache_path: Path,
) -> gpd.GeoDataFrame:
    bounds_hash = hash_polygon_gdf(bounds)
    filtered_points_cache_path = cache_path / f"{bounds_hash}.gpkg"

    if filtered_points_cache_path.exists():
        filtered_points = gpd.read_file(filtered_points_cache_path)
    else:
        all_points = get_or_load_all_jobs(cache_path)
        filtered_points = gpd.sjoin(
            all_points,
            bounds,
            how="inner",
            predicate="within",
        ).drop(columns=["index_right"])
        filtered_points.to_file(filtered_points_cache_path)
    return filtered_points


def get_or_load_hotspot_mask(
    points: gpd.GeoDataFrame,
    *,
    cache_path: Path,
) -> np.ndarray:
    points_hash = hash_point_gdf(points)[:HASH_LENGTH]
    hotspot_cache_path = cache_path / f"{points_hash}.npy"

    if hotspot_cache_path.exists():
        hotspot_mask = np.load(hotspot_cache_path)
    else:
        hotspot_mask = get_hotspot_mask(points, distance_band=500)
        np.save(hotspot_cache_path, hotspot_mask)
    return hotspot_mask


def get_or_load_point_clusters(
    points: gpd.GeoDataFrame,
    args: dict,
    hotspot_mask: np.ndarray,
    *,
    cache_path: Path,
) -> gpd.GeoDataFrame:
    points_hash = hash_point_gdf(points)[:HASH_LENGTH]
    spatial_hash = hash_spatial_args(args, points_hash)
    cluster_cache_path = cache_path / f"{spatial_hash}.gpkg"

    if cluster_cache_path.exists():
        out = gpd.read_file(cluster_cache_path)
    else:
        if args["method"] == 1:
            clusters = cluster_points(
                points,
                hdbscan_params={
                    "min_cluster_size": args["min_spatial_size"],
                    "cluster_selection_epsilon": args["epsilon"],
                },
            )
        elif args["method"] == 2:
            clusters = cluster_points_weighted(
                points,
                hotspot_mask=hotspot_mask,
                hdbscan_params={
                    "min_cluster_size": args["min_spatial_size"],
                    "cluster_selection_epsilon": args["epsilon"],
                },
            )
        else:
            err = f"Unknown method: {args['method']}"
            raise ValueError(err)

        out = points.assign(spatial_cluster=clusters)
        out.to_file(cluster_cache_path)
    return out


def get_or_load_all_jobs(cache_path: Path) -> gpd.GeoDataFrame:
    jobs_path = Path(os.environ["JOBS_PATH"])
    all_jobs_cache_path = cache_path / "all_jobs.gpkg"
    if all_jobs_cache_path.exists():
        all_jobs = gpd.read_file(all_jobs_cache_path)
    else:
        all_jobs = (
            pd.read_csv(
                jobs_path / "denue_2023_estimaciones.csv",
                usecols=["latitud", "longitud", "num_empleos_esperados", "codigo_act"],
            )
            .assign(
                geometry=lambda df: gpd.points_from_xy(df["longitud"], df["latitud"]),
            )
            .drop(columns=["latitud", "longitud"])
            .pipe(gpd.GeoDataFrame, geometry="geometry", crs="EPSG:4326")
            .to_crs("EPSG:6372")
            .rename(columns={"num_empleos_esperados": "jobs", "codigo_act": "scian"})
        )
        all_jobs.to_file(all_jobs_cache_path)
    return all_jobs
