import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely


def load_jobs_dataframe() -> gpd.GeoDataFrame:
    df = (
        pd.read_csv(
            Path(os.environ["JOBS_PATH"]) / "denue_2023_estimaciones.csv",
            encoding="latin1",
            usecols=["latitud", "longitud", "num_empleos_esperados", "codigo_act"],
        )
        .assign(geometry=lambda df: gpd.points_from_xy(df["longitud"], df["latitud"]))
        .drop(columns=["latitud", "longitud"])
    )
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(
        "EPSG:6372",
    )


def filter_jobs_by_geometry(
    df_jobs: gpd.GeoDataFrame,
    geom: shapely.geometry.Polygon,
) -> gpd.GeoDataFrame:
    return df_jobs[df_jobs.intersects(geom)].copy()
