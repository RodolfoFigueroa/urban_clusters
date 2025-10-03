import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from esda import G_Local
from hdbscan import HDBSCAN
from libpysal.weights import DistanceBand

from urban_clusters.constants import SECTOR_NAME_MAP, SUBSECTOR_NAME_MAP

SIGNIFICANCE_LEVEL = 0.05


def load_denue(denue_path: os.PathLike) -> gpd.GeoDataFrame:
    """Load a DENUE geographic file for a specific region.

    Calculates the sector and subsector attributes, and reprojects the data
    to the appropriate UTM coordinate reference system.

    Parameters
    ----------
    denue_path : os.PathLike
        Path to a DENUE geographic file.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the DENUE data, including sector and subsector
        attributes, and reprojected to the appropriate UTM CRS.

    """
    df = (
        gpd.read_file(denue_path)
        .assign(
            sector=lambda df: (
                df["codigo_act"].astype(str).str[:2].astype(int).map(SECTOR_NAME_MAP)
            ),
            subsector=lambda df: (
                df["codigo_act"].astype(str).str[:3].astype(int).map(SUBSECTOR_NAME_MAP)
            ),
        )
        .filter(
            ["geometry", "num_empleos_esperados", "sector", "subsector", "codigo_act"],
        )
        .rename(columns={"num_empleos_esperados": "jobs", "codigo_act": "scian"})
    )
    return df.to_crs(df.estimate_utm_crs())


def cluster_points(points: gpd.GeoDataFrame, *, hdbscan_params: dict) -> pd.Series:
    """Cluster points using HDBSCAN.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        A GeoDataFrame with point geometries.

    hdbscan_params : dict
        A dictionary with the parameters to pass to the HDBSCAN model.
        See the HDBSCAN documentation for more details:
        https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan

    Returns
    -------
    pd.Series
        A Series with the cluster labels for each point, with -1 indicating
        noise points.

    """
    model = HDBSCAN(**hdbscan_params)

    labels = model.fit_predict(points.get_coordinates().to_numpy())
    if not isinstance(labels, np.ndarray):
        err = "HDBSCAN did not return an array of labels."
        raise TypeError(err)

    return pd.Series(labels, index=points.index, name="cluster").replace(-1, np.nan)


def generate_hull(
    geoms: np.ndarray,
    alpha: float = 0.2,
) -> shapely.Geometry | None:
    """Generate a concave hull for a given cluster of points.

    Parameters
    ----------
    geoms : np.ndarray
        An array of shapely Point geometries representing the points in the cluster.
    alpha : float, optional
        The alpha parameter for the concave hull algorithm, by default 0.2.
        Lower values result in a tighter hull, while higher values result
        in a looser hull.

    Returns
    -------
    shapely.geometry.Polygon or None
        A Polygon representing the concave hull of the cluster, or None
        if the cluster has fewer than 4 points.

    """
    if len(geoms) < 4:
        return None

    multi = shapely.geometry.MultiPoint(geoms)
    return shapely.concave_hull(multi, alpha)


def generate_hulls(
    points: gpd.GeoDataFrame,
    *,
    cluster_col: str,
    buffer: float = 20,
) -> gpd.GeoDataFrame:
    """Generate concave hulls for each cluster of points.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        A GeoDataFrame with point geometries and a column indicating the cluster
        each point belongs to.
    cluster_col : str, optional
        The name of the column in `points` that contains the cluster labels,
        by default "cluster".
    buffer : float, optional
        The distance to buffer the hulls, by default 20. This helps to smooth the hulls
        and account for any inaccuracies in the point locations.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the concave hulls for each cluster, with a 'cluster'
        column indicating the cluster label and a 'geometry' column containing
        the hull geometries.

    """
    hulls = {}
    for cluster_id, subdf in points.groupby(cluster_col):
        geoms = subdf["geometry"].to_numpy()
        hulls[cluster_id] = generate_hull(geoms)

    df_hulls = pd.Series(hulls).rename("geometry").reset_index()
    return (
        gpd.GeoDataFrame(df_hulls, geometry="geometry", crs=points.crs)
        .rename(
            columns={"index": cluster_col},
        )
        .assign(
            geometry=lambda df: df["geometry"].buffer(buffer, resolution=32),
            **{cluster_col: lambda df: df[cluster_col].astype(int)},
        )
    )


def join_hulls_to_points(
    points: gpd.GeoDataFrame,
    hulls: gpd.GeoDataFrame,
    *,
    cluster_col: str,
) -> pd.Series:
    joined_within = points.sjoin(hulls, how="inner", predicate="within").drop(
        columns=["index_right"],
    )
    joined_touches = points.sjoin(hulls, how="inner", predicate="touches").drop(
        columns=["index_right"],
    )
    joined = (
        pd.concat([joined_within, joined_touches])
        .reset_index(names="index")
        .drop_duplicates(subset="index")
        .set_index("index")
    )
    return joined[cluster_col].copy()


def get_hotspot_mask(
    points: gpd.GeoDataFrame,
    distance_band: float,
    *,
    jobs_col: str = "jobs",
) -> pd.Series:
    """Identify statistically significant hotspots using the Getis-Ord Gi* statistic.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        A GeoDataFrame with point geometries and a 'num_empleos_esperados' column
        representing the expected number of jobs at each point.
    distance_band : float
        The distance threshold to define neighbors in the spatial weights matrix.

    Returns
    -------
    pd.Series
        A Series indicating whether each point is a hotspot.

    """
    w = DistanceBand.from_dataframe(points, threshold=distance_band, binary=False)
    go = G_Local(points[jobs_col], w, star=True)

    sig = go.p_sim < SIGNIFICANCE_LEVEL

    return (go.Zs > 0) & sig


def cluster_points_weighted(
    points: gpd.GeoDataFrame,
    hotspot_mask: pd.Series,
    *,
    hdbscan_params: dict,
) -> pd.Series:
    """Cluster points using a weighted HDBSCAN approach.

    This function first identifies statistically significant hotspots using the
    Getis-Ord Gi* statistic, and then applies HDBSCAN clustering to those hotspots
    using a distance band to define spatial relationships.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        A GeoDataFrame with point geometries and a 'num_empleos_esperados' column
        representing the expected number of jobs at each point.
    hotspot_mask : pd.Series
        A Series indicating whether each point is a hotspot.
    hdbscan_params : dict
        A dictionary with the parameters to pass to the HDBSCAN model.
        See the HDBSCAN documentation for more details:
        https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan

    Returns
    -------
    pd.Series
        A Series with the cluster labels for each point, with -1 indicating
        noise points.

    """
    df_hotspots = points[hotspot_mask].assign(
        hotspot_cluster=lambda df: cluster_points(
            gpd.GeoDataFrame(df),
            hdbscan_params=hdbscan_params,
        ),
    )

    df_hulls = generate_hulls(
        df_hotspots.query("hotspot_cluster.notna()"),
        cluster_col="hotspot_cluster",
    )
    return join_hulls_to_points(
        points,
        df_hulls,
        cluster_col="hotspot_cluster",
    ).reindex(
        points.index,
        fill_value=np.nan,
    )
