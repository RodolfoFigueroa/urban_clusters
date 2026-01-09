import argparse
from urban_clusters.spatial import generate_hulls
import geopandas as gpd


def generate_concave_hull():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "POINTS",
        type=str,
        help="Path to a geometry file containing the employment points.",
    )
    parser.add_argument(
        "OUTPUT",
        type=str,
        help="Path to save the convex hull geometry file.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Alpha parameter for the alpha shape algorithm.",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=20,
        help="Buffer distance to apply to the hulls.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="spatial_cluster",
        help="Column name to group points by when generating convex hulls.",
    )

    args = parser.parse_args()

    points = gpd.read_file(args.POINTS)
    hulls = generate_hulls(
        points,
        cluster_col=args.group_by,
        alpha=args.alpha,
        buffer=args.buffer,
    )
    hulls.to_file(args.OUTPUT)