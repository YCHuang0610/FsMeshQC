import numpy as np
from utils.geometric_calculation import (
    triangle_edges, triangle_area_from_edges,
    triangle_angles, triangle_inradius, triangle_circumradius
)


def compute_mesh_quality(V, F):
    """
    The mesh quality was calculated by taking the average
    of triangle qualities across all triangles in the cortical
    meshes. 
    https://doi.org/10.1016/j.neuroimage.2020.117012
    计算每个三角形的质量指标，返回 dict（每个键对应 (M,) 数组）
    指标说明：
        - area: 面积（mm^2）
        - min_edge, max_edge: 最短/最长边
        - min_angle, max_angle: 最小/最大内角（度）
        - shape_quality: 4*sqrt(3)*A / (a^2+b^2+c^2) ∈ (0,1]，等边=1
        - radius_ratio: 2*r_in/R_circ ∈ (0,1]，等边=1
        - aspect_proxy: 1 / shape_quality，等边=1，越大越瘦长
    """
    e0, _, e2, a, b, c = triangle_edges(V, F)
    area = triangle_area_from_edges(e2, e0)  # 任取两条边
    A, B, C = triangle_angles(a, b, c)

    # shape_quality（强推，稳健常用）
    denom = (a*a + b*b + c*c)
    # 避免 0 除
    sq = np.zeros_like(area)
    mask = denom > 0
    sq[mask] = (4.0 * np.sqrt(3.0) * area[mask]) / denom[mask]
    sq = np.clip(sq, 0.0, 1.0)  # 理论上<=1

    # radius_ratio（同样 0~1，等边=1）
    r_in = triangle_inradius(area, a, b, c)
    R_circ = triangle_circumradius(area, a, b, c)
    rr = np.zeros_like(area)
    ok = np.isfinite(R_circ) & (R_circ > 0)
    rr[ok] = 2.0 * r_in[ok] / R_circ[ok]
    rr = np.clip(rr, 0.0, 1.0)

    out = {
        "area": area,
        "edge_a": a, "edge_b": b, "edge_c": c,
        "min_edge": np.min(np.vstack([a, b, c]), axis=0),
        "max_edge": np.max(np.vstack([a, b, c]), axis=0),
        "angle_A": A, "angle_B": B, "angle_C": C,
        "min_angle": np.min(np.vstack([A, B, C]), axis=0),
        "max_angle": np.max(np.vstack([A, B, C]), axis=0),
        "shape_quality": sq,
        "radius_ratio": rr,
        "aspect_proxy": np.where(sq > 0, 1.0 / sq, np.inf),
    }
    return out

def summarize_quality(qdict):
    """
    给出若干关键指标的统计摘要（中位数/均值/5-95分位/最差值等）
    """
    def stats(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        return {
            "n": x.size,
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "p05": float(np.percentile(x, 5)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    keys_of_interest = ["shape_quality", "radius_ratio", "aspect_proxy",
                        "min_angle", "max_angle", "area", "min_edge", "max_edge"]
    return {k: stats(qdict[k]) for k in keys_of_interest}
