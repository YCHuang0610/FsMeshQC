import json
from pathlib import Path
import numpy as np
import pandas as pd
from .meshQuality import summarize_quality


def save_mesh_quality(
    qdict: dict,
    F: np.ndarray,
    out_prefix: str,
    save_csv: bool = True,
    save_parquet: bool = False,
    save_npz: bool = True,
    save_summary_json: bool = True,
    bad_sq_thresh: float = 0.2,       # shape_quality 低于此值视为“坏”
    bad_minangle_thresh: float = 10.0 # 最小角小于此阈值（度）视为“坏”
):
    """
    将 mesh quality 结果保存到文件：
      - <prefix>_faces.(csv/parquet/npz)：逐面片指标
      - <prefix>_summary.json：关键指标的统计摘要
      - <prefix>_bad_faces.csv：坏三角形索引与指标（若有）
    """
    out_prefix = Path(out_prefix)
    out_dir = out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 逐面片指标表 ----
    face_ids = np.arange(F.shape[0])
    faces_df = pd.DataFrame({
        "face_id": face_ids,
        "v0": F[:, 0], "v1": F[:, 1], "v2": F[:, 2],
        "area": qdict["area"],
        "edge_a": qdict["edge_a"], "edge_b": qdict["edge_b"], "edge_c": qdict["edge_c"],
        "min_edge": qdict["min_edge"], "max_edge": qdict["max_edge"],
        "angle_A": qdict["angle_A"], "angle_B": qdict["angle_B"], "angle_C": qdict["angle_C"],
        "min_angle": qdict["min_angle"], "max_angle": qdict["max_angle"],
        "shape_quality": qdict["shape_quality"],
        "radius_ratio": qdict["radius_ratio"],
        "aspect_proxy": qdict["aspect_proxy"],
    })

    if save_csv:
        faces_csv = out_prefix.with_suffix("").as_posix() + "_faces.csv"
        faces_df.to_csv(faces_csv, index=False)
        print(f"[saved] {faces_csv}")

    if save_parquet:
        faces_parquet = out_prefix.with_suffix("").as_posix() + "_faces.parquet"
        faces_df.to_parquet(faces_parquet, index=False)
        print(f"[saved] {faces_parquet}")

    if save_npz:
        faces_npz = out_prefix.with_suffix("").as_posix() + "_faces.npz"
        np.savez_compressed(
            faces_npz,
            face_id=face_ids, F=F,
            area=qdict["area"],
            edge_a=qdict["edge_a"], edge_b=qdict["edge_b"], edge_c=qdict["edge_c"],
            min_edge=qdict["min_edge"], max_edge=qdict["max_edge"],
            angle_A=qdict["angle_A"], angle_B=qdict["angle_B"], angle_C=qdict["angle_C"],
            min_angle=qdict["min_angle"], max_angle=qdict["max_angle"],
            shape_quality=qdict["shape_quality"],
            radius_ratio=qdict["radius_ratio"],
            aspect_proxy=qdict["aspect_proxy"],
        )
        print(f"[saved] {faces_npz}")

    # ---- 2) 摘要统计 ----
    if save_summary_json:
        summary = summarize_quality(qdict)
        summary_json = out_prefix.with_suffix("").as_posix() + "_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[saved] {summary_json}")

    # ---- 3) 坏三角形导出 ----
    bad_mask = (qdict["shape_quality"] < bad_sq_thresh) | (qdict["min_angle"] < bad_minangle_thresh)
    bad_idx = np.where(bad_mask)[0]
    if bad_idx.size > 0:
        bad_df = faces_df.loc[bad_idx].copy()
        bad_df.sort_values(by=["shape_quality", "min_angle"], inplace=True)
        bad_csv = out_prefix.with_suffix("").as_posix() + "_bad_faces.csv"
        bad_df.to_csv(bad_csv, index=False)
        print(f"[saved] {bad_csv}  (n_bad={bad_idx.size})")
    else:
        print("[info] no bad faces under current thresholds.")
