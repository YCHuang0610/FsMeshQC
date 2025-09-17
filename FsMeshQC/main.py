
"""
FsMeshQC - FreeSurfer网格质量控制工具
====================================

描述:
-----
这个工具用于分析FreeSurfer表面文件的网格质量，计算各种质量指标
并生成质量报告。适用于FreeSurfer生成的皮层表面文件，如lh.white、rh.pial等。

功能:
-----
- 读取FreeSurfer格式的网格表面文件
- 计算网格质量指标（形状质量、角度等）
- 提供质量摘要统计信息
- 输出多种格式的质量报告（CSV、NPZ、JSON等）
- 识别低质量的网格面片

用法示例:
--------
基本用法:
    python main.py -i /path/to/lh.white

指定输出位置:
    python main.py -i /path/to/rh.pial -o ./results/rh_pial_quality

仅查看摘要，不保存文件:
    python main.py -i /path/to/lh.white --summary-only

调整质量阈值:
    python main.py -i /path/to/lh.white --bad-sq-thresh 0.15 --bad-angle-thresh 8.0

作者: ychunhuang
版本: 0.1.0
许可证: MIT
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from utils.fs_io import read_freesurfer_surf
from utils.meshQuality import compute_mesh_quality, summarize_quality
from utils.saveResults import save_mesh_quality


def parse_arguments():
    """处理命令行参数"""
    parser = argparse.ArgumentParser(
        description="FreeSurfer网格质量控制工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input_surf", 
        help="输入的FreeSurfer表面文件路径 (例如: lh.white, rh.pial)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        help="输出文件前缀 (默认与输入文件同目录下的<文件名>_quality)",
        default=None
    )
    
    parser.add_argument(
        "--csv", 
        help="是否保存CSV格式的面质量数据", 
        action="store_true",
        default=True
    )
    
    parser.add_argument(
        "--no-csv", 
        help="不保存CSV格式的面质量数据", 
        action="store_false",
        dest="csv"
    )
    
    parser.add_argument(
        "--parquet", 
        help="保存Parquet格式的面质量数据", 
        action="store_true"
    )
    
    parser.add_argument(
        "--npz", 
        help="是否保存NPZ格式的面质量数据", 
        action="store_true",
        default=True
    )
    
    parser.add_argument(
        "--no-npz", 
        help="不保存NPZ格式的面质量数据", 
        action="store_false",
        dest="npz"
    )
    
    parser.add_argument(
        "--json", 
        help="是否保存JSON格式的摘要数据", 
        action="store_true", 
        default=True
    )
    
    parser.add_argument(
        "--no-json", 
        help="不保存JSON格式的摘要数据", 
        action="store_false",
        dest="json"
    )
    
    parser.add_argument(
        "--bad-sq-thresh", 
        help="形状质量阈值，低于此值视为有问题的三角形", 
        type=float, 
        default=0.2
    )
    
    parser.add_argument(
        "--bad-angle-thresh", 
        help="最小角阈值（度），小于此值视为有问题的三角形", 
        type=float, 
        default=10.0
    )
    
    parser.add_argument(
        "--summary-only", 
        help="仅显示质量摘要，不保存文件", 
        action="store_true"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input_surf):
        print(f"错误：找不到输入文件 {args.input_surf}")
        sys.exit(1)
    
    # 设置输出前缀
    if args.output is None:
        input_path = Path(args.input_surf)
        args.output = input_path.with_suffix("").as_posix() + "_quality"
    
    # 读取FreeSurfer表面文件
    print("================START================")
    print(f"正在读取 {args.input_surf}...")
    try:
        V, F = read_freesurfer_surf(args.input_surf)
        print(f"  顶点数: {V.shape[0]}, 面片数: {F.shape[0]}")
    except Exception as e:
        print(f"错误：读取表面文件失败 - {e}")
        sys.exit(1)
    
    # 计算网格质量
    print("正在计算网格质量指标...")
    q = compute_mesh_quality(V, F)
    summary = summarize_quality(q)
    
    # 显示质量摘要
    print("\n网格质量摘要:")
    print(f"  形状质量 (shape_quality): 中位数={summary['shape_quality']['median']:.4f}, "
            f"平均值={summary['shape_quality']['mean']:.4f}, "
            f"最小值={summary['shape_quality']['min']:.4f}")
    print(f"  最小角 (min_angle): 中位数={summary['min_angle']['median']:.2f}°, "
            f"平均值={summary['min_angle']['mean']:.2f}°, "
            f"最小值={summary['min_angle']['min']:.2f}°")
    
    # 低质量三角形统计
    bad_mask = (q["shape_quality"] < args.bad_sq_thresh) | (q["min_angle"] < args.bad_angle_thresh)
    bad_count = np.sum(bad_mask)
    total_count = F.shape[0]
    bad_percent = (bad_count / total_count) * 100
    print(f"低质量三角形: {bad_count}/{total_count} ({bad_percent:.2f}%)")
    
    # 如果只需要摘要，则不保存文件
    if args.summary_only:
        return
    
    # 保存结果
    print("\n保存结果...")
    save_mesh_quality(
        qdict=q,
        F=F,
        out_prefix=args.output,
        save_csv=args.csv,
        save_parquet=args.parquet,
        save_npz=args.npz,
        save_summary_json=args.json,
        bad_sq_thresh=args.bad_sq_thresh,
        bad_minangle_thresh=args.bad_angle_thresh
    )


if __name__ == "__main__":
    main()
