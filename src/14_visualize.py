#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
14_visualize.py  —  论文图表生成
================================
按 CCF 论文 S5.2 关键图表清单:
  Fig.4  西安文化用地空间分布地图
  Fig.5  区县级体检指数热力图
  Fig.6  诊断报告样例与证据链 (文本输出)
  Tab.1  数据集统计表
  Tab.2  主实验结果对比表
  Tab.3  消融实验结果表
"""

import json, csv, sys, argparse, logging, os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.geo_utils import parse_coord
from src.utils.kg_utils import merge_relation_distribution, relation_display_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("visualize")


def load_config() -> Dict:
    import yaml
    with open(ROOT / "config/params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dirs(config: Dict):
    fig_dir = ROOT / config["visualization"]["figure_dir"]
    tab_dir = ROOT / config["visualization"]["table_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, tab_dir


def _valid_region(value: Any) -> bool:
    return str(value).strip() not in ("", "[]", "nan", "None", "null", "未识别", "缺失", "未知")


def _load_clean_pois(config: Dict) -> List[Dict[str, Any]]:
    clean_path = ROOT / config["data"].get("clean_poi", "")
    raw_path = ROOT / config["data"]["poi_raw"]
    poi_path = clean_path if clean_path.exists() else raw_path
    return json.loads(poi_path.read_text("utf-8"))


def _load_quality_report(config: Dict) -> Dict[str, Any]:
    quality_path = ROOT / config["data"].get("quality_report", "")
    if quality_path.exists():
        return json.loads(quality_path.read_text("utf-8"))
    return {}


def _load_kg_topology(config: Dict) -> Dict[str, Any]:
    kg_dir = ROOT / config["kg"]["output_dir"]
    topology_path = kg_dir / "kg_topology.json"
    if topology_path.exists():
        return json.loads(topology_path.read_text("utf-8"))
    return {}


def _extract_poi_type(poi: Dict[str, Any]) -> str:
    return (
        poi.get("标准类型")
        or poi.get("类型")
        or poi.get("type")
        or poi.get("分类")
        or "未分类"
    )


def _extract_region(poi: Dict[str, Any]) -> str:
    region = (
        poi.get("区")
        or poi.get("行政区")
        or poi.get("district")
        or poi.get("所属区")
        or poi.get("地区")
        or ""
    )
    return str(region).strip()


# ================================================================
# Fig.4  西安文化用地空间分布地图 (Folium)
# ================================================================

def fig4_spatial_distribution(config: Dict):
    """生成西安文化用地空间分布交互地图"""
    import folium

    fig_dir, _ = _ensure_dirs(config)

    pois = _load_clean_pois(config)

    type_colors = {
        "遗址遗迹类": "#e74c3c",
        "博物馆纪念馆类": "#3498db",
        "宗教场所类": "#9b59b6",
        "历史街区类": "#f39c12",
        "文化公园/广场类": "#1abc9c",
        "其他文化设施类": "#2ecc71",
    }

    def extract_lat_lon(poi: Dict[str, Any]) -> Optional[tuple]:
        lat = poi.get("坐标_lat", poi.get("纬度", poi.get("lat")))
        lon = poi.get("坐标_lon", poi.get("经度", poi.get("lng", poi.get("lon"))))
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)

        coord = parse_coord(str(poi.get("坐标", "")).strip())
        if coord is not None:
            return coord
        return None

    center = [34.26, 108.94]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    type_counts = Counter()
    for p in pois:
        coord = extract_lat_lon(p)
        if coord is None:
            continue

        lat, lon = coord
        name = p.get("名称", p.get("name", p.get("中文名", "")))
        poi_type = _extract_poi_type(p)
        region = _extract_region(p)

        if not (33.5 < lat < 35.0 and 107.5 < lon < 110.0):
            continue

        type_counts[poi_type] += 1
        color = type_colors.get(poi_type, "#95a5a6")

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            popup=folium.Popup(
                f"<b>{name}</b><br>类型: {poi_type}<br>区域: {region}<br>坐标: {lat:.6f}, {lon:.6f}",
                max_width=260,
            ),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
        ).add_to(m)

    # 图例
    legend_items = "".join(
        f'<li><span style="background:{c};width:12px;height:12px;display:inline-block;margin-right:5px;border-radius:50%;"></span>{t} ({type_counts.get(t,0)})</li>'
        for t, c in type_colors.items()
    )
    legend_html = f'''
    <div style="position:fixed;bottom:50px;left:50px;z-index:1000;background:white;
    padding:12px;border-radius:5px;border:1px solid #ccc;font-size:12px;">
    <b>Xi'an Cultural Land (N={sum(type_counts.values())})</b>
    <ul style="list-style:none;padding:5px 0 0 0;margin:0;">{legend_items}</ul>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = fig_dir / "fig4_spatial_distribution.html"
    m.save(str(out_path))
    log.info(f"Fig.4 saved: {out_path}")
    return out_path


# ================================================================
# Fig.5  区县级体检指数热力图 (matplotlib)
# ================================================================

def fig5_health_index_heatmap(config: Dict):
    """区县级体检指数热力图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir, _ = _ensure_dirs(config)
    viz = config["visualization"]

    plt.rcParams["font.family"] = ["Times New Roman", viz.get("font_family", "SimHei"), "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = [viz.get("font_family", "SimHei")]
    plt.rcParams["axes.unicode_minus"] = False

    # 加载健康指数
    hi_path = ROOT / config["indicator"]["health_index_csv"]
    if not hi_path.exists():
        log.warning(f"健康指数文件不存在: {hi_path}, 跳过 Fig.5")
        return None

    df = pd.read_csv(hi_path)
    region_col = "区域" if "区域" in df.columns else "region"
    supply_col = "供给得分" if "供给得分" in df.columns else "supply_norm"
    demand_col = "需求得分" if "需求得分" in df.columns else "demand_norm"
    quality_col = "质量得分" if "质量得分" in df.columns else "quality_norm"
    hi_col = "体检指数HI" if "体检指数HI" in df.columns else "health_index"

    df = df[df[region_col].map(_valid_region)].copy()
    df = df.sort_values(hi_col, ascending=False).reset_index(drop=True)
    if df.empty:
        log.warning("无区域数据，跳过 Fig.5")
        return None

    regions = df[region_col].astype(str).tolist()
    supply_vals = df[supply_col].astype(float).tolist()
    demand_vals = df[demand_col].astype(float).tolist()
    quality_vals = df[quality_col].astype(float).tolist()
    hi_vals = df[hi_col].astype(float).tolist()

    # 构造矩阵
    data = np.array([supply_vals, demand_vals, quality_vals, hi_vals])
    dim_labels = ["Supply", "Demand", "Quality", "Health Index"]

    figsize = viz.get("figsize_chart", [8, 6])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(dim_labels)))
    ax.set_yticklabels(dim_labels, fontsize=10)

    # 标注数值
    for i in range(len(dim_labels)):
        for j in range(len(regions)):
            val = data[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Score", fontsize=10)
    ax.set_title("区县级文化用地体检指数", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    out_path = fig_dir / "fig5_health_index_heatmap.png"
    plt.savefig(str(out_path), dpi=viz.get("dpi", 300), bbox_inches="tight")
    plt.close()
    log.info(f"Fig.5 saved: {out_path}")
    return out_path


# ================================================================
# Fig.6  KG类型分布条形图
# ================================================================

def fig6_kg_type_distribution(config: Dict):
    """知识图谱关系类型分布"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir, _ = _ensure_dirs(config)
    viz = config["visualization"]
    plt.rcParams["font.family"] = ["Times New Roman", viz.get("font_family", "SimHei"), "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = [viz.get("font_family", "SimHei")]
    plt.rcParams["axes.unicode_minus"] = False

    topology = _load_kg_topology(config)
    merged_rel = topology.get("relation_distribution_merged") or merge_relation_distribution(
        topology.get("relation_distribution_raw") or topology.get("relation_distribution", {})
    )
    rel_counter = Counter(merged_rel)

    if not rel_counter:
        log.warning("无三元组数据，跳过 Fig.6")
        return None

    items = rel_counter.most_common(15)
    labels = [relation_display_name(k) for k, _ in items]
    values = [v for _, v in items]

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    figsize = viz.get("figsize_chart", [8, 6])
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("知识图谱关系类型分布（合并后 Top 15）", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9)

    ax.invert_yaxis()
    plt.tight_layout()
    out_path = fig_dir / "fig6_kg_relation_distribution.png"
    plt.savefig(str(out_path), dpi=viz.get("dpi", 300), bbox_inches="tight")
    plt.close()
    log.info(f"Fig.6 saved: {out_path}")
    return out_path


# ================================================================
# Tab.1  数据集统计表
# ================================================================

def tab1_dataset_statistics(config: Dict):
    """生成 Xi'an-CulturalLand-310 数据集统计表"""
    _, tab_dir = _ensure_dirs(config)

    pois = _load_clean_pois(config)
    quality = _load_quality_report(config)

    # 统计
    total = len(pois)
    type_counter = Counter()
    region_counter = Counter()
    missing_region_count = 0
    has_baike = 0
    total_notes = 0
    total_comments = 0

    for p in pois:
        poi_type = _extract_poi_type(p)
        region = _extract_region(p)
        if _valid_region(region):
            region_counter[region] += 1
        else:
            missing_region_count += 1
        type_counter[poi_type] += 1

        if p.get("百科摘要") or p.get("baike_summary"):
            has_baike += 1

        xhs = p.get("xiaohongshu_clean") or p.get("xiaohongshu") or []
        total_notes += len(xhs)
        for note in xhs:
            total_comments += len(note.get("评论", []))

    # 写表
    rows = [
        ["数据项", "规模", "说明"],
        ["核心POI", str(total), f"{len(type_counter)}类文化用地"],
        ["百科语义", str(has_baike), f"覆盖率{has_baike/total*100:.1f}%"],
        ["小红书笔记", str(total_notes), f"含{total_comments}条评论"],
        ["行政区", str(len(region_counter)), ", ".join(f"{k}({v})" for k, v in region_counter.most_common(5))],
    ]
    if missing_region_count:
        rows.append(["区域缺失样本", str(missing_region_count), "未纳入区域分布统计"])
    for t, c in type_counter.most_common():
        rows.append([f"  - {t}", str(c), f"占比{c/total*100:.1f}%"])

    n_rels = _load_kg_topology(config).get("num_edges")
    if n_rels is not None:
        rows.append(["知识三元组", str(n_rels), "来源于清洗后的图谱拓扑统计"])

    removed_notes = quality.get("dq5_sentiment", {}).get("removed_notes")
    if removed_notes is not None:
        rows.append(["情感清洗移除笔记", str(removed_notes), "剔除城市无关或低质量文本"])

    out_path = tab_dir / "tab1_dataset_statistics.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    log.info(f"Tab.1 saved: {out_path}")

    # 同时生成 LaTeX
    _csv_to_latex(rows, tab_dir / "tab1_dataset_statistics.tex",
                  caption="Xi'an-CulturalLand-310 Dataset Statistics",
                  label="tab:dataset")
    return out_path


# ================================================================
# Tab.2  主实验结果对比表
# ================================================================

def tab2_main_results(config: Dict):
    """生成主实验结果对比表"""
    _, tab_dir = _ensure_dirs(config)

    eval_dir = ROOT / config["evaluation"]["output_dir"]
    comparison_csv = eval_dir / "comparison_table.csv"

    if not comparison_csv.exists():
        log.warning("对比表不存在, 生成空模板")
        # 生成模板
        headers = ["Method", "Coverage", "Precision", "Recall", "F1",
                    "Evidence Completeness", "Faithfulness", "Agreement(Kappa)"]
        methods = ["Rule-Only (B1)", "Vanilla-LLM (B2)", "Text-RAG (B3)",
                    "LightRAG (B4)", "CulturLand-Check (Ours)"]
        rows = [headers]
        for m_name in methods:
            rows.append([m_name] + ["-"] * (len(headers) - 1))

        out_path = tab_dir / "tab2_main_results.csv"
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        _csv_to_latex(rows, tab_dir / "tab2_main_results.tex",
                      caption="Main Experimental Results",
                      label="tab:main_results")
        log.info(f"Tab.2 template saved: {out_path}")
        return out_path

    # 读取实际数据
    rows = []
    with open(comparison_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    out_path = tab_dir / "tab2_main_results.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    _csv_to_latex(rows, tab_dir / "tab2_main_results.tex",
                  caption="Main Experimental Results",
                  label="tab:main_results")
    log.info(f"Tab.2 saved: {out_path}")
    return out_path


# ================================================================
# Tab.3  消融实验结果表
# ================================================================

def tab3_ablation_results(config: Dict):
    """生成消融实验结果表"""
    _, tab_dir = _ensure_dirs(config)

    eval_dir = ROOT / config["evaluation"]["output_dir"]

    ablation_names = {
        "ablation_no_graph": "w/o Schema Constraint (A1)",
        "ablation_no_sentiment": "w/o Demand Dimension (A2)",
        "ablation_no_policy": "w/o Quality Dimension (A3)",
        "ablation_no_spatial": "w/o Tri-dim Retrieval (A4)",
    }

    headers = ["Setting", "Avg Evidence Completeness", "Num Regions", "Notes"]
    rows = [headers]

    # Ours
    ours_dir = eval_dir / "ours"
    if ours_dir.exists() and (ours_dir / "eval_metrics.json").exists():
        ours = json.loads((ours_dir / "eval_metrics.json").read_text("utf-8"))
        rows.append(["Full Model (Ours)",
                      str(ours.get("avg_evidence_completeness", "-")),
                      str(ours.get("num_regions", "-")), ""])
    else:
        rows.append(["Full Model (Ours)", "-", "-", "需先运行完整流水线"])

    for abl_key, abl_name in ablation_names.items():
        abl_dir = eval_dir / abl_key
        if abl_dir.exists() and (abl_dir / "eval_metrics.json").exists():
            data = json.loads((abl_dir / "eval_metrics.json").read_text("utf-8"))
            rows.append([abl_name,
                         str(data.get("avg_evidence_completeness", "-")),
                         str(data.get("num_regions", "-")),
                         data.get("ablation_description", "")])
        else:
            rows.append([abl_name, "-", "-", "需运行消融实验"])

    out_path = tab_dir / "tab3_ablation_results.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    _csv_to_latex(rows, tab_dir / "tab3_ablation_results.tex",
                  caption="Ablation Study Results",
                  label="tab:ablation")
    log.info(f"Tab.3 saved: {out_path}")
    return out_path


# ================================================================
# 补充: POI 类型饼图
# ================================================================

def fig_poi_type_pie(config: Dict):
    """POI 类型分布饼图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir, _ = _ensure_dirs(config)
    viz = config["visualization"]
    plt.rcParams["font.family"] = ["Times New Roman", viz.get("font_family", "SimHei"), "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = [viz.get("font_family", "SimHei")]
    plt.rcParams["axes.unicode_minus"] = False

    quality = _load_quality_report(config)
    type_counter = Counter(quality.get("dq6_type", {}).get("type_distribution", {}))
    if not type_counter:
        pois = _load_clean_pois(config)
        for p in pois:
            type_counter[_extract_poi_type(p)] += 1

    labels = [k for k, _ in type_counter.most_common()]
    sizes = [v for _, v in type_counter.most_common()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.8
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(f"Xi'an Cultural Land Type Distribution (N={sum(sizes)})",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = fig_dir / "fig_poi_type_pie.png"
    plt.savefig(str(out_path), dpi=viz.get("dpi", 300), bbox_inches="tight")
    plt.close()
    log.info(f"POI type pie chart saved: {out_path}")
    return out_path


# ================================================================
# 补充: 区域 POI 数量柱状图
# ================================================================

def fig_region_bar(config: Dict):
    """各区域 POI 数量柱状图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir, _ = _ensure_dirs(config)
    viz = config["visualization"]
    plt.rcParams["font.family"] = ["Times New Roman", viz.get("font_family", "SimHei"), "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = [viz.get("font_family", "SimHei")]
    plt.rcParams["axes.unicode_minus"] = False

    pois = _load_clean_pois(config)

    region_counter = Counter()
    for p in pois:
        r = _extract_region(p)
        if not _valid_region(r):
            continue
        region_counter[r] += 1

    items = region_counter.most_common()
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), values, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(labels))))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of POIs", fontsize=11)
    ax.set_title("Cultural Land Distribution by District", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = fig_dir / "fig_region_bar.png"
    plt.savefig(str(out_path), dpi=viz.get("dpi", 300), bbox_inches="tight")
    plt.close()
    log.info(f"Region bar chart saved: {out_path}")
    return out_path


# ================================================================
# LaTeX 辅助
# ================================================================

def _csv_to_latex(rows: List[List[str]], output_path: Path,
                  caption: str = "", label: str = ""):
    """将 CSV 行转为 LaTeX tabular"""
    if not rows:
        return
    n_cols = len(rows[0])
    col_spec = "|" + "c|" * n_cols

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
    ]

    for i, row in enumerate(rows):
        escaped = [cell.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&") for cell in row]
        lines.append(" & ".join(escaped) + r" \\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"LaTeX table saved: {output_path}")


# ================================================================
# main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CulturLand-Check 论文图表生成")
    parser.add_argument("--config", default="config/params.yaml")
    parser.add_argument("--figure", choices=["all", "fig4", "fig5", "fig6", "pie", "bar"],
                        default="all", help="生成哪些图表")
    parser.add_argument("--table", choices=["all", "tab1", "tab2", "tab3", "none"],
                        default="all", help="生成哪些表格")
    args = parser.parse_args()

    config = load_config()

    # 图表
    if args.figure in ("all", "fig4"):
        try:
            fig4_spatial_distribution(config)
        except Exception as e:
            log.error(f"Fig.4 生成失败: {e}")

    if args.figure in ("all", "fig5"):
        try:
            fig5_health_index_heatmap(config)
        except Exception as e:
            log.error(f"Fig.5 生成失败: {e}")

    if args.figure in ("all", "fig6"):
        try:
            fig6_kg_type_distribution(config)
        except Exception as e:
            log.error(f"Fig.6 生成失败: {e}")

    if args.figure in ("all", "pie"):
        try:
            fig_poi_type_pie(config)
        except Exception as e:
            log.error(f"POI pie 生成失败: {e}")

    if args.figure in ("all", "bar"):
        try:
            fig_region_bar(config)
        except Exception as e:
            log.error(f"Region bar 生成失败: {e}")

    # 表格
    if args.table in ("all", "tab1"):
        try:
            tab1_dataset_statistics(config)
        except Exception as e:
            log.error(f"Tab.1 生成失败: {e}")

    if args.table in ("all", "tab2"):
        try:
            tab2_main_results(config)
        except Exception as e:
            log.error(f"Tab.2 生成失败: {e}")

    if args.table in ("all", "tab3"):
        try:
            tab3_ablation_results(config)
        except Exception as e:
            log.error(f"Tab.3 生成失败: {e}")

    log.info("\n All figures and tables generated.")


if __name__ == "__main__":
    main()
