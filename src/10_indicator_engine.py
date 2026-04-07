"""
10_indicator_engine.py — 三维指标计算引擎 (Tri-dimensional Indicator System)

供给侧: 设施密度 D_k, 服务覆盖率 C_k, 类型多样性 H_k, 可达性 Acc_k
需求侧: 舆情活跃度 Act_k, 情感得分 Sent_k, 需求缺口比 Gap_k
质量侧: 政策符合度 S_pol, 文化价值评分 V_cul, 空间协同度 Syn_k

综合体检指数: HI_k = α·Supply_k + β·Demand_k + γ·Quality_k

输入: clean_poi.json, clean_sentiment.json, kg_topology.json, triplets_full.json
输出: indicator_values.csv, health_index.csv
"""
import json
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import yaml
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.geo_utils import GeoUtils, parse_coord, haversine
from src.utils.sentiment import SentimentAnalyzer


def load_config() -> Dict[str, Any]:
    with open(ROOT / "config" / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# 供给侧指标
# ──────────────────────────────────────────────
def compute_supply_indicators(
    pois: List[Dict],
    region_areas: Dict[str, float],
    service_radius_m: float = 1000.0
) -> Dict[str, Dict[str, float]]:
    """
    供给侧指标计算

    Returns:
        {region: {"density": D_k, "coverage": C_k, "diversity": H_k, "accessibility": Acc_k}}
    """
    # 按区划分
    region_pois: Dict[str, List[Dict]] = defaultdict(list)
    for poi in pois:
        district = poi.get("区", "") or poi.get("行政区", "")
        if district and str(district) not in ("[]", "None", "null", ""):
            region_pois[district].append(poi)

    geo = GeoUtils()
    results = {}

    for region, rpois in region_pois.items():
        n_pois = len(rpois)
        area_km2 = region_areas.get(region, 500.0)  # 默认500km²

        # 设施密度 D_k = N_k / A_k
        density = n_pois / max(area_km2, 0.1)

        # 服务覆盖率 (简化: 基于网格估计)
        coords = []
        for p in rpois:
            c = parse_coord(p.get("坐标", ""))
            if c:
                coords.append(c)
        if coords:
            coverage_dict = geo.compute_coverage(rpois, service_radius_m)
            coverage = coverage_dict.get(region, 0.0)
        else:
            coverage = 0.0

        # 类型多样性 (Shannon Entropy)
        type_counts = defaultdict(int)
        for p in rpois:
            t = p.get("标准类型", p.get("类型", "其他"))
            type_counts[t] += 1
        total = sum(type_counts.values())
        diversity = 0.0
        if total > 0:
            for count in type_counts.values():
                p_i = count / total
                if p_i > 0:
                    diversity -= p_i * math.log(p_i)
        # 标准化: 除以 log(6) 使其在 [0,1]
        max_entropy = math.log(6) if len(type_counts) > 0 else 1
        diversity_norm = diversity / max_entropy if max_entropy > 0 else 0

        # 可达性 (平均POI间距的倒数)
        accessibility = geo.compute_accessibility(rpois).get(region, 0.0) if len(coords) > 1 else 0.0

        results[region] = {
            "poi_count": n_pois,
            "area_km2": area_km2,
            "density": round(density, 4),
            "coverage": round(coverage, 4),
            "diversity": round(diversity_norm, 4),
            "accessibility": round(accessibility, 4),
        }

    return results


# ──────────────────────────────────────────────
# 需求侧指标
# ──────────────────────────────────────────────
def compute_demand_indicators(
    pois: List[Dict],
    sentiments: Dict[str, Any],
    analyzer: SentimentAnalyzer
) -> Dict[str, Dict[str, float]]:
    """
    需求侧指标计算

    Returns:
        {region: {"activity": Act_k, "sentiment_score": Sent_k, "demand_gap": Gap_k}}
    """
    # 按区划分
    region_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"note_counts": [], "sentiments": [], "poi_count": 0}
    )

    for poi in pois:
        district = poi.get("区", "") or poi.get("行政区", "")
        if not district:
            continue

        poi_name = poi.get("名称", poi.get("中文名", ""))
        region_data[district]["poi_count"] += 1

        # 获取该POI的清洗后笔记
        sent_data = sentiments.get(poi_name, {})
        notes = sent_data.get("notes", []) or poi.get("xiaohongshu_clean", []) or poi.get("xiaohongshu", [])

        note_count = len(notes)
        region_data[district]["note_counts"].append(note_count)

        # 情感分析
        if notes:
            poi_sentiment = analyzer.analyze_notes(notes, poi_name=poi_name)
            region_data[district]["sentiments"].append(poi_sentiment.get("avg_sentiment", 0.0))
        else:
            region_data[district]["sentiments"].append(0.0)

    # 全市平均活跃度（用于计算缺口比）
    all_activities = []
    for rdata in region_data.values():
        if rdata["poi_count"] > 0:
            act = sum(rdata["note_counts"]) / rdata["poi_count"]
            all_activities.append(act)
    city_avg_activity = np.mean(all_activities) if all_activities else 1.0
    city_high_activity = np.percentile(all_activities, 75) if len(all_activities) >= 4 else city_avg_activity

    results = {}
    for region, rdata in region_data.items():
        n_pois = rdata["poi_count"]
        if n_pois == 0:
            continue

        # 舆情活跃度 Act_k = Σ|Notes_p| / |P_k|
        activity = sum(rdata["note_counts"]) / n_pois

        # 情感得分 Sent_k = mean(sentiment(n))
        sentiments_arr = [s for s in rdata["sentiments"] if s != 0.0]
        sentiment_score = np.mean(sentiments_arr) if sentiments_arr else 0.0

        # 需求缺口比 Gap_k = (Act_high - Act_actual) / Act_high
        demand_gap = (city_high_activity - activity) / max(city_high_activity, 0.01)
        demand_gap = max(0.0, min(1.0, demand_gap))

        results[region] = {
            "poi_count": n_pois,
            "total_notes": sum(rdata["note_counts"]),
            "activity": round(activity, 4),
            "sentiment_score": round(float(sentiment_score), 4),
            "demand_gap": round(demand_gap, 4),
        }

    return results


# ──────────────────────────────────────────────
# 质量侧指标
# ──────────────────────────────────────────────
# 保护等级权重
GRADE_WEIGHTS = {
    "世界文化遗产": 1.0,
    "全国重点文物保护单位": 0.9,
    "省级文物保护单位": 0.7,
    "市级文物保护单位": 0.5,
    "区级文物保护单位": 0.3,
    "一般不可移动文物": 0.2,
}


def compute_quality_indicators(
    pois: List[Dict],
    triplets: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """
    质量侧指标计算

    Returns:
        {region: {"policy_compliance": S_pol, "cultural_value": V_cul, "spatial_synergy": Syn_k}}
    """
    region_pois: Dict[str, List[Dict]] = defaultdict(list)
    for poi in pois:
        district = poi.get("区", "") or poi.get("行政区", "")
        if district:
            region_pois[district].append(poi)

    # 构建空间关系图统计
    spatial_edges: Dict[str, int] = defaultdict(int)  # region -> 空间关系数
    for tri in triplets:
        if tri.get("relation") in ("邻近", "同区"):
            source_poi = tri.get("source_poi", tri.get("head", ""))
            for poi in pois:
                name = poi.get("名称", poi.get("中文名", ""))
                if name == source_poi:
                    district = poi.get("区", "") or poi.get("行政区", "")
                    if district:
                        spatial_edges[district] += 1
                    break

    results = {}
    for region, rpois in region_pois.items():
        n_pois = len(rpois)

        # 政策符合度（基于保护类型覆盖率的简化计算）
        protected_count = 0
        for p in rpois:
            ptype = p.get("保护类型", "")
            if ptype and str(ptype) not in ("None", "null", "", "未知"):
                protected_count += 1
        policy_compliance = protected_count / max(n_pois, 1)

        # 文化价值评分 V_cul = α·grade + β·age + γ·rarity
        value_scores = []
        for p in rpois:
            grade = p.get("保护类型", "")
            grade_score = 0.0
            for g_key, g_val in GRADE_WEIGHTS.items():
                if g_key in str(grade):
                    grade_score = g_val
                    break

            # 历史年代评分（越古老越高分）
            year = p.get("建成年份")
            age_score = 0.0
            if year and isinstance(year, (int, float)):
                # 映射: 公元前 → 1.0, 唐代(618) → 0.8, 明清(1400) → 0.5, 近现代(1900) → 0.2
                if year < 0:
                    age_score = 1.0
                elif year < 600:
                    age_score = 0.9
                elif year < 1000:
                    age_score = 0.8
                elif year < 1400:
                    age_score = 0.6
                elif year < 1800:
                    age_score = 0.4
                else:
                    age_score = 0.2

            # 稀缺性（该类型在全市的占比倒数）
            poi_type = p.get("标准类型", p.get("类型", ""))
            type_count = sum(1 for pp in pois if pp.get("标准类型", pp.get("类型", "")) == poi_type)
            rarity_score = 1.0 / math.log(max(type_count, 2))

            v = 0.4 * grade_score + 0.3 * age_score + 0.3 * rarity_score
            value_scores.append(v)

        cultural_value = np.mean(value_scores) if value_scores else 0.0

        # 空间协同度 Syn_k = |E_spatial(P_k)| / (|P_k| · (|P_k|-1) / 2)
        max_edges = n_pois * (n_pois - 1) / 2 if n_pois > 1 else 1
        spatial_synergy = spatial_edges.get(region, 0) / max_edges

        results[region] = {
            "poi_count": n_pois,
            "policy_compliance": round(policy_compliance, 4),
            "cultural_value": round(float(cultural_value), 4),
            "spatial_synergy": round(min(spatial_synergy, 1.0), 4),
        }

    return results


# ──────────────────────────────────────────────
# 归一化与综合指数
# ──────────────────────────────────────────────
def min_max_normalize(values: List[float]) -> List[float]:
    """Min-Max 归一化到 [0, 1]"""
    if not values:
        return []
    v_min, v_max = min(values), max(values)
    if v_max - v_min < 1e-10:
        return [0.5] * len(values)
    return [(v - v_min) / (v_max - v_min) for v in values]


def compute_health_index(
    supply: Dict[str, Dict],
    demand: Dict[str, Dict],
    quality: Dict[str, Dict],
    weights: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    计算综合体检指数 HI_k = α·Supply + β·Demand + γ·Quality
    """
    alpha = weights.get("supply", 0.35)
    beta = weights.get("demand", 0.35)
    gamma = weights.get("quality", 0.30)

    all_regions = set(supply.keys()) | set(demand.keys()) | set(quality.keys())

    # 各维度子指标聚合
    region_scores: Dict[str, Dict] = {}
    for region in all_regions:
        s = supply.get(region, {})
        d = demand.get(region, {})
        q = quality.get(region, {})

        supply_score = np.mean([
            s.get("density", 0), s.get("coverage", 0),
            s.get("diversity", 0), s.get("accessibility", 0)
        ]) if s else 0

        demand_score = np.mean([
            min(d.get("activity", 0) / 20, 1),  # 标准化活跃度
            (d.get("sentiment_score", 0) + 1) / 2,  # [-1,1] → [0,1]
            1 - d.get("demand_gap", 0.5),  # 缺口越小越好
        ]) if d else 0

        quality_score = np.mean([
            q.get("policy_compliance", 0),
            q.get("cultural_value", 0),
            q.get("spatial_synergy", 0),
        ]) if q else 0

        region_scores[region] = {
            "supply_score": round(float(supply_score), 4),
            "demand_score": round(float(demand_score), 4),
            "quality_score": round(float(quality_score), 4),
        }

    # 归一化各维度得分
    regions = list(region_scores.keys())
    if regions:
        for dim in ("supply_score", "demand_score", "quality_score"):
            raw = [region_scores[r][dim] for r in regions]
            normed = min_max_normalize(raw)
            for r, v in zip(regions, normed):
                region_scores[r][f"{dim}_norm"] = round(v, 4)

    # 综合指数
    results = {}
    for region in regions:
        rs = region_scores[region]
        hi = (alpha * rs.get("supply_score_norm", rs["supply_score"]) +
              beta * rs.get("demand_score_norm", rs["demand_score"]) +
              gamma * rs.get("quality_score_norm", rs["quality_score"]))
        results[region] = {
            **rs,
            "health_index": round(hi, 4),
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
        }

    return results


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def run_indicator_engine(config: Dict[str, Any]) -> Dict[str, Any]:
    """执行三维指标计算"""
    logger.info("=" * 60)
    logger.info("开始三维指标计算")
    logger.info("=" * 60)

    # 加载数据
    clean_poi_path = ROOT / config["data"]["clean_poi"]
    if not clean_poi_path.exists():
        clean_poi_path = ROOT / config["data"]["poi_raw"]
    with open(clean_poi_path, "r", encoding="utf-8") as f:
        pois = json.load(f)
    logger.info(f"加载 {len(pois)} 条POI")

    # 加载舆情
    sentiment_path = ROOT / config["data"]["clean_sentiment"]
    sentiments = {}
    if sentiment_path.exists():
        with open(sentiment_path, "r", encoding="utf-8") as f:
            sentiments = json.load(f)
    logger.info(f"加载 {len(sentiments)} 条舆情数据")

    # 加载三元组（用于空间协同度）
    triplets_path = ROOT / config["kg"]["triplets_full"]
    triplets = []
    if triplets_path.exists():
        with open(triplets_path, "r", encoding="utf-8") as f:
            triplets = json.load(f)
    logger.info(f"加载 {len(triplets)} 条三元组")

    # 区域面积
    region_areas = GeoUtils.get_region_areas()

    # 指标权重
    weights = config.get("indicator", {}).get("weights", {
        "supply": 0.35, "demand": 0.35, "quality": 0.30
    })
    service_radius = config.get("indicator", {}).get("service_radius_m", 1000)

    # ── 供给侧 ──
    logger.info("计算供给侧指标...")
    supply = compute_supply_indicators(pois, region_areas, service_radius)
    logger.info(f"  {len(supply)} 个区域的供给侧指标")

    # ── 需求侧 ──
    logger.info("计算需求侧指标...")
    analyzer = SentimentAnalyzer(method="snownlp")
    demand = compute_demand_indicators(pois, sentiments, analyzer)
    logger.info(f"  {len(demand)} 个区域的需求侧指标")

    # ── 质量侧 ──
    logger.info("计算质量侧指标...")
    quality = compute_quality_indicators(pois, triplets)
    logger.info(f"  {len(quality)} 个区域的质量侧指标")

    # ── 综合指数 ──
    logger.info("计算综合体检指数...")
    health = compute_health_index(supply, demand, quality, weights)

    # 输出
    out_dir = ROOT / config["indicator"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # indicator_values.csv — 各区域详细指标
    ind_path = ROOT / config["indicator"]["indicator_csv"]
    ind_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ind_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "区域", "POI数", "面积km2",
            "密度", "覆盖率", "多样性", "可达性",
            "活跃度", "情感得分", "需求缺口",
            "政策符合度", "文化价值", "空间协同度",
        ])
        all_regions = sorted(set(supply.keys()) | set(demand.keys()) | set(quality.keys()))
        for region in all_regions:
            s = supply.get(region, {})
            d = demand.get(region, {})
            q = quality.get(region, {})
            writer.writerow([
                region,
                s.get("poi_count", d.get("poi_count", q.get("poi_count", 0))),
                s.get("area_km2", ""),
                s.get("density", ""), s.get("coverage", ""),
                s.get("diversity", ""), s.get("accessibility", ""),
                d.get("activity", ""), d.get("sentiment_score", ""),
                d.get("demand_gap", ""),
                q.get("policy_compliance", ""), q.get("cultural_value", ""),
                q.get("spatial_synergy", ""),
            ])
    logger.info(f"指标值: {ind_path}")

    # health_index.csv — 综合指数排名
    hi_path = ROOT / config["indicator"]["health_index_csv"]
    sorted_regions = sorted(health.items(), key=lambda x: -x[1].get("health_index", 0))
    with open(hi_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["排名", "区域", "供给得分", "需求得分", "质量得分",
                         "体检指数HI", "α", "β", "γ"])
        for rank, (region, scores) in enumerate(sorted_regions, 1):
            writer.writerow([
                rank, region,
                scores["supply_score"], scores["demand_score"], scores["quality_score"],
                scores["health_index"],
                scores["weights"]["alpha"], scores["weights"]["beta"], scores["weights"]["gamma"],
            ])
    logger.info(f"体检指数: {hi_path}")

    # 打印排名
    logger.info("区县体检指数排名:")
    for rank, (region, scores) in enumerate(sorted_regions, 1):
        logger.info(f"  {rank}. {region}: HI={scores['health_index']:.3f} "
                    f"(S={scores['supply_score']:.3f}, D={scores['demand_score']:.3f}, Q={scores['quality_score']:.3f})")

    logger.info("=" * 60)
    logger.info("三维指标计算完成")
    logger.info("=" * 60)

    return {
        "supply": supply,
        "demand": demand,
        "quality": quality,
        "health_index": {k: v["health_index"] for k, v in health.items()},
        "ranking": [(r, s["health_index"]) for r, s in sorted_regions],
    }


if __name__ == "__main__":
    config = load_config()
    results = run_indicator_engine(config)

    print("\n" + "=" * 60)
    print("区县体检指数排名")
    print("=" * 60)
    for rank, (region, hi) in enumerate(results["ranking"], 1):
        print(f"  {rank}. {region}: {hi:.4f}")
    print("=" * 60)
