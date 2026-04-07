"""
8_csge_extraction.py — Cultural-Schema-Guided Extraction (CSGE)
实现 Algorithm 1: 基于文化用地本体 Schema 约束的知识抽取

阶段一: Schema-Constrained Extraction — LLM按Schema约束生成候选三元组
阶段二: Spatial-Semantic Validation — 关系类型过滤 + 空间一致性校验

输入: artifacts/v20260225/data/clean_poi.json + config/ontology.json
输出: artifacts/v20260225/kg/triplets_full.json
"""
import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict
from copy import deepcopy

import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.llm_client import LLMClient
from src.utils.geo_utils import parse_coord, haversine


# ──────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────
def load_config() -> Dict[str, Any]:
    with open(ROOT / "config" / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ontology() -> Dict[str, Any]:
    with open(ROOT / "config" / "ontology.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template() -> str:
    with open(ROOT / "config" / "prompts" / "csge_extract.txt", "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────
# 本体辅助
# ──────────────────────────────────────────────
class OntologyHelper:
    """本体 Schema 辅助类"""

    def __init__(self, ontology: Dict[str, Any]):
        self.ontology = ontology
        entity_types = ontology.get("entity_types", {})
        relation_types = ontology.get("relation_types", {})

        if isinstance(entity_types, dict):
            self.entity_types = entity_types
        else:
            self.entity_types = {
                et["name"]: {k: v for k, v in et.items() if k != "name"}
                for et in entity_types
                if isinstance(et, dict) and et.get("name")
            }

        if isinstance(relation_types, dict):
            self.relation_types = relation_types
        else:
            self.relation_types = {
                rt["name"]: {k: v for k, v in rt.items() if k != "name"}
                for rt in relation_types
                if isinstance(rt, dict) and rt.get("name")
            }
        self.valid_relation_names: Set[str] = set(self.relation_types.keys())
        self.type_relations = ontology.get("type_specific_relations", {})

    def get_valid_relations(self, poi_type: str) -> List[str]:
        """获取该POI类型允许的关系列表"""
        return self.type_relations.get(poi_type, list(self.valid_relation_names))

    def get_core_attributes(self, poi_type: str) -> List[str]:
        """获取该POI类型的核心属性"""
        type_attr_map = {
            "遗址遗迹类": ["保护类型", "建成朝代", "面积", "建成/发现时间"],
            "博物馆纪念馆类": ["开放时间", "门票价格", "面积", "建议游玩时长"],
            "宗教场所类": ["保护类型", "建成朝代", "建成/发现时间", "地理位置"],
            "历史街区类": ["面积", "地理位置", "建成/发现时间", "保护类型"],
            "文化公园/广场类": ["开放时间", "面积", "地理位置", "建议游玩时长"],
            "其他文化设施类": ["开放时间", "门票价格", "面积", "地理位置"],
        }
        poi_core = self.entity_types.get("POI", {}).get(
            "core_attributes",
            ["名称", "类型", "坐标", "区"]
        )
        merged = []
        for item in list(poi_core) + type_attr_map.get(poi_type, []):
            if item and item not in merged:
                merged.append(item)
        return merged or ["名称", "类型", "坐标", "区"]

    def is_spatial_relation(self, relation: str) -> bool:
        """判断是否为空间关系"""
        rt = self.relation_types.get(relation, {})
        return rt.get("is_spatial", False)

    def is_valid_relation(self, relation: str, poi_type: str) -> bool:
        """验证关系类型对该POI类型是否合法"""
        valid = self.get_valid_relations(poi_type)
        return relation in valid


# ──────────────────────────────────────────────
# 阶段一: Schema-Constrained Extraction
# ──────────────────────────────────────────────
def build_extraction_prompt(
    poi: Dict, poi_type: str, ontology_helper: OntologyHelper, template: str
) -> str:
    """
    构建 Schema 约束的抽取 Prompt
    """
    valid_rels = ontology_helper.get_valid_relations(poi_type)
    core_attrs = ontology_helper.get_core_attributes(poi_type)

    prompt = template.format(
        type_name=poi_type,
        valid_relations="、".join(valid_rels),
        core_attributes="、".join(core_attrs),
        poi_name=poi.get("名称", poi.get("中文名", "")),
        summary=poi.get("百科摘要", "") or "",
        tags=poi.get("百科标签", "") or "",
        history=poi.get("历史信息", "") or "",
        coord=poi.get("坐标", ""),
        district=poi.get("区", "") or poi.get("行政区", "")
    )
    return prompt


def extract_triplets_for_poi(
    poi: Dict, poi_type: str, ontology_helper: OntologyHelper,
    llm: LLMClient, template: str
) -> Dict[str, Any]:
    """
    阶段一: 为单个POI执行Schema约束抽取

    Returns:
        {
            "poi_name": str,
            "poi_type": str,
            "entities": [...],
            "relations": [...],
            "raw_response": str
        }
    """
    prompt = build_extraction_prompt(poi, poi_type, ontology_helper, template)
    system_msg = "你是文化用地知识图谱抽取专家。请严格按照Schema约束生成JSON。"

    resp = llm.chat(prompt, system=system_msg)
    result = {
        "poi_name": poi.get("名称", poi.get("中文名", "")),
        "poi_type": poi_type,
        "entities": [],
        "relations": [],
        "raw_response": resp[:500]  # 截断保留用于调试
    }

    try:
        data = llm.extract_json(resp)
        if data:
            result["entities"] = data.get("entities", [])
            result["relations"] = data.get("relations", [])
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"  JSON解析失败: {e}")

    return result


# ──────────────────────────────────────────────
# 阶段二: Spatial-Semantic Validation
# ──────────────────────────────────────────────
def validate_triplets(
    extraction: Dict[str, Any],
    poi: Dict,
    all_poi_coords: Dict[str, Tuple[float, float]],
    ontology_helper: OntologyHelper,
    spatial_threshold_km: float = 50.0,
    min_confidence: float = 0.6
) -> Tuple[List[Dict], List[Dict]]:
    """
    阶段二: 三元组校验

    Args:
        extraction: 阶段一的抽取结果
        poi: 原始POI数据
        all_poi_coords: 全部POI的坐标字典 {name: (lat, lon)}
        ontology_helper: 本体辅助器
        spatial_threshold_km: 空间关系距离阈值(km)
        min_confidence: 最小置信度

    Returns:
        (accepted_triplets, rejected_triplets)
    """
    poi_type = extraction["poi_type"]
    poi_name = extraction["poi_name"]
    accepted, rejected = [], []

    for rel in extraction.get("relations", []):
        head = rel.get("head", "")
        relation = rel.get("relation", "")
        tail = rel.get("tail", "")
        confidence = float(rel.get("confidence", 0.5))

        rejection_reason = None

        # 检查1: 置信度过滤
        if confidence < min_confidence:
            rejection_reason = f"置信度过低({confidence:.2f}<{min_confidence})"

        # 检查2: 关系类型合法性
        elif not ontology_helper.is_valid_relation(relation, poi_type):
            rejection_reason = f"关系'{relation}'不在{poi_type}的允许关系集中"

        # 检查3: 空间一致性（仅对空间关系检查）
        elif ontology_helper.is_spatial_relation(relation):
            head_coord = all_poi_coords.get(head)
            tail_coord = all_poi_coords.get(tail)
            if head_coord and tail_coord:
                dist = haversine(head_coord[0], head_coord[1],
                                 tail_coord[0], tail_coord[1])
                if dist > spatial_threshold_km:
                    rejection_reason = f"空间距离过大({dist:.1f}km>{spatial_threshold_km}km)"

        # 检查4: 基本完整性
        elif not head or not relation or not tail:
            rejection_reason = "三元组不完整"

        triplet = {
            "head": head,
            "relation": relation,
            "tail": tail,
            "confidence": round(confidence, 3),
            "source_poi": poi_name,
            "poi_type": poi_type
        }

        if rejection_reason:
            triplet["rejection_reason"] = rejection_reason
            rejected.append(triplet)
        else:
            accepted.append(triplet)

    return accepted, rejected


# ──────────────────────────────────────────────
# 实体合并与去重
# ──────────────────────────────────────────────
def merge_and_deduplicate(
    all_entities: List[Dict],
    all_triplets: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    实体对齐去重 + 三元组去重

    Returns:
        (unique_entities, unique_triplets)
    """
    # 实体去重: 按 (name, type) 合并
    entity_map: Dict[str, Dict] = {}
    for ent in all_entities:
        key = f"{ent.get('name', '')}||{ent.get('type', '')}"
        if key not in entity_map:
            entity_map[key] = ent
        else:
            # 合并属性
            existing = entity_map[key]
            for k, v in ent.get("attributes", {}).items():
                if k not in existing.get("attributes", {}):
                    existing.setdefault("attributes", {})[k] = v

    # 三元组去重: 按 (head, relation, tail) 保留最高置信度
    triplet_map: Dict[str, Dict] = {}
    for tri in all_triplets:
        key = f"{tri['head']}||{tri['relation']}||{tri['tail']}"
        if key not in triplet_map:
            triplet_map[key] = tri
        else:
            if tri.get("confidence", 0) > triplet_map[key].get("confidence", 0):
                triplet_map[key] = tri

    return list(entity_map.values()), list(triplet_map.values())


# ──────────────────────────────────────────────
# 与已有三元组合并
# ──────────────────────────────────────────────
def merge_with_existing(
    new_triplets: List[Dict],
    existing_path: Path
) -> List[Dict]:
    """
    将新抽取的三元组与已有triplets.json合并
    """
    if not existing_path.exists():
        logger.info("未找到已有三元组文件，使用新抽取结果")
        return new_triplets

    with open(existing_path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    logger.info(f"已有三元组文件包含 {len(existing)} 条记录")

    # 将已有格式转为统一三元组格式
    existing_triplets = []
    for item in existing:
        entity_name = item.get("entity_name", "")
        for rel in item.get("relations", []):
            existing_triplets.append({
                "head": entity_name,
                "relation": rel.get("relation_type", ""),
                "tail": rel.get("entity_name", ""),
                "confidence": 0.8,  # 已有数据默认置信度
                "source_poi": item.get("source_poi", entity_name),
                "poi_type": item.get("entity_type", ""),
                "source": "existing_triplets"
            })

    # 标记新抽取的来源
    for tri in new_triplets:
        tri["source"] = "csge_extraction"

    all_triplets = existing_triplets + new_triplets
    # 去重
    _, unique = merge_and_deduplicate([], all_triplets)
    logger.info(f"合并后共 {len(unique)} 条唯一三元组 "
                f"(已有={len(existing_triplets)}, 新增={len(new_triplets)})")
    return unique


# ──────────────────────────────────────────────
# 统计与报告
# ──────────────────────────────────────────────
def compute_extraction_stats(
    all_accepted: List[Dict],
    all_rejected: List[Dict],
    total_pois: int,
    covered_pois: Set[str]
) -> Dict[str, Any]:
    """计算抽取统计"""
    rel_dist = defaultdict(int)
    for tri in all_accepted:
        rel_dist[tri["relation"]] += 1

    reject_reasons = defaultdict(int)
    for tri in all_rejected:
        reject_reasons[tri.get("rejection_reason", "未知")] += 1

    coverage = len(covered_pois) / max(total_pois, 1)

    return {
        "total_pois": total_pois,
        "covered_pois": len(covered_pois),
        "coverage_rate": round(coverage, 4),
        "total_accepted": len(all_accepted),
        "total_rejected": len(all_rejected),
        "acceptance_rate": round(len(all_accepted) / max(len(all_accepted) + len(all_rejected), 1), 4),
        "relation_distribution": dict(sorted(rel_dist.items(), key=lambda x: -x[1])),
        "rejection_reasons": dict(sorted(reject_reasons.items(), key=lambda x: -x[1])),
        "avg_triplets_per_poi": round(len(all_accepted) / max(len(covered_pois), 1), 2)
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def run_csge_extraction(config: Dict[str, Any], limit: int = 0) -> Dict[str, Any]:
    """
    执行完整的 CSGE 抽取流程

    Returns:
        抽取统计报告
    """
    logger.info("=" * 60)
    logger.info("开始 CSGE 知识抽取 (Algorithm 1)")
    logger.info("=" * 60)

    # 加载数据
    clean_poi_path = ROOT / config["data"]["clean_poi"]
    if not clean_poi_path.exists():
        logger.warning(f"clean_poi 不存在: {clean_poi_path}")
        logger.info("回退使用原始 POI 数据")
        clean_poi_path = ROOT / config["data"]["poi_raw"]

    with open(clean_poi_path, "r", encoding="utf-8") as f:
        pois = json.load(f)
    if limit and limit > 0:
        pois = pois[:limit]
    logger.info(f"加载 {len(pois)} 条POI")

    # 本体 & 模板
    ontology = load_ontology()
    onto_helper = OntologyHelper(ontology)
    template = load_prompt_template()

    # LLM
    llm = LLMClient(config["llm"])
    logger.info(f"LLM: {config['llm']['provider']} / {config['llm']['model']}")

    # 构建全局坐标索引
    all_coords: Dict[str, Tuple[float, float]] = {}
    for poi in pois:
        name = poi.get("名称", poi.get("中文名", ""))
        coord = parse_coord(poi.get("坐标", ""))
        if coord and name:
            all_coords[name] = coord

    # 阈值参数
    spatial_threshold = config.get("kg", {}).get("neighbor_distance_km", 2.0) * 10  # 宽松空间阈值
    min_conf = config.get("kg", {}).get("min_confidence", 0.6)

    # 抽取结果收集
    all_entities = []
    all_accepted = []
    all_rejected = []
    covered_pois: Set[str] = set()

    # 输出目录
    out_dir = ROOT / config["kg"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, poi in enumerate(pois):
        poi_name = poi.get("名称", poi.get("中文名", f"POI_{idx}"))
        poi_type = poi.get("标准类型", poi.get("类型", "其他文化设施类"))
        logger.info(f"[{idx+1}/{len(pois)}] CSGE抽取: {poi_name} ({poi_type})")

        # 阶段一: 抽取
        extraction = extract_triplets_for_poi(
            poi, poi_type, onto_helper, llm, template
        )

        # 收集实体
        all_entities.extend(extraction.get("entities", []))

        # 自动补充基础三元组（规则生成）
        base_triplets = generate_base_triplets(poi, poi_name, poi_type, all_coords, onto_helper)
        extraction["relations"].extend(base_triplets)

        # 阶段二: 校验
        accepted, rejected = validate_triplets(
            extraction, poi, all_coords, onto_helper,
            spatial_threshold_km=spatial_threshold,
            min_confidence=min_conf
        )

        all_accepted.extend(accepted)
        all_rejected.extend(rejected)

        if accepted:
            covered_pois.add(poi_name)

        logger.info(f"  → 接受 {len(accepted)} / 拒绝 {len(rejected)} 条三元组")

    # 合并去重
    logger.info("执行实体对齐与三元组去重...")
    unique_entities, unique_triplets = merge_and_deduplicate(all_entities, all_accepted)
    logger.info(f"去重后: {len(unique_entities)} 实体, {len(unique_triplets)} 三元组")

    # 与已有三元组合并
    existing_path = ROOT / config["data"]["triplets_raw"]
    final_triplets = merge_with_existing(unique_triplets, existing_path)

    # 更新覆盖POI集合
    for tri in final_triplets:
        sp = tri.get("source_poi", "")
        if sp:
            covered_pois.add(sp)

    # 统计
    stats = compute_extraction_stats(all_accepted, all_rejected, len(pois), covered_pois)
    logger.info(f"抽取统计: 覆盖率={stats['coverage_rate']:.1%}, "
                f"接受率={stats['acceptance_rate']:.1%}, "
                f"平均三元组/POI={stats['avg_triplets_per_poi']}")

    # 保存输出
    triplets_path = ROOT / config["kg"]["triplets_full"]
    triplets_path.parent.mkdir(parents=True, exist_ok=True)
    with open(triplets_path, "w", encoding="utf-8") as f:
        json.dump(final_triplets, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存三元组: {triplets_path} ({len(final_triplets)} 条)")

    stats_path = ROOT / config["kg"]["kg_stats"]
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存抽取统计: {stats_path}")

    # 保存被拒绝的三元组（用于分析）
    rejected_path = out_dir / "rejected_triplets.json"
    with open(rejected_path, "w", encoding="utf-8") as f:
        json.dump(all_rejected[:200], f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("CSGE 抽取完成")
    logger.info(f"  覆盖率: {stats['coverage_rate']:.1%} (目标>70%)")
    logger.info(f"  关系分布: {stats['relation_distribution']}")
    logger.info("=" * 60)

    return stats


# ──────────────────────────────────────────────
# 基础三元组规则生成
# ──────────────────────────────────────────────
def generate_base_triplets(
    poi: Dict, poi_name: str, poi_type: str,
    all_coords: Dict[str, Tuple[float, float]],
    onto_helper: OntologyHelper
) -> List[Dict]:
    """
    基于规则为POI生成基础三元组（无需LLM）
    """
    triplets = []
    district = poi.get("区", "") or poi.get("行政区", "")

    # 位于关系
    if district:
        triplets.append({
            "head": poi_name,
            "relation": "位于",
            "tail": district,
            "confidence": 1.0,
        })

    # 属于关系（类型）
    if poi_type:
        triplets.append({
            "head": poi_name,
            "relation": "属于",
            "tail": poi_type,
            "confidence": 1.0,
        })

    # 始建于关系
    year = poi.get("建成年份") or poi.get("建成/发现时间")
    era = poi.get("建成朝代")
    if year or era:
        time_entity = str(era) if era else str(year)
        if time_entity and time_entity not in ("None", "null", ""):
            triplets.append({
                "head": poi_name,
                "relation": "始建于",
                "tail": time_entity,
                "confidence": 0.9,
            })

    # 邻近关系（距离<2km的POI对）
    coord = all_coords.get(poi_name)
    if coord:
        neighbor_threshold = 2.0  # km
        for other_name, other_coord in all_coords.items():
            if other_name == poi_name:
                continue
            dist = haversine(coord[0], coord[1], other_coord[0], other_coord[1])
            if dist <= neighbor_threshold:
                # 避免重复（只保留字典序较小的作为head）
                if poi_name < other_name:
                    triplets.append({
                        "head": poi_name,
                        "relation": "邻近",
                        "tail": other_name,
                        "confidence": round(max(0.6, 1.0 - dist / neighbor_threshold), 3),
                    })

    # 同区关系（同一行政区的标记）
    # 注：这将在全局merge时自动处理

    return triplets


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSGE 知识抽取")
    parser.add_argument("--config", default=None, help="自定义配置文件路径")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理的POI数量（0=全部）")
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过LLM调用，仅使用规则抽取（用于测试）")
    args = parser.parse_args()

    config = load_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f))

    if args.no_llm:
        logger.info("[no-llm] 跳过LLM调用，Stage 2 在 --no-llm 模式下输出空三元组集合")
        # 输出空结果文件，让后续阶段可以继续
        import os
        out_dir = ROOT / config["kg"]["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        empty_stats = {
            "total_pois": 0, "covered_pois": 0, "coverage_rate": 0.0,
            "total_accepted": 0, "total_rejected": 0, "acceptance_rate": 0.0,
            "relation_distribution": {}, "rejection_reasons": {},
            "avg_triplets_per_poi": 0.0, "note": "no-llm mode, skipped"
        }
        with open(out_dir / "csge_stats.json", "w", encoding="utf-8") as f:
            json.dump(empty_stats, f, ensure_ascii=False, indent=2)
        logger.info(f"[no-llm] 已写入空统计: {out_dir / 'csge_stats.json'}")
        import sys; sys.exit(0)

    stats = run_csge_extraction(config, limit=args.limit)

    print("\n" + "=" * 60)
    print("CSGE 抽取统计")
    print("=" * 60)
    print(f"POI总数: {stats['total_pois']}")
    print(f"已覆盖: {stats['covered_pois']} ({stats['coverage_rate']:.1%})")
    print(f"三元组: 接受={stats['total_accepted']}, 拒绝={stats['total_rejected']}")
    print(f"接受率: {stats['acceptance_rate']:.1%}")
    print(f"关系分布: {json.dumps(stats['relation_distribution'], ensure_ascii=False, indent=2)}")
    print("=" * 60)
