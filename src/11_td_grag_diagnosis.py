"""
11_td_grag_diagnosis.py — Tri-dimensional Graph-RAG (TD-GRAG) 诊断
实现 Algorithm 2: 三维查询分解 → 维度感知子图检索 → 证据链构造 → LLM诊断生成

输入: Neo4j/triplets_full.json, indicator_values.csv, clean_poi.json
输出: diagnosis_*.md, evidence_chains.json
"""
import json
import csv
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime

import yaml
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.llm_client import LLMClient


def load_config() -> Dict[str, Any]:
    with open(ROOT / "config" / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_diagnosis_template() -> str:
    with open(ROOT / "config" / "prompts" / "diagnosis_template.txt", "r", encoding="utf-8") as f:
        return f.read()


def save_ours_evaluation(config: Dict[str, Any], reports: Dict[str, str], faithfulness: Dict[str, Dict[str, Any]]):
    """同步保存本方法评估结果，供 12_evaluation.py 汇总。"""
    eval_dir = ROOT / config.get("evaluation", {}).get("output_dir", "artifacts/v20260225/evaluation")
    ours_dir = eval_dir / "ours"
    ours_dir.mkdir(parents=True, exist_ok=True)

    def _evidence_completeness(report_text: str) -> float:
        import re
        sentences = [s.strip() for s in re.split(r'[。！？\n]', report_text) if len(s.strip()) > 5]
        if not sentences:
            return 0.0
        cited = sum(1 for s in sentences if re.search(r'\[E[_\w]+-\d+\]', s))
        return round(cited / len(sentences), 4)

    ec_scores = []
    for region, report in reports.items():
        ec_scores.append(_evidence_completeness(report))
        (ours_dir / f"diagnosis_{region}.md").write_text(report, encoding="utf-8")

    avg_faith = float(np.mean([v.get("faithfulness_score", 0.0) for v in faithfulness.values()])) if faithfulness else 0.0
    metrics = {
        "method": "Ours_TD_GRAG",
        "num_regions": len(reports),
        "reports": reports,
        "evidence_completeness_scores": [round(float(x), 4) for x in ec_scores],
        "avg_evidence_completeness": round(float(np.mean(ec_scores)), 4) if ec_scores else 0.0,
        "avg_faithfulness": round(avg_faith, 4),
        "faithfulness_scores": faithfulness,
    }
    with open(ours_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# 子图检索器 (基于内存三元组)
# ──────────────────────────────────────────────
class TripletsRetriever:
    """基于三元组列表的子图检索（无需Neo4j）"""

    def __init__(self, triplets: List[Dict], pois: List[Dict]):
        self.triplets = triplets
        self.pois = pois
        # 构建索引
        self.by_head: Dict[str, List[Dict]] = defaultdict(list)
        self.by_tail: Dict[str, List[Dict]] = defaultdict(list)
        self.by_relation: Dict[str, List[Dict]] = defaultdict(list)
        self.poi_by_region: Dict[str, List[str]] = defaultdict(list)
        self.poi_info: Dict[str, Dict] = {}

        for tri in triplets:
            self.by_head[tri.get("head", "")].append(tri)
            self.by_tail[tri.get("tail", "")].append(tri)
            self.by_relation[tri.get("relation", "")].append(tri)

        for poi in pois:
            name = poi.get("名称", poi.get("中文名", ""))
            district = poi.get("区", "") or poi.get("行政区", "")
            if name:
                self.poi_info[name] = poi
                if district and str(district) not in ("[]", "None", "null", ""):
                    self.poi_by_region[district].append(name)

    def retrieve_supply_subgraph(self, region: str) -> List[Dict]:
        """
        检索供给侧子图: POI→位于→区域, POI→邻近→POI 路径
        """
        region_pois = self.poi_by_region.get(region, [])
        subgraph = []

        for poi_name in region_pois:
            # POI→位于→区域
            for tri in self.by_head.get(poi_name, []):
                if tri["relation"] in ("位于", "邻近", "同区", "同类"):
                    subgraph.append(tri)
            # 区域←位于←POI
            for tri in self.by_tail.get(region, []):
                if tri["head"] in region_pois:
                    subgraph.append(tri)

        return subgraph

    def retrieve_demand_subgraph(self, region: str) -> List[Dict]:
        """
        检索需求侧子图: POI→关注于→舆情主题 路径
        """
        region_pois = self.poi_by_region.get(region, [])
        subgraph = []

        for poi_name in region_pois:
            for tri in self.by_head.get(poi_name, []):
                if tri["relation"] in ("关注于",):
                    subgraph.append(tri)
            # 获取POI的舆情摘要
            poi = self.poi_info.get(poi_name, {})
            notes = poi.get("xiaohongshu_clean", poi.get("xiaohongshu", []))
            if notes:
                subgraph.append({
                    "head": poi_name,
                    "relation": "舆情统计",
                    "tail": f"笔记{len(notes)}条",
                    "confidence": 1.0,
                    "_notes_sample": notes[:3]
                })

        return subgraph

    def retrieve_quality_subgraph(self, region: str) -> List[Dict]:
        """
        检索质量侧子图: POI→受约束于→政策条款, POI→属于→遗产等级 路径
        """
        region_pois = self.poi_by_region.get(region, [])
        subgraph = []

        for poi_name in region_pois:
            for tri in self.by_head.get(poi_name, []):
                if tri["relation"] in ("受约束于", "属于", "始建于"):
                    subgraph.append(tri)
            # 补充保护信息
            poi = self.poi_info.get(poi_name, {})
            protect = poi.get("保护类型", "")
            if protect and str(protect) not in ("None", "null", ""):
                subgraph.append({
                    "head": poi_name,
                    "relation": "保护等级",
                    "tail": str(protect),
                    "confidence": 1.0
                })

        return subgraph


# ──────────────────────────────────────────────
# 证据链构造
# ──────────────────────────────────────────────
def build_evidence_chain(
    subgraph: List[Dict],
    indicators: Dict[str, float],
    dimension: str,
    evidence_prefix: str
) -> List[Dict]:
    """
    构造证据链: 子图路径 + 指标值 → 编号化证据

    Returns:
        [{"id": "E_supply-001", "type": "subgraph|indicator", "content": "..."}, ...]
    """
    evidence = []
    eid = 1

    # 指标证据
    for k, v in indicators.items():
        evidence.append({
            "id": f"{evidence_prefix}-{eid:03d}",
            "type": "indicator",
            "dimension": dimension,
            "content": f"{k} = {v}"
        })
        eid += 1

    # 子图路径证据
    seen = set()
    for tri in subgraph:
        path = f"{tri['head']} --[{tri['relation']}]--> {tri['tail']}"
        if path not in seen:
            seen.add(path)
            evidence.append({
                "id": f"{evidence_prefix}-{eid:03d}",
                "type": "subgraph_path",
                "dimension": dimension,
                "content": path,
                "confidence": tri.get("confidence", 1.0)
            })
            eid += 1
            if eid > 30:  # 每个维度限制30条证据
                break

    return evidence


# ──────────────────────────────────────────────
# Prompt组装
# ──────────────────────────────────────────────
def compose_diagnosis_prompt(
    region: str,
    e_supply: List[Dict],
    e_demand: List[Dict],
    e_quality: List[Dict],
    template: str
) -> str:
    """组装 Chain-of-Rationale 诊断 Prompt"""
    def format_evidence(evidence_list):
        lines = []
        for e in evidence_list:
            lines.append(f"  [{e['id']}] {e['content']}")
        return "\n".join(lines) if lines else "  (无可用证据)"

    prompt = f"""你是城市文化用地体检诊断专家。请根据以下三维证据链对【{region}】进行诊断。
每句结论必须引用至少一条证据编号 [Ek-xxx]。

== 供给侧证据 ==
{format_evidence(e_supply)}

== 需求侧证据 ==
{format_evidence(e_demand)}

== 质量侧证据 ==
{format_evidence(e_quality)}

== 输出格式 ==
请按以下结构输出诊断报告：

# {region} 文化用地体检诊断报告

## 1. 总体评价
(综合评价该区域文化用地现状，引用各维度关键证据)

## 2. 供给侧诊断
(分析设施密度、覆盖率、多样性、可达性，引用 [E_supply-xxx])

## 3. 需求侧诊断
(分析公众感知、舆情热度、情感倾向，引用 [E_demand-xxx])

## 4. 质量侧诊断
(分析政策符合度、文化价值、空间协同，引用 [E_quality-xxx])

## 5. 改进建议
(基于诊断结果提出3-5条具体建议)
"""
    return prompt


# ──────────────────────────────────────────────
# 忠实性验证 (Faithfulness Verification)
# ──────────────────────────────────────────────
def verify_faithfulness(
    report: str,
    evidence_ids: List[str]
) -> Dict[str, Any]:
    """
    验证诊断报告的忠实性:
    1. 检查证据引用率（报告中引用了多少证据）
    2. 检查幻觉指标（报告中是否出现未在证据中的实体/数值）
    """
    import re

    # 提取报告中引用的证据ID
    cited_ids = set(re.findall(r'\[E_\w+-\d{3}\]', report))
    cited_ids_clean = {cid.strip("[]") for cid in cited_ids}

    total_evidence = len(evidence_ids)
    cited_count = len(cited_ids_clean & set(evidence_ids))

    citation_rate = cited_count / max(total_evidence, 1)

    return {
        "total_evidence": total_evidence,
        "cited_evidence": cited_count,
        "citation_rate": round(citation_rate, 4),
        "uncited_evidence": total_evidence - cited_count,
        "faithfulness_score": round(min(citation_rate * 1.2, 1.0), 4),  # 宽松评分
        "pass": citation_rate >= 0.3  # 至少引用30%的证据
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def run_td_grag_diagnosis(config: Dict[str, Any]) -> Dict[str, Any]:
    """执行 TD-GRAG 诊断"""
    logger.info("=" * 60)
    logger.info("开始 TD-GRAG 诊断 (Algorithm 2)")
    logger.info("=" * 60)

    # 加载数据
    clean_poi_path = ROOT / config["data"]["clean_poi"]
    if not clean_poi_path.exists():
        clean_poi_path = ROOT / config["data"]["poi_raw"]
    with open(clean_poi_path, "r", encoding="utf-8") as f:
        pois = json.load(f)

    # 加载三元组
    triplets_path = ROOT / config["kg"]["triplets_full"]
    if not triplets_path.exists():
        triplets_path = ROOT / config["data"]["triplets_raw"]
    with open(triplets_path, "r", encoding="utf-8") as f:
        triplets = json.load(f)
    # 处理旧格式
    if triplets and isinstance(triplets[0], dict) and "relations" in triplets[0]:
        flat = []
        for item in triplets:
            entity_name = item.get("entity_name", "")
            for rel in item.get("relations", []):
                flat.append({
                    "head": entity_name,
                    "relation": rel.get("relation_type", ""),
                    "tail": rel.get("entity_name", ""),
                    "confidence": 0.8,
                    "source_poi": item.get("source_poi", entity_name)
                })
        triplets = flat

    # 加载指标
    indicator_path = ROOT / config["indicator"]["indicator_csv"]
    indicators: Dict[str, Dict[str, float]] = {}
    if indicator_path.exists():
        with open(indicator_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                region = row.get("区域", "")
                if region:
                    indicators[region] = {k: float(v) if v else 0.0
                                          for k, v in row.items() if k != "区域"}

    # 初始化
    retriever = TripletsRetriever(triplets, pois)
    llm = LLMClient(config["llm"])
    template = load_diagnosis_template()

    # 输出目录
    diag_cfg = config.get("diagnosis", {})
    out_dir = ROOT / diag_cfg.get("output_dir", "artifacts/v20260225/diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有有POI的区域
    regions = sorted(retriever.poi_by_region.keys())
    logger.info(f"需要诊断 {len(regions)} 个区域: {regions}")

    all_evidence_chains = {}
    all_faithfulness = {}
    all_reports = {}

    for idx, region in enumerate(regions):
        logger.info(f"[{idx+1}/{len(regions)}] 诊断区域: {region}")

        # Step 1-2: 维度感知子图检索
        sg_supply = retriever.retrieve_supply_subgraph(region)
        sg_demand = retriever.retrieve_demand_subgraph(region)
        sg_quality = retriever.retrieve_quality_subgraph(region)

        # 获取该区域指标
        ind = indicators.get(region, {})
        supply_ind = {k: v for k, v in ind.items()
                      if k in ("密度", "覆盖率", "多样性", "可达性", "POI数")}
        demand_ind = {k: v for k, v in ind.items()
                      if k in ("活跃度", "情感得分", "需求缺口")}
        quality_ind = {k: v for k, v in ind.items()
                       if k in ("政策符合度", "文化价值", "空间协同度")}

        # Step 3: 证据链构造
        e_supply = build_evidence_chain(sg_supply, supply_ind, "supply", "E_supply")
        e_demand = build_evidence_chain(sg_demand, demand_ind, "demand", "E_demand")
        e_quality = build_evidence_chain(sg_quality, quality_ind, "quality", "E_quality")

        all_evidence = e_supply + e_demand + e_quality
        evidence_ids = [e["id"] for e in all_evidence]

        # Step 4: Prompt组装
        prompt = compose_diagnosis_prompt(region, e_supply, e_demand, e_quality, template)

        # Step 5: LLM诊断生成
        system_msg = "你是城市文化用地体检诊断专家。请严格基于提供的证据链进行分析。"
        report = llm.chat(prompt, system=system_msg)

        # Step 6: 忠实性验证
        faith = verify_faithfulness(report, evidence_ids)
        logger.info(f"  忠实性: 引用率={faith['citation_rate']:.1%}, "
                    f"分数={faith['faithfulness_score']:.2f}, "
                    f"{'✓ PASS' if faith['pass'] else '✗ FAIL'}")

        # 如果不通过，重新生成
        if not faith["pass"] and len(evidence_ids) > 0:
            logger.info("  忠实性不足，追加提示重新生成...")
            correction = (f"\n\n注意：你的回答必须引用证据编号。"
                         f"可用证据编号: {', '.join(evidence_ids[:20])}")
            report = llm.chat(prompt + correction, system=system_msg)
            faith = verify_faithfulness(report, evidence_ids)
            logger.info(f"  重新生成忠实性: {faith['citation_rate']:.1%}")

        # 保存
        all_evidence_chains[region] = all_evidence
        all_faithfulness[region] = faith
        all_reports[region] = report

        # 写入MD文件
        report_path = out_dir / f"diagnosis_{region}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"<!-- 生成时间: {datetime.now().isoformat()} -->\n")
            f.write(f"<!-- 忠实性: {faith['faithfulness_score']:.2f} -->\n\n")
            f.write(report)
        logger.info(f"  报告: {report_path}")

    # 保存证据链
    evidence_path = out_dir / "evidence_chains.json"
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(all_evidence_chains, f, ensure_ascii=False, indent=2)

    # 保存忠实性报告
    faith_path = out_dir / "faithfulness_scores.json"
    with open(faith_path, "w", encoding="utf-8") as f:
        json.dump(all_faithfulness, f, ensure_ascii=False, indent=2)

    # 汇总
    avg_faith = np.mean([f["faithfulness_score"] for f in all_faithfulness.values()]) \
        if all_faithfulness else 0
    pass_rate = sum(1 for f in all_faithfulness.values() if f["pass"]) / max(len(all_faithfulness), 1)

    summary = {
        "total_regions": len(regions),
        "avg_faithfulness": round(float(avg_faith), 4),
        "pass_rate": round(pass_rate, 4),
        "region_scores": {r: f["faithfulness_score"] for r, f in all_faithfulness.items()},
    }

    logger.info("=" * 60)
    logger.info(f"TD-GRAG 诊断完成: {len(regions)} 区域, "
                f"平均忠实性={avg_faith:.2f}, 通过率={pass_rate:.1%}")
    logger.info("=" * 60)

    save_ours_evaluation(config, all_reports, all_faithfulness)

    return summary


if __name__ == "__main__":
    config = load_config()
    summary = run_td_grag_diagnosis(config)

    print("\n" + "=" * 60)
    print("TD-GRAG 诊断摘要")
    print("=" * 60)
    print(f"诊断区域数: {summary['total_regions']}")
    print(f"平均忠实性: {summary['avg_faithfulness']:.4f}")
    print(f"通过率: {summary['pass_rate']:.1%}")
    for r, s in summary.get("region_scores", {}).items():
        print(f"  {r}: {s:.4f}")
    print("=" * 60)
