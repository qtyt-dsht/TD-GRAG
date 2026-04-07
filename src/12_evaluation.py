#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12_evaluation.py — 对照实验 / 消融实验 / 鲁棒性检验 评估框架
============================================================
评估维度 (按 CCF 论文 §4.4):
  RQ1 — 知识图谱质量: Precision / Recall / F1 / Coverage
  RQ2 — 诊断质量: Evidence Completeness / Agreement(Kappa) / Faithfulness / Usefulness
  RQ3 — 问题识别: 异常识别 Precision / Recall / 问题分级 Weighted-F1

统计检验 (§4.5):
  H1: Ours vs Text-RAG  → 配对 t 检验
  H2: Ours vs A3(w/o Quality) → McNemar 检验
  H3: Ours vs A2(w/o Demand) → 配对 t 检验
"""

import json, csv, sys, argparse, logging, math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("evaluation")


# ================================================================
# §1  指标计算工具
# ================================================================

def precision_recall_f1(predicted: List[str], gold: List[str]) -> Dict[str, float]:
    """计算 Precision / Recall / F1"""
    pred_set = set(predicted)
    gold_set = set(gold)
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def cohens_kappa(ratings_a: List[int], ratings_b: List[int]) -> float:
    """Cohen's Kappa 一致性系数"""
    assert len(ratings_a) == len(ratings_b), "两组评分长度需一致"
    n = len(ratings_a)
    if n == 0:
        return 0.0
    labels = sorted(set(ratings_a) | set(ratings_b))
    k = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(ratings_a, ratings_b):
        matrix[label_idx[a]][label_idx[b]] += 1
    po = np.trace(matrix) / n
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (n * n)
    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return round(float(kappa), 4)


def evidence_completeness(report_text: str) -> float:
    """诊断报告证据链完整率 — 统计引用 [E_xxx] 的句子占比"""
    import re
    sentences = [s.strip() for s in re.split(r'[。！？\n]', report_text) if len(s.strip()) > 5]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r'\[E[_\w]+-\d+\]', s))
    return round(cited / len(sentences), 4)


def weighted_f1_multiclass(y_true: List[int], y_pred: List[int]) -> float:
    """加权 F1-Score（多分类）"""
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    if total == 0:
        return 0.0
    wf1 = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = sum(1 for t in y_true if t == label)
        wf1 += f1 * support / total
    return round(wf1, 4)


# ================================================================
# §2  统计检验
# ================================================================

def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Dict[str, float]:
    """配对 t 检验（H1, H3）"""
    from scipy import stats
    n = len(scores_a)
    assert n == len(scores_b) and n > 1, "样本量需一致且 > 1"
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_d = np.mean(diffs)
    std_d = np.std(diffs, ddof=1)
    if std_d == 0:
        return {
            "t_stat": 0.0,
            "p_value": 1.0,
            "mean_diff": round(float(mean_d), 4),
            "significant_005": False,
            "note": "paired differences are identical"
        }
    t_stat = mean_d / (std_d / math.sqrt(n))
    p_value = float(stats.t.sf(abs(t_stat), df=n - 1) * 2)  # two-tailed
    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(p_value, 6),
        "mean_diff": round(float(mean_d), 4),
        "significant_005": p_value < 0.05
    }


def mcnemar_test(pred_a: List[int], pred_b: List[int], gold: List[int]) -> Dict[str, float]:
    """McNemar 检验（H2）: 比较两种方法在同一金标准上的差异"""
    from scipy import stats
    n = len(gold)
    assert n == len(pred_a) == len(pred_b)
    # b: A对B错, c: A错B对
    b = sum(1 for i in range(n) if pred_a[i] == gold[i] and pred_b[i] != gold[i])
    c = sum(1 for i in range(n) if pred_a[i] != gold[i] and pred_b[i] == gold[i])
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant_005": False, "b": b, "c": c}
    # 连续校正
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(stats.chi2.sf(chi2, df=1))
    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(p_value, 6),
        "significant_005": p_value < 0.05,
        "b": b, "c": c
    }


# ================================================================
# §3  KG 质量评估 (RQ1)
# ================================================================

def evaluate_kg_quality(
    triplets_path: Path,
    gold_triplets_path: Optional[Path] = None,
    total_pois: int = 310
) -> Dict[str, Any]:
    """
    评估知识图谱质量
    - 自动: 覆盖率 = 图谱中POI / 总POI
    - 手动需提供 gold_triplets: P/R/F1
    """
    triplets = json.loads(triplets_path.read_text("utf-8")) if triplets_path.exists() else []
    # 统计覆盖率：只统计真实核心 POI，而不是把所有实体头节点都算成 POI。
    entities = set()
    for t in triplets:
        if t.get("head", ""):
            entities.add(t.get("head", ""))
        if t.get("tail", ""):
            entities.add(t.get("tail", ""))

    clean_poi_path = ROOT / "artifacts/v20260225/data/clean_poi.json"
    poi_name_set = set()
    if clean_poi_path.exists():
        try:
            clean_pois = json.loads(clean_poi_path.read_text("utf-8"))
            poi_name_set = {
                p.get("名称", p.get("中文名", ""))
                for p in clean_pois
                if p.get("名称", p.get("中文名", ""))
            }
        except Exception:
            poi_name_set = set()

    if poi_name_set:
        poi_names = set()
        for t in triplets:
            for key in ("source_poi", "head", "tail"):
                value = t.get(key, "")
                if value in poi_name_set:
                    poi_names.add(value)
    else:
        poi_names = set(t.get("head", "") for t in triplets if t.get("head", ""))
    coverage = len(poi_names) / total_pois if total_pois > 0 else 0.0

    result = {
        "total_triplets": len(triplets),
        "unique_entities": len(entities),
        "unique_pois_in_graph": len(poi_names),
        "coverage": round(coverage, 4),
    }

    # 如果有金标准
    if gold_triplets_path and gold_triplets_path.exists():
        gold_payload = json.loads(gold_triplets_path.read_text("utf-8"))
        if isinstance(gold_payload, dict):
            gold = gold_payload.get("triples", [])
            allowed_relations = set(gold_payload.get("allowed_relations", []))
            audited_poi_names = set(gold_payload.get("poi_names", []))
        else:
            gold = gold_payload
            allowed_relations = set()
            audited_poi_names = set()

        filtered_triplets = triplets
        if allowed_relations:
            filtered_triplets = [
                t for t in filtered_triplets
                if t.get("relation", "") in allowed_relations
            ]
        if audited_poi_names:
            filtered_triplets = [
                t for t in filtered_triplets
                if t.get("head", "") in audited_poi_names
            ]

        pred_keys = set(f"{t['head']}|{t['relation']}|{t['tail']}" for t in filtered_triplets)
        gold_keys = set(f"{t['head']}|{t['relation']}|{t['tail']}" for t in gold)
        prf = precision_recall_f1(list(pred_keys), list(gold_keys))
        result.update(prf)
        result["gold_scope_size"] = len(gold_keys)
        if allowed_relations:
            result["gold_scope_relations"] = sorted(allowed_relations)
    else:
        log.info("无金标准三元组文件，跳过 P/R/F1 计算（需人工抽检）")
        # 模拟：随机抽样估计
        result["precision_note"] = "需人工抽样100条验证"
        result["recall_note"] = "需对比金标准"

    return result


# ================================================================
# §4  诊断质量评估 (RQ2)
# ================================================================

def evaluate_diagnosis_quality(
    diagnosis_dir: Path,
    faithfulness_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    评估诊断报告质量
    - Evidence Completeness: 自动解析
    - Faithfulness: 读取已有验证结果
    """
    results = {"per_report": [], "avg_evidence_completeness": 0.0, "avg_faithfulness": 0.0}

    # 读取诊断报告
    report_files = sorted(diagnosis_dir.glob("diagnosis_*.md")) if diagnosis_dir.exists() else []
    if not report_files:
        log.warning(f"诊断目录 {diagnosis_dir} 无报告文件")
        return results

    completeness_scores = []
    for rf in report_files:
        text = rf.read_text("utf-8", errors="replace")
        ec = evidence_completeness(text)
        completeness_scores.append(ec)
        results["per_report"].append({
            "file": rf.name,
            "evidence_completeness": ec,
            "word_count": len(text)
        })

    results["avg_evidence_completeness"] = round(np.mean(completeness_scores), 4) if completeness_scores else 0.0
    results["num_reports"] = len(report_files)

    # 忠实度分数
    if faithfulness_path and faithfulness_path.exists():
        faith_data = json.loads(faithfulness_path.read_text("utf-8"))
        if isinstance(faith_data, dict):
            faith_scores = [
                v.get("score", v.get("faithfulness_score", 0))
                for v in faith_data.values()
                if isinstance(v, dict)
            ]
        else:
            faith_scores = [
                v.get("score", v.get("faithfulness_score", 0))
                for v in faith_data
                if isinstance(v, dict)
            ]
        results["avg_faithfulness"] = round(np.mean(faith_scores), 4) if faith_scores else 0.0
    else:
        results["avg_faithfulness_note"] = "需运行 11_td_grag_diagnosis.py 生成"

    return results


# ================================================================
# §5  问题识别评估 (RQ3) — 需人工标注
# ================================================================

def evaluate_anomaly_detection(
    predicted_anomalies: List[Dict],
    gold_anomalies: List[Dict],
) -> Dict[str, Any]:
    """
    评估异常识别能力
    predicted/gold 格式: [{"region": str, "severity": int(1-3), "type": str}]
    """
    # 区域级别匹配
    pred_regions = set(a["region"] for a in predicted_anomalies)
    gold_regions = set(a["region"] for a in gold_anomalies)
    region_prf = precision_recall_f1(list(pred_regions), list(gold_regions))

    # 严重程度分级
    matched_regions = pred_regions & gold_regions
    if matched_regions:
        pred_severity = {a["region"]: a.get("severity", 0) for a in predicted_anomalies if a["region"] in matched_regions}
        gold_severity = {a["region"]: a.get("severity", 0) for a in gold_anomalies if a["region"] in matched_regions}
        common = sorted(matched_regions)
        y_pred = [pred_severity.get(r, 0) for r in common]
        y_true = [gold_severity.get(r, 0) for r in common]
        severity_wf1 = weighted_f1_multiclass(y_true, y_pred)
    else:
        severity_wf1 = 0.0

    return {
        "anomaly_detection": region_prf,
        "severity_weighted_f1": severity_wf1,
        "num_predicted": len(predicted_anomalies),
        "num_gold": len(gold_anomalies),
    }


# ================================================================
# §6  消融实验框架
# ================================================================

ABLATION_CONFIGS = {
    "no_graph": {
        "name": "w/o Schema Constraint (A1)",
        "description": "移除CSGE的本体约束 → 开放式抽取",
        "modify": {"kg": {"min_confidence": 0.0}, "csge_schema_constraint": False}
    },
    "no_sentiment": {
        "name": "w/o Demand Dimension (A2)",
        "description": "移除舆情指标与需求侧子图",
        "modify": {"indicator": {"weights": {"supply": 0.50, "demand": 0.0, "quality": 0.50}},
                   "diagnosis": {"dimensions": ["supply", "quality"]}}
    },
    "no_policy": {
        "name": "w/o Quality Dimension (A3)",
        "description": "移除政策约束与质量侧子图",
        "modify": {"indicator": {"weights": {"supply": 0.50, "demand": 0.50, "quality": 0.0}},
                   "diagnosis": {"dimensions": ["supply", "demand"]}}
    },
    "no_spatial": {
        "name": "w/o Tri-dim Retrieval (A4)",
        "description": "用通用子图检索替代三维检索",
        "modify": {"diagnosis": {"use_generic_retrieval": True}}
    }
}


def generate_ablation_config(base_config: Dict, ablation_id: str) -> Dict:
    """为消融实验生成修改后的配置"""
    import copy
    if ablation_id not in ABLATION_CONFIGS:
        raise ValueError(f"未知消融ID: {ablation_id}, 可选: {list(ABLATION_CONFIGS.keys())}")
    config = copy.deepcopy(base_config)
    ablation = ABLATION_CONFIGS[ablation_id]

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v

    deep_update(config, ablation["modify"])
    config["_ablation_id"] = ablation_id
    config["_ablation_name"] = ablation["name"]
    return config


# ================================================================
# §7  对照实验收集与汇总
# ================================================================

def collect_baseline_results(eval_dir: Path) -> Dict[str, Any]:
    """
    收集所有 baseline 和消融实验的结果
    期望目录结构:
      eval_dir/
        ours/eval_metrics.json
        text_rag/eval_metrics.json
        no_retrieval_llm/eval_metrics.json
        rule_only/eval_metrics.json
        lightrag/eval_metrics.json
        ablation_no_graph/eval_metrics.json
        ...
    """
    all_results = {}
    if not eval_dir.exists():
        log.warning(f"评估目录不存在: {eval_dir}")
        return all_results

    for subdir in sorted(eval_dir.iterdir()):
        if subdir.is_dir():
            metrics_file = subdir / "eval_metrics.json"
            if metrics_file.exists():
                all_results[subdir.name] = json.loads(metrics_file.read_text("utf-8"))
                log.info(f"  加载 {subdir.name} 的评估结果")
    return all_results


def generate_comparison_table(all_results: Dict[str, Dict], output_path: Path):
    """生成对照实验汇总表 (CSV)"""
    if not all_results:
        log.warning("无结果可汇总")
        return

    # 收集所有指标键
    all_keys = set()
    for name, metrics in all_results.items():
        def _flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    _flatten(v, key)
                elif isinstance(v, (int, float)):
                    all_keys.add(key)
        _flatten(metrics)

    sorted_keys = sorted(all_keys)
    rows = []
    for name, metrics in all_results.items():
        row = {"method": name}
        def _flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    _flatten(v, key)
                elif isinstance(v, (int, float)):
                    row[key] = v
        _flatten(metrics)
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + sorted_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log.info(f"✅ 对照实验汇总表已保存: {output_path}")


# ================================================================
# §8  鲁棒性检验
# ================================================================

def robustness_cross_region(
    health_indices: Dict[str, List[float]],
) -> Dict[str, Any]:
    """不同区域抽样一致性检验 (§9.3)"""
    from scipy import stats
    regions = list(health_indices.keys())
    if len(regions) < 2:
        return {"note": "区域数不足，无法做一致性检验"}

    # Kruskal-Wallis H 检验 (非参数)
    groups = [health_indices[r] for r in regions]
    h_stat, p_value = stats.kruskal(*groups)
    # 变异系数
    all_vals = [v for g in groups for v in g]
    cv = float(np.std(all_vals) / np.mean(all_vals)) if np.mean(all_vals) > 0 else 0.0

    return {
        "kruskal_h": round(float(h_stat), 4),
        "kruskal_p": round(float(p_value), 6),
        "coefficient_of_variation": round(cv, 4),
        "num_regions": len(regions),
        "total_samples": len(all_vals),
    }


def robustness_repeat_runs(
    repeat_results: List[Dict[str, float]],
) -> Dict[str, Any]:
    """重复实验波动性检验 — 核心指标3次重复运行的标准差"""
    if not repeat_results:
        return {"note": "无重复实验结果"}

    metric_names = repeat_results[0].keys()
    stability = {}
    for metric in metric_names:
        values = [r.get(metric, 0) for r in repeat_results]
        stability[metric] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "cv": round(float(np.std(values) / np.mean(values)), 4) if np.mean(values) > 0 else 0.0,
            "stable_5pct": float(np.std(values) / np.mean(values)) < 0.05 if np.mean(values) > 0 else True,
        }
    return stability


# ================================================================
# §9  综合评估流水线
# ================================================================

def run_full_evaluation(config: Dict) -> Dict[str, Any]:
    """运行完整评估流水线"""
    eval_config = config.get("evaluation", {})
    eval_dir = ROOT / eval_config.get("output_dir", "artifacts/v20260225/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_dir = ROOT / config.get("diagnosis", {}).get("output_dir", "artifacts/v20260225/diagnosis")
    kg_dir = ROOT / config.get("kg", {}).get("output_dir", "artifacts/v20260225/kg")

    results = {
        "timestamp": datetime.now().isoformat(),
        "config_snapshot": {k: v for k, v in eval_config.items() if not k.startswith("_")},
    }

    # --- RQ1: KG 质量 ---
    log.info("="*60)
    log.info("RQ1: 知识图谱质量评估")
    log.info("="*60)
    triplets_path = ROOT / config.get("kg", {}).get("triplets_full", "artifacts/v20260225/kg/triplets_full.json")
    gold_path = eval_dir / "gold_triplets.json"  # 需人工创建
    kg_eval = evaluate_kg_quality(triplets_path, gold_path if gold_path.exists() else None)
    results["rq1_kg_quality"] = kg_eval
    log.info(f"  图谱覆盖率: {kg_eval.get('coverage', 'N/A')}")
    log.info(f"  总三元组数: {kg_eval.get('total_triplets', 0)}")

    # --- RQ2: 诊断质量 ---
    log.info("="*60)
    log.info("RQ2: 诊断质量评估")
    log.info("="*60)
    faith_path = diagnosis_dir / "faithfulness_scores.json"
    diag_eval = evaluate_diagnosis_quality(diagnosis_dir, faith_path if faith_path.exists() else None)
    ours_metrics_path = eval_dir / "ours" / "eval_metrics.json"
    if ours_metrics_path.exists():
        ours_metrics = json.loads(ours_metrics_path.read_text("utf-8"))
        diag_eval["avg_evidence_completeness"] = ours_metrics.get(
            "avg_evidence_completeness",
            diag_eval.get("avg_evidence_completeness", 0.0),
        )
        diag_eval["avg_faithfulness"] = ours_metrics.get(
            "avg_faithfulness",
            diag_eval.get("avg_faithfulness", 0.0),
        )
        diag_eval["num_reports"] = ours_metrics.get(
            "num_regions",
            diag_eval.get("num_reports", 0),
        )
        diag_eval["metric_source"] = str(ours_metrics_path.relative_to(ROOT))
    results["rq2_diagnosis_quality"] = diag_eval
    log.info(f"  报告数量: {diag_eval.get('num_reports', 0)}")
    log.info(f"  平均证据完整率: {diag_eval.get('avg_evidence_completeness', 'N/A')}")
    log.info(f"  平均忠实度: {diag_eval.get('avg_faithfulness', 'N/A')}")

    # --- RQ3: 问题识别（如有标注数据） ---
    log.info("="*60)
    log.info("RQ3: 问题识别评估")
    log.info("="*60)
    pred_anomaly_path = diagnosis_dir / "predicted_anomalies.json"
    gold_anomaly_path = eval_dir / "gold_anomalies.json"
    if pred_anomaly_path.exists() and gold_anomaly_path.exists():
        pred_anom = json.loads(pred_anomaly_path.read_text("utf-8"))
        gold_anom = json.loads(gold_anomaly_path.read_text("utf-8"))
        anom_eval = evaluate_anomaly_detection(pred_anom, gold_anom)
        results["rq3_anomaly_detection"] = anom_eval
    else:
        log.info("  暂无异常标注数据，跳过 RQ3")
        results["rq3_anomaly_detection"] = {"note": "需创建 gold_anomalies.json 标注文件"}

    # --- 对照实验汇总 ---
    log.info("="*60)
    log.info("对照实验结果汇总")
    log.info("="*60)
    baseline_results = collect_baseline_results(eval_dir)
    if baseline_results:
        results["comparison"] = baseline_results
        comparison_csv = eval_dir / "comparison_table.csv"
        generate_comparison_table(baseline_results, comparison_csv)
    else:
        log.info("  暂无 baseline 结果，需先运行 13_baselines.py")
        results["comparison"] = {"note": "运行 13_baselines.py 后再评估"}

    # --- 统计检验（如有对照数据） ---
    log.info("="*60)
    log.info("统计检验")
    log.info("="*60)
    stat_tests = {}

    # H1: Ours vs Text-RAG  证据链完整率
    ours_data = baseline_results.get("ours", {})
    text_rag_data = baseline_results.get("text_rag", {})
    if ours_data and text_rag_data:
        ours_ec = ours_data.get("evidence_completeness_scores", [])
        trag_ec = text_rag_data.get("evidence_completeness_scores", [])
        if ours_ec and trag_ec and len(ours_ec) == len(trag_ec):
            stat_tests["H1_ours_vs_textrag"] = paired_t_test(ours_ec, trag_ec)
            log.info(f"  H1 (Ours vs Text-RAG): p={stat_tests['H1_ours_vs_textrag']['p_value']}")

    # H2: Ours vs A3 (McNemar) — 需分类预测
    a3_data = baseline_results.get("ablation_no_policy", {})
    if ours_data and a3_data:
        ours_pred = ours_data.get("anomaly_predictions", [])
        a3_pred = a3_data.get("anomaly_predictions", [])
        gold_labels = ours_data.get("gold_labels", [])
        if ours_pred and a3_pred and gold_labels:
            stat_tests["H2_ours_vs_nopolicy"] = mcnemar_test(ours_pred, a3_pred, gold_labels)
            log.info(f"  H2 (Ours vs w/o Policy): p={stat_tests['H2_ours_vs_nopolicy']['p_value']}")

    # H3: Ours vs A2 — 异常识别召回率
    a2_data = baseline_results.get("ablation_no_sentiment", {})
    if ours_data and a2_data:
        ours_recall = ours_data.get("anomaly_recall_scores", [])
        a2_recall = a2_data.get("anomaly_recall_scores", [])
        if ours_recall and a2_recall and len(ours_recall) == len(a2_recall):
            stat_tests["H3_ours_vs_nosentiment"] = paired_t_test(ours_recall, a2_recall)
            log.info(f"  H3 (Ours vs w/o Sentiment): p={stat_tests['H3_ours_vs_nosentiment']['p_value']}")

    if not stat_tests:
        log.info("  统计检验需完成全部对照实验后运行")
        stat_tests["note"] = "需先运行所有 baseline 和消融实验"
    results["statistical_tests"] = stat_tests

    # --- 保存 ---
    output_path = eval_dir / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"\n✅ 评估结果已保存: {output_path}")

    # 生成简明摘要
    _print_summary(results)

    return results


def _print_summary(results: Dict):
    """打印评估摘要"""
    import sys
    import io
    # Windows cmd 可能是 GBK，强制 utf-8 输出避免 emoji 编码错误
    if hasattr(sys.stdout, 'buffer'):
        out = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    else:
        out = sys.stdout

    def _p(s: str):
        try:
            out.write(s + "\n")
            out.flush()
        except Exception:
            # 最终回退: 去掉非ASCII字符
            safe = s.encode('ascii', errors='replace').decode('ascii')
            print(safe)

    _p("\n" + "="*60)
    _p("[EVAL] CulturLand-Check 评估摘要")
    _p("="*60)

    rq1 = results.get("rq1_kg_quality", {})
    _p(f"\n[RQ1] 知识图谱质量")
    _p(f"  - 三元组总数: {rq1.get('total_triplets', 'N/A')}")
    _p(f"  - 实体总数: {rq1.get('unique_entities', 'N/A')}")
    _p(f"  - 图谱覆盖率: {rq1.get('coverage', 'N/A')}")
    if "precision" in rq1:
        _p(f"  - P/R/F1: {rq1['precision']}/{rq1['recall']}/{rq1['f1']}")

    rq2 = results.get("rq2_diagnosis_quality", {})
    _p(f"\n[RQ2] 诊断质量")
    _p(f"  - 诊断报告数: {rq2.get('num_reports', 'N/A')}")
    _p(f"  - 平均证据完整率: {rq2.get('avg_evidence_completeness', 'N/A')}")
    _p(f"  - 平均忠实度: {rq2.get('avg_faithfulness', 'N/A')}")

    rq3 = results.get("rq3_anomaly_detection", {})
    if "anomaly_detection" in rq3:
        ad = rq3["anomaly_detection"]
        _p(f"\n[RQ3] 问题识别")
        _p(f"  - 异常识别 P/R/F1: {ad.get('precision', 'N/A')}/{ad.get('recall', 'N/A')}/{ad.get('f1', 'N/A')}")
        _p(f"  - 分级加权F1: {rq3.get('severity_weighted_f1', 'N/A')}")

    tests = results.get("statistical_tests", {})
    if "note" not in tests:
        _p(f"\n[统计检验]")
        for key, val in tests.items():
            if isinstance(val, dict) and "p_value" in val:
                sig = "[Y]" if val.get("significant_005") else "[N]"
                _p(f"  - {key}: p={val['p_value']} {sig}")

    _p("\n" + "="*60)


# ================================================================
# main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CulturLand-Check 评估框架")
    parser.add_argument("--config", default="config/params.yaml")
    parser.add_argument("--mode", choices=["full", "kg", "diagnosis", "comparison"],
                        default="full", help="评估模式")
    args = parser.parse_args()

    import yaml
    config_path = ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.mode == "full":
        run_full_evaluation(config)
    elif args.mode == "kg":
        triplets_path = ROOT / config["kg"]["triplets_full"]
        result = evaluate_kg_quality(triplets_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "diagnosis":
        diagnosis_dir = ROOT / config["diagnosis"]["output_dir"]
        result = evaluate_diagnosis_quality(diagnosis_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "comparison":
        eval_dir = ROOT / config["evaluation"]["output_dir"]
        results = collect_baseline_results(eval_dir)
        generate_comparison_table(results, eval_dir / "comparison_table.csv")


if __name__ == "__main__":
    main()
