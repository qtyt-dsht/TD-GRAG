#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
13_baselines.py — 4 种 Baseline 方法实现
==========================================
按 CCF 论文 §4.2:
  B1: Rule-Only      — 规则引擎 + 阈值判断, 无 LLM
  B2: Vanilla-LLM    — 纯 LLM, 无检索增强
  B3: Text-RAG       — 文本分块 + TF-IDF 检索 + LLM
  B4: LightRAG       — 简化版 GraphRAG (通用社区检测 + LLM)

每个 baseline 输出同构的 eval_metrics.json, 便于 12_evaluation.py 汇总.
"""

import json, sys, argparse, logging, re, math
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("baselines")


# ================================================================
# 公共工具
# ================================================================

def load_config() -> Dict:
    import yaml
    with open(ROOT / "config/params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pois(config: Dict) -> List[Dict]:
    """加载 POI 数据（优先清洗后数据）"""
    clean_path = ROOT / config["data"].get("clean_poi", "")
    raw_path = ROOT / config["data"]["poi_raw"]
    path = clean_path if clean_path.exists() else raw_path
    return json.loads(path.read_text("utf-8"))


def get_region(poi: Dict) -> str:
    """统一读取行政区字段。"""
    for key in ("区", "行政区", "district"):
        value = poi.get(key, "")
        if value and str(value) not in ("[]", "None", "null", ""):
            return str(value)
    return ""


def get_poi_name(poi: Dict) -> str:
    for key in ("名称", "中文名", "name"):
        value = poi.get(key, "")
        if value and str(value) not in ("[]", "None", "null", ""):
            return str(value)
    return ""


def load_triplets(config: Dict) -> List[Dict]:
    full_path = ROOT / config["kg"].get("triplets_full", "")
    raw_path = ROOT / config["data"]["triplets_raw"]
    path = full_path if Path(full_path).exists() else raw_path

    data = json.loads(Path(path).read_text("utf-8"))
    # 兼容旧格式
    if data and isinstance(data[0], dict) and "relations" in data[0]:
        flat = []
        for item in data:
            name = item.get("entity_name", "")
            for rel in item.get("relations", []):
                flat.append({
                    "head": name,
                    "relation": rel.get("relation", ""),
                    "tail": rel.get("target", ""),
                    "confidence": rel.get("confidence", 0.8)
                })
        return flat
    return data


def load_indicators(config: Dict) -> Dict[str, Dict]:
    """加载指标数据"""
    import csv
    csv_path = ROOT / config["indicator"]["indicator_csv"]
    if not csv_path.exists():
        return {}
    result = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = row.get("region", row.get("区域", row.get("区", "")))
            if not region or region in ("[]", "None", ""):
                continue
            safe_vals = {}
            for k, v in row.items():
                if k in ("region", "区", "区域", "POI数", "面积km2"):
                    continue
                if v:
                    try:
                        safe_vals[k] = float(v)
                    except (ValueError, TypeError):
                        pass
            result[region] = safe_vals
    return result


def save_baseline_result(baseline_name: str, config: Dict, result: Dict):
    """保存 baseline 结果"""
    eval_dir = ROOT / config["evaluation"]["output_dir"] / baseline_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / "eval_metrics.json"
    result["timestamp"] = datetime.now().isoformat()
    result["baseline"] = baseline_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"✅ {baseline_name} 结果已保存: {output_path}")

    # 同时保存诊断报告
    if "reports" in result:
        for region, report in result["reports"].items():
            rp = eval_dir / f"diagnosis_{region}.md"
            rp.write_text(report, encoding="utf-8")


# ================================================================
# B1: Rule-Only — 规则引擎 + 阈值判断
# ================================================================

class RuleOnlyBaseline:
    """纯规则 baseline, 无 LLM 调用"""

    # 阈值参考值
    THRESHOLDS = {
        "density_low": 0.3,
        "coverage_low": 0.5,
        "sentiment_low": 0.4,
        "activity_low": 3.0,
    }

    def run(self, config: Dict) -> Dict:
        log.info("="*50)
        log.info("B1: Rule-Only Baseline")
        log.info("="*50)

        pois = load_pois(config)
        indicators = load_indicators(config)

        reports = {}
        diagnoses = []

        if not indicators:
            # 如果没有预计算指标, 使用 POI 简单统计
            log.info("  无预计算指标, 基于 POI 统计进行规则判断")
            region_pois = defaultdict(list)
            for p in pois:
                region = get_region(p)
                if region:
                    region_pois[region].append(p)

            for region, region_poi_list in region_pois.items():
                n = len(region_poi_list)
                issues = []
                severity = 1  # 1=轻, 2=中, 3=重

                if n < 10:
                    issues.append(f"文化用地数量不足 ({n}处)")
                    severity = max(severity, 2)

                # 类型多样性
                types = set(p.get("type", "") for p in region_poi_list)
                if len(types) < 3:
                    issues.append(f"类型多样性不足 (仅{len(types)}类)")

                # 百科覆盖
                baike_count = sum(1 for p in region_poi_list if p.get("百科摘要") or p.get("baike_summary"))
                if n > 0 and baike_count / n < 0.6:
                    issues.append(f"信息完整度低 (百科覆盖{baike_count}/{n})")

                report = f"# {region} 规则诊断报告\n\n"
                report += f"## 基本统计\n- 文化用地数量: {n}\n- 类型数: {len(types)}\n\n"
                if issues:
                    report += "## 发现问题\n"
                    for iss in issues:
                        report += f"- ⚠️ {iss}\n"
                    severity = min(3, severity)
                else:
                    report += "## 评估结果\n- ✅ 未发现明显问题\n"

                reports[region] = report
                diagnoses.append({"region": region, "severity": severity, "issues": issues})
        else:
            for region, ind in indicators.items():
                issues = []
                severity = 1

                density = ind.get("density", ind.get("density_D_k", 0))
                coverage = ind.get("coverage", ind.get("coverage_C_k", 0))
                sentiment = ind.get("sentiment_score", ind.get("sentiment", 0.5))
                activity = ind.get("activity", ind.get("Act_k", 0))

                if density < self.THRESHOLDS["density_low"]:
                    issues.append(f"设施密度偏低 ({density:.3f})")
                    severity = max(severity, 2)
                if coverage < self.THRESHOLDS["coverage_low"]:
                    issues.append(f"服务覆盖不足 ({coverage:.3f})")
                    severity = max(severity, 2)
                if sentiment < self.THRESHOLDS["sentiment_low"]:
                    issues.append(f"公众满意度较低 ({sentiment:.3f})")
                    severity = max(severity, 3)
                if activity < self.THRESHOLDS["activity_low"]:
                    issues.append(f"舆情活跃度不足 ({activity:.1f})")

                report = f"# {region} 规则诊断报告\n\n"
                report += f"## 指标概览\n"
                for k, v in ind.items():
                    report += f"- {k}: {v:.4f}\n"
                report += "\n"
                if issues:
                    report += "## 发现问题\n"
                    for iss in issues:
                        report += f"- ⚠️ {iss}\n"
                else:
                    report += "## 评估结果\n- ✅ 各项指标正常\n"

                reports[region] = report
                diagnoses.append({"region": region, "severity": severity, "issues": issues})

        return {
            "method": "B1_Rule_Only",
            "num_regions": len(reports),
            "reports": reports,
            "diagnoses": diagnoses,
            "evidence_completeness_scores": [0.0] * len(reports),  # 规则无证据链
            "avg_evidence_completeness": 0.0,
        }


# ================================================================
# B2: Vanilla-LLM — 纯 LLM, 无检索增强
# ================================================================

class VanillaLLMBaseline:
    """直接将所有信息喂给 LLM, 不做结构化检索"""

    def run(self, config: Dict) -> Dict:
        log.info("="*50)
        log.info("B2: Vanilla-LLM Baseline (No Retrieval)")
        log.info("="*50)

        from src.utils.llm_client import LLMClient
        llm = LLMClient(config["llm"])
        pois = load_pois(config)
        indicators = load_indicators(config)

        # 按区域分组
        region_pois = defaultdict(list)
        for p in pois:
            region = get_region(p)
            if region:
                region_pois[region].append(p)

        reports = {}
        ec_scores = []

        for region, region_poi_list in region_pois.items():
            log.info(f"  诊断: {region} ({len(region_poi_list)} POIs)")

            # 构造简单的 POI 列表文本
            poi_text = ""
            for p in region_poi_list[:20]:  # 限制长度
                name = p.get("名称", p.get("name", ""))
                poi_type = p.get("type", p.get("类型", ""))
                summary = (p.get("百科摘要", p.get("baike_summary", "")) or "")[:100]
                poi_text += f"- {name} ({poi_type}): {summary}\n"

            # 指标摘要
            ind_text = ""
            if region in indicators:
                for k, v in indicators[region].items():
                    ind_text += f"- {k}: {v:.4f}\n"

            prompt = f"""你是城市文化用地体检专家。请根据以下信息对{region}的文化用地进行诊断。

## 文化用地列表
{poi_text}

## 指标数据
{ind_text if ind_text else '暂无指标数据'}

请输出包含以下内容的诊断报告:
1. 总体评价
2. 供给侧分析（设施密度、覆盖）
3. 需求侧分析（舆情、满意度）
4. 质量侧分析（保护合规、文化价值）
5. 改进建议
"""
            try:
                response = llm.chat([
                    {"role": "system", "content": "你是城市规划与文化遗产保护领域的诊断专家。"},
                    {"role": "user", "content": prompt}
                ], temperature=0.3)
                reports[region] = f"# {region} 诊断报告 (Vanilla-LLM)\n\n{response}"
            except Exception as e:
                log.error(f"  LLM调用失败: {e}")
                reports[region] = f"# {region}\n\n[LLM调用失败: {e}]"

            # 无证据链引用
            ec_scores.append(0.0)

        return {
            "method": "B2_Vanilla_LLM",
            "num_regions": len(reports),
            "reports": reports,
            "evidence_completeness_scores": ec_scores,
            "avg_evidence_completeness": 0.0,
            "llm_stats": llm.get_stats(),
        }


# ================================================================
# B3: Text-RAG — 文本分块 + TF-IDF 检索 + LLM
# ================================================================

class TextRAGBaseline:
    """标准 RAG: 文本分块 → TF-IDF向量检索 → LLM生成"""

    def __init__(self, chunk_size: int = 300, top_k: int = 5):
        self.chunk_size = chunk_size
        self.top_k = top_k

    def _build_corpus(self, pois: List[Dict]) -> List[Dict]:
        """构建文本块语料库"""
        chunks = []
        for p in pois:
            name = p.get("名称", p.get("name", ""))
            region = p.get("行政区", p.get("district", "未知"))
            poi_type = p.get("type", p.get("类型", ""))
            summary = p.get("百科摘要", p.get("baike_summary", "")) or ""

            # 基本信息块
            basic_text = f"{name}是{region}的{poi_type}。"
            if summary:
                basic_text += summary[:self.chunk_size]
            chunks.append({"text": basic_text, "source": name, "region": region, "type": "basic"})

            # 舆情块
            xhs = p.get("xiaohongshu", [])
            for note in xhs[:5]:
                note_text = f"关于{name}的舆情: {note.get('标题', '')} {note.get('内容', '')}"
                if len(note_text) > 30:
                    chunks.append({"text": note_text[:self.chunk_size], "source": name, "region": region, "type": "sentiment"})

        return chunks

    def _tfidf_retrieve(self, query: str, corpus: List[Dict], top_k: int = 5) -> List[Dict]:
        """TF-IDF 检索"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [c["text"] for c in corpus]
        if not texts:
            return []

        try:
            vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b")
            tfidf_matrix = vectorizer.fit_transform(texts + [query])
            query_vec = tfidf_matrix[-1]
            doc_vecs = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vec, doc_vecs).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:
                    chunk = corpus[idx].copy()
                    chunk["score"] = float(similarities[idx])
                    results.append(chunk)
            return results
        except Exception as e:
            log.warning(f"  TF-IDF检索失败: {e}")
            return corpus[:top_k]

    def run(self, config: Dict) -> Dict:
        log.info("="*50)
        log.info("B3: Text-RAG Baseline")
        log.info("="*50)

        from src.utils.llm_client import LLMClient
        llm = LLMClient(config["llm"])
        pois = load_pois(config)
        indicators = load_indicators(config)

        # 构建语料库
        corpus = self._build_corpus(pois)
        log.info(f"  语料库: {len(corpus)} 文本块")

        region_pois = defaultdict(list)
        for p in pois:
            region = get_region(p)
            if region:
                region_pois[region].append(p)

        reports = {}
        ec_scores = []

        for region in region_pois.keys():
            log.info(f"  诊断: {region}")

            query = f"{region} 文化用地 设施分布 保护现状 公众评价"
            retrieved = self._tfidf_retrieve(query, corpus, self.top_k)

            # 构造 RAG 上下文
            context = "## 检索到的相关信息\n\n"
            for i, chunk in enumerate(retrieved, 1):
                context += f"[R{i}] ({chunk['source']}) {chunk['text'][:200]}\n\n"

            ind_text = ""
            if region in indicators:
                for k, v in indicators[region].items():
                    ind_text += f"- {k}: {v:.4f}\n"

            prompt = f"""你是城市文化用地体检专家。请根据检索到的信息对{region}的文化用地进行诊断。
引用时使用 [Rn] 格式标注信息来源。

{context}

## 指标数据
{ind_text if ind_text else '暂无'}

请输出包含以下内容的诊断报告:
1. 总体评价 (引用检索信息)
2. 供给侧分析
3. 需求侧分析
4. 质量侧分析
5. 改进建议
"""
            try:
                response = llm.chat([
                    {"role": "system", "content": "你是城市规划与文化遗产保护领域的诊断专家。回答时引用[Rn]来源。"},
                    {"role": "user", "content": prompt}
                ], temperature=0.3)
                report = f"# {region} 诊断报告 (Text-RAG)\n\n{response}"
                reports[region] = report

                # 计算引用覆盖率 (Text-RAG 用 [Rn] 而非 [E_xxx])
                sentences = [s.strip() for s in re.split(r'[。！？\n]', response) if len(s.strip()) > 5]
                cited = sum(1 for s in sentences if re.search(r'\[R\d+\]', s))
                ec = cited / len(sentences) if sentences else 0.0
                ec_scores.append(round(ec, 4))
            except Exception as e:
                log.error(f"  LLM调用失败: {e}")
                reports[region] = f"# {region}\n\n[LLM调用失败: {e}]"
                ec_scores.append(0.0)

        return {
            "method": "B3_Text_RAG",
            "num_regions": len(reports),
            "reports": reports,
            "corpus_size": len(corpus),
            "evidence_completeness_scores": ec_scores,
            "avg_evidence_completeness": round(float(np.mean(ec_scores)), 4) if ec_scores else 0.0,
            "llm_stats": llm.get_stats(),
        }


# ================================================================
# B4: LightRAG — 简化版 GraphRAG
# ================================================================

class LightRAGBaseline:
    """
    简化的 GraphRAG baseline:
    - 用图谱但不做维度感知检索
    - 通用社区检测 (connected components) 替代三维检索
    - 类似 Microsoft GraphRAG 的全局/局部模式
    """

    def __init__(self, max_hops: int = 2, top_k: int = 10):
        self.max_hops = max_hops
        self.top_k = top_k

    def _generic_subgraph_retrieval(
        self, triplets: List[Dict], query_region: str, top_k: int = 10
    ) -> List[Dict]:
        """通用子图检索 — 不区分维度, 简单关键词匹配"""
        relevant = []
        for t in triplets:
            head = t.get("head", "")
            tail = t.get("tail", "")
            if query_region in head or query_region in tail:
                relevant.append(t)
            elif any(kw in head or kw in tail for kw in [query_region, "西安"]):
                relevant.append(t)

        # 按置信度排序取 top-k
        relevant.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
        return relevant[:top_k]

    def _build_evidence_text(self, subgraph: List[Dict]) -> str:
        """将子图转为文本"""
        if not subgraph:
            return "（无检索结果）"
        lines = []
        for i, t in enumerate(subgraph, 1):
            lines.append(f"[G{i}] {t['head']} --[{t['relation']}]--> {t['tail']} (conf={t.get('confidence', 'N/A')})")
        return "\n".join(lines)

    def run(self, config: Dict) -> Dict:
        log.info("="*50)
        log.info("B4: LightRAG Baseline (Generic GraphRAG)")
        log.info("="*50)

        from src.utils.llm_client import LLMClient
        llm = LLMClient(config["llm"])
        pois = load_pois(config)
        triplets = load_triplets(config)
        indicators = load_indicators(config)

        log.info(f"  图谱三元组: {len(triplets)}")

        region_pois = defaultdict(list)
        for p in pois:
            region = get_region(p)
            if region:
                region_pois[region].append(p)

        reports = {}
        ec_scores = []

        for region in region_pois.keys():
            log.info(f"  诊断: {region}")

            # 通用子图检索（不分维度）
            subgraph = self._generic_subgraph_retrieval(triplets, region, self.top_k)
            evidence_text = self._build_evidence_text(subgraph)

            ind_text = ""
            if region in indicators:
                for k, v in indicators[region].items():
                    ind_text += f"- {k}: {v:.4f}\n"

            prompt = f"""你是城市文化用地体检专家。请根据以下知识图谱信息对{region}的文化用地进行诊断。
引用时使用 [Gn] 格式标注图谱证据来源。

## 图谱检索结果
{evidence_text}

## 指标数据
{ind_text if ind_text else '暂无'}

请输出包含以下内容的诊断报告:
1. 总体评价 (引用图谱证据)
2. 设施供给分析
3. 公众需求分析
4. 保护质量分析
5. 改进建议
"""
            try:
                response = llm.chat([
                    {"role": "system", "content": "你是城市规划与文化遗产保护领域的诊断专家。回答时引用[Gn]来源。"},
                    {"role": "user", "content": prompt}
                ], temperature=0.3)
                report = f"# {region} 诊断报告 (LightRAG)\n\n{response}"
                reports[region] = report

                # 计算引用覆盖率
                sentences = [s.strip() for s in re.split(r'[。！？\n]', response) if len(s.strip()) > 5]
                cited = sum(1 for s in sentences if re.search(r'\[G\d+\]', s))
                ec = cited / len(sentences) if sentences else 0.0
                ec_scores.append(round(ec, 4))
            except Exception as e:
                log.error(f"  LLM调用失败: {e}")
                reports[region] = f"# {region}\n\n[LLM调用失败: {e}]"
                ec_scores.append(0.0)

        return {
            "method": "B4_LightRAG",
            "num_regions": len(reports),
            "reports": reports,
            "triplets_used": len(triplets),
            "evidence_completeness_scores": ec_scores,
            "avg_evidence_completeness": round(float(np.mean(ec_scores)), 4) if ec_scores else 0.0,
            "llm_stats": llm.get_stats(),
        }


# ================================================================
# 消融实验运行器
# ================================================================

def run_ablation(ablation_id: str, config: Dict) -> Dict:
    """运行单个消融实验"""
    import importlib
    eval_mod = importlib.import_module("src.12_evaluation")
    generate_ablation_config = eval_mod.generate_ablation_config
    ABLATION_CONFIGS = eval_mod.ABLATION_CONFIGS

    if ablation_id not in ABLATION_CONFIGS:
        log.error(f"未知消融ID: {ablation_id}")
        return {}

    ablation_info = ABLATION_CONFIGS[ablation_id]
    log.info(f"运行消融: {ablation_info['name']} — {ablation_info['description']}")

    # 生成消融配置
    ablation_config = generate_ablation_config(config, ablation_id)

    # 用消融配置运行诊断流水线的简化版
    from src.utils.llm_client import LLMClient
    llm = LLMClient(ablation_config["llm"])
    pois = load_pois(ablation_config)
    triplets = load_triplets(ablation_config)
    indicators = load_indicators(ablation_config)

    diag_config = ablation_config.get("diagnosis", {})
    dimensions = diag_config.get("dimensions", ["supply", "demand", "quality"])
    use_generic = diag_config.get("use_generic_retrieval", False)

    region_pois = defaultdict(list)
    for p in pois:
        region = get_region(p)
        if region:
            region_pois[region].append(p)

    reports = {}
    ec_scores = []

    for region in region_pois.keys():
        log.info(f"  消融诊断 ({ablation_id}): {region}")

        # 根据消融设置构造上下文
        context_parts = []
        region_poi_names = {get_poi_name(p) for p in region_pois[region] if get_poi_name(p)}
        region_triplets = [
            t for t in triplets
            if t.get("source_poi", "") in region_poi_names
            or t.get("head", "") in region_poi_names
            or t.get("tail", "") in region_poi_names
            or t.get("tail", "") == region
        ]

        if "supply" in dimensions:
            supply_trips = [
                t for t in region_triplets
                if t.get("relation", "") in ("位于", "邻近", "同区", "同类", "属于", "locate_in", "has_type")
            ][:8]
            if supply_trips:
                ctx = "[供给侧证据]\n"
                for i, t in enumerate(supply_trips, 1):
                    ctx += f"[E_supply-{i:03d}] {t['head']} --{t['relation']}--> {t['tail']}\n"
                context_parts.append(ctx)

        if "demand" in dimensions:
            demand_text = "[需求侧证据]\n"
            demand_trips = [
                t for t in region_triplets
                if t.get("relation") in ("关注于",)
                and (t.get("tail", "") in region_poi_names or t.get("head", "") in region_poi_names)
            ][:8]
            for i, t in enumerate(demand_trips, 1):
                demand_text += f"[E_demand-{i:03d}] {t['head']} --{t['relation']}--> {t['tail']}\n"
            if demand_trips:
                context_parts.append(demand_text)

        if "quality" in dimensions:
            quality_trips = [
                t for t in region_triplets
                if t.get("relation", "") in ("受约束于", "始建于")
            ][:8]
            if quality_trips:
                ctx = "[质量侧证据]\n"
                for i, t in enumerate(quality_trips, 1):
                    ctx += f"[E_quality-{i:03d}] {t['head']} --{t['relation']}--> {t['tail']}\n"
                context_parts.append(ctx)

        ind_text = ""
        if region in indicators:
            for k, v in indicators[region].items():
                ind_text += f"- {k}: {v:.4f}\n"

        prompt = f"""你是城市文化用地体检专家。请根据以下证据对{region}的文化用地进行诊断。
每句结论必须引用证据编号[E_xxx-nnn]。

{"".join(context_parts) if context_parts else "（无可用证据）"}

## 指标数据
{ind_text if ind_text else '暂无'}

请输出诊断报告:
1. 总体评价
{"2. 供给侧诊断" if "supply" in dimensions else ""}
{"3. 需求侧诊断" if "demand" in dimensions else ""}
{"4. 质量侧诊断" if "quality" in dimensions else ""}
5. 改进建议
"""
        try:
            response = llm.chat([
                {"role": "system", "content": "你是城市规划领域诊断专家。每句结论须引用[E_xxx-nnn]编号。"},
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            report = f"# {region} 消融诊断 ({ablation_info['name']})\n\n{response}"
            reports[region] = report

            sentences = [s.strip() for s in re.split(r'[。！？\n]', response) if len(s.strip()) > 5]
            cited = sum(1 for s in sentences if re.search(r'\[E[_\w]+-\d+\]', s))
            ec = cited / len(sentences) if sentences else 0.0
            ec_scores.append(round(ec, 4))
        except Exception as e:
            log.error(f"  消融诊断失败: {e}")
            reports[region] = f"# {region}\n\n[失败: {e}]"
            ec_scores.append(0.0)

    return {
        "method": f"Ablation_{ablation_id}",
        "ablation_name": ablation_info["name"],
        "ablation_description": ablation_info["description"],
        "dimensions_used": dimensions,
        "num_regions": len(reports),
        "reports": reports,
        "evidence_completeness_scores": ec_scores,
        "avg_evidence_completeness": round(float(np.mean(ec_scores)), 4) if ec_scores else 0.0,
        "llm_stats": llm.get_stats(),
    }


# ================================================================
# main
# ================================================================

BASELINE_MAP = {
    "rule_only": RuleOnlyBaseline,
    "no_retrieval_llm": VanillaLLMBaseline,
    "text_rag": TextRAGBaseline,
    "lightrag": LightRAGBaseline,
}


def main():
    parser = argparse.ArgumentParser(description="CulturLand-Check Baseline 方法")
    parser.add_argument("--config", default="config/params.yaml")
    parser.add_argument("--baseline", choices=list(BASELINE_MAP.keys()) + ["all"],
                        default="all", help="运行哪个 baseline")
    parser.add_argument("--ablation", choices=["no_graph", "no_sentiment", "no_policy", "no_spatial", "all", "none"],
                        default="none", help="运行哪个消融实验")
    parser.add_argument("--no-llm", action="store_true", help="仅运行无需 LLM 的 baseline")
    args = parser.parse_args()

    config = load_config()

    # --- 运行 Baselines ---
    if args.baseline == "all":
        baselines_to_run = list(BASELINE_MAP.keys())
    else:
        baselines_to_run = [args.baseline]

    if args.no_llm:
        baselines_to_run = [b for b in baselines_to_run if b == "rule_only"]

    for bname in baselines_to_run:
        log.info(f"\n{'#'*60}")
        log.info(f"# 运行 Baseline: {bname}")
        log.info(f"{'#'*60}")
        baseline_cls = BASELINE_MAP[bname]
        baseline = baseline_cls()
        result = baseline.run(config)
        save_baseline_result(bname, config, result)

    # --- 运行消融实验 ---
    if args.ablation != "none":
        ablation_ids = list(ABLATION_MAP := {"no_graph": True, "no_sentiment": True, "no_policy": True, "no_spatial": True})
        if args.ablation != "all":
            ablation_ids = [args.ablation]

        if args.no_llm:
            log.info("--no-llm 模式下跳过消融实验")
        else:
            for aid in ablation_ids:
                result = run_ablation(aid, config)
                save_baseline_result(f"ablation_{aid}", config, result)

    log.info("\n✅ 所有实验运行完成")


if __name__ == "__main__":
    main()
