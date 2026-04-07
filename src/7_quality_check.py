"""
7_quality_check.py — 数据质量检测与修复 (Algorithm 3: Data Quality Pipeline)

功能：
  DQ-1: 面积字段缺失修复（LLM从百科摘要抽取）
  DQ-2: 建成/发现时间缺失修复（LLM从百科+历史信息抽取）
  DQ-5: 舆情数据跨城噪声过滤（城市关键词匹配）
  DQ-6: POI类型统一归类（6类Schema对齐）
  坐标有效性校验

输入：data/6_poi_info_final.json
输出：artifacts/v20260225/data/clean_poi.json
      artifacts/v20260225/data/clean_sentiment.json
      artifacts/v20260225/data/quality_report.json
"""
import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict
from copy import deepcopy

import yaml
from loguru import logger

# 添加项目根到 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.llm_client import LLMClient
from src.utils.sentiment import SentimentAnalyzer
from src.utils.geo_utils import parse_coord, haversine

# ──────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────
def load_config() -> Dict[str, Any]:
    """加载全局配置"""
    cfg_path = ROOT / "config" / "params.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ontology() -> Dict[str, Any]:
    """加载本体Schema"""
    onto_path = ROOT / "config" / "ontology.json"
    with open(onto_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# DQ-1: 面积缺失修复
# ──────────────────────────────────────────────
AREA_PROMPT = """请从以下文本中抽取该文化场所的**占地面积**信息。

文本：
{text}

要求：
1. 如果文本中明确提到面积数字，请直接提取。
2. 面积统一换算为**平方米**。
3. 如果是"亩"，请乘以 666.67 换算。
4. 如果是"公顷/ha"，请乘以 10000 换算。
5. 如果是"平方公里/km²"，请乘以 1000000 换算。
6. 如果文本中无任何面积信息，返回 null。

请仅返回 JSON 格式：{{"area_sqm": <数字或null>, "source": "<原文摘录>"}}"""


def repair_area(poi: Dict, llm: LLMClient) -> Tuple[Optional[float], str]:
    """
    DQ-1: 尝试从百科摘要/历史信息中抽取面积
    返回 (area_sqm, repair_method)
    """
    # 已有面积且有效
    raw_area = poi.get("面积")
    if raw_area and str(raw_area).strip() not in ("", "null", "None", "未知", "—"):
        try:
            # 尝试直接解析
            area_str = str(raw_area).replace(",", "").replace("，", "")
            # 处理 "xxx平方米" / "xxx㎡" 等
            m = re.search(r"([\d.]+)\s*(?:平方米|㎡|m²|m2)?", area_str)
            if m:
                val = float(m.group(1))
                # 检查是否需要单位换算
                if "万" in area_str:
                    val *= 10000
                if "公顷" in area_str or "ha" in area_str.lower():
                    val *= 10000
                if "亩" in area_str:
                    val *= 666.67
                if "平方公里" in area_str or "km" in area_str.lower():
                    val *= 1_000_000
                if val > 0:
                    return val, "原始数据"
        except (ValueError, TypeError):
            pass

    # 需要LLM修复
    text_parts = []
    for key in ("百科摘要", "百科标签", "历史信息"):
        v = poi.get(key)
        if v and str(v).strip() not in ("", "null", "None"):
            text_parts.append(f"【{key}】{v}")
    
    if not text_parts:
        return None, "无可用文本"

    prompt = AREA_PROMPT.format(text="\n".join(text_parts))
    resp = llm.chat(prompt, system="你是一个专业的地理信息抽取助手。")
    
    try:
        data = llm.extract_json(resp)
        if data and data.get("area_sqm") is not None:
            return float(data["area_sqm"]), "LLM抽取"
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    
    return None, "LLM未能抽取"


# ──────────────────────────────────────────────
# DQ-2: 建成时间缺失修复
# ──────────────────────────────────────────────
DATE_PROMPT = """请从以下文本中抽取该文化场所的**始建/建成/发现时间**。

文本：
{text}

要求：
1. 优先提取最早的始建年份。
2. 如果只有朝代（如"唐代"），请转换为近似公元年份。
3. 如果有"公元前"，请用负数表示。
4. 如果文本中无任何时间信息，返回 null。

朝代近似年份参考：
- 西周: -1046, 东周/春秋: -770, 战国: -475
- 秦: -221, 西汉: -206, 东汉: 25
- 三国: 220, 西晋: 265, 东晋: 317
- 南北朝: 420, 隋: 581, 唐: 618
- 五代十国: 907, 北宋: 960, 南宋: 1127
- 元: 1271, 明: 1368, 清: 1644
- 民国: 1912, 新中国: 1949

请仅返回 JSON 格式：{{"year": <整数或null>, "era": "<朝代或年代>", "source": "<原文摘录>"}}"""


def repair_date(poi: Dict, llm: LLMClient) -> Tuple[Optional[int], Optional[str], str]:
    """
    DQ-2: 尝试从百科摘要/历史信息中抽取建成时间
    返回 (year, era, repair_method)
    """
    # 已有时间且有效
    raw_time = poi.get("建成/发现时间") or poi.get("建成时间")
    if raw_time and str(raw_time).strip() not in ("", "null", "None", "未知", "—"):
        raw_str = str(raw_time)
        # 尝试提取年份
        m = re.search(r"(\d{3,4})", raw_str)
        if m:
            return int(m.group(1)), raw_str, "原始数据"

    # 需要LLM修复
    text_parts = []
    for key in ("百科摘要", "历史信息", "百科标签"):
        v = poi.get(key)
        if v and str(v).strip() not in ("", "null", "None"):
            text_parts.append(f"【{key}】{v}")
    
    if not text_parts:
        return None, None, "无可用文本"

    prompt = DATE_PROMPT.format(text="\n".join(text_parts))
    resp = llm.chat(prompt, system="你是一个专业的历史年代抽取助手。")
    
    try:
        data = llm.extract_json(resp)
        if data and data.get("year") is not None:
            return int(data["year"]), data.get("era"), "LLM抽取"
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    
    return None, None, "LLM未能抽取"


# ──────────────────────────────────────────────
# DQ-5: 舆情噪声过滤
# ──────────────────────────────────────────────
def filter_sentiment_noise(
    poi: Dict,
    xian_keywords: List[str],
    noise_cities: List[str],
    threshold: float = 0.3
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """
    DQ-5: 过滤舆情跨城噪声
    返回 (保留笔记列表, 删除笔记列表, 统计信息)
    """
    notes = poi.get("xiaohongshu", [])
    if not notes:
        return [], [], {"total": 0, "kept": 0, "removed": 0}

    kept, removed = [], []
    poi_name = poi.get("名称", poi.get("中文名", ""))

    for note in notes:
        title = note.get("标题", "")
        content = note.get("笔记内容", "")
        full_text = f"{title} {content}"

        # 计算西安相关性
        xian_hits = sum(1 for kw in xian_keywords if kw in full_text)
        # 计算噪声城市命中
        noise_hits = sum(1 for city in noise_cities if city in full_text)
        # POI名称命中
        poi_hit = 1 if poi_name and poi_name in full_text else 0

        # 相关性评分：西安关键词+POI名称 vs 噪声
        relevance = (xian_hits + poi_hit * 2) / max(xian_hits + poi_hit * 2 + noise_hits, 1)

        if relevance >= threshold or (xian_hits == 0 and noise_hits == 0 and poi_hit > 0):
            kept.append(note)
        elif noise_hits == 0:
            # 没有明确的噪声城市，保守保留
            kept.append(note)
        else:
            removed.append(note)

    stats = {
        "total": len(notes),
        "kept": len(kept),
        "removed": len(removed),
        "removal_rate": len(removed) / max(len(notes), 1)
    }
    return kept, removed, stats


# ──────────────────────────────────────────────
# DQ-6: POI类型统一
# ──────────────────────────────────────────────
TYPE_CLASSIFY_PROMPT = """请将以下文化场所归入6类之一：
1. 遗址遗迹类
2. 博物馆纪念馆类
3. 宗教场所类
4. 历史街区类
5. 文化公园/广场类
6. 其他文化设施类

场所信息：
- 名称：{name}
- 当前类型：{current_type}
- 百科标签：{tags}

请仅返回 JSON 格式：{{"type": "<6类之一>", "confidence": <0-1>, "reason": "<简要理由>"}}"""

VALID_TYPES = [
    "遗址遗迹类", "博物馆纪念馆类", "宗教场所类",
    "历史街区类", "文化公园/广场类", "其他文化设施类"
]


def classify_type(poi: Dict, llm: LLMClient) -> Tuple[str, float]:
    """
    DQ-6: 统一POI类型分类
    """
    current = poi.get("类型", "")
    # 如果已经是标准类型
    if current in VALID_TYPES:
        return current, 1.0

    # 简单规则映射
    rule_map = {
        "遗址": "遗址遗迹类", "遗迹": "遗址遗迹类", "古墓": "遗址遗迹类",
        "陵": "遗址遗迹类", "墓": "遗址遗迹类", "城墙": "遗址遗迹类",
        "博物": "博物馆纪念馆类", "纪念": "博物馆纪念馆类", "展览": "博物馆纪念馆类",
        "寺": "宗教场所类", "庙": "宗教场所类", "观": "宗教场所类",
        "教堂": "宗教场所类", "清真": "宗教场所类",
        "街": "历史街区类", "巷": "历史街区类", "坊": "历史街区类",
        "公园": "文化公园/广场类", "广场": "文化公园/广场类",
    }
    name = poi.get("名称", poi.get("中文名", ""))
    for keyword, typ in rule_map.items():
        if keyword in name or keyword in current:
            return typ, 0.9

    # 使用LLM分类
    prompt = TYPE_CLASSIFY_PROMPT.format(
        name=name,
        current_type=current,
        tags=poi.get("百科标签", "")
    )
    resp = llm.chat(prompt, system="你是一个文化遗产分类专家。")
    try:
        data = llm.extract_json(resp)
        if data and data.get("type") in VALID_TYPES:
            return data["type"], float(data.get("confidence", 0.7))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return "其他文化设施类", 0.5


# ──────────────────────────────────────────────
# 坐标校验
# ──────────────────────────────────────────────
def validate_coordinate(poi: Dict) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    校验坐标是否在西安范围内
    西安大致范围：纬度 33.4-35.0, 经度 107.4-109.8
    """
    coord_str = poi.get("坐标", "")
    coord = parse_coord(coord_str)
    if coord is None:
        return None, "坐标解析失败"
    
    lat, lon = coord
    if 33.4 <= lat <= 35.0 and 107.4 <= lon <= 109.8:
        return coord, "有效"
    else:
        return coord, f"坐标超出西安范围({lat:.4f},{lon:.4f})"


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def run_quality_check(config: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
    """
    执行完整的数据质量检测与修复流程

    Args:
        config: 全局配置字典
        use_llm: 是否使用LLM进行修复（False则仅做规则修复+统计）

    Returns:
        质量报告字典
    """
    logger.info("=" * 60)
    logger.info("开始数据质量检测 (Algorithm 3: Data Quality Pipeline)")
    logger.info("=" * 60)

    # 加载原始数据
    poi_path = ROOT / config["data"]["poi_raw"]
    logger.info(f"加载POI数据: {poi_path}")
    with open(poi_path, "r", encoding="utf-8") as f:
        pois_raw = json.load(f)
    logger.info(f"共 {len(pois_raw)} 条POI记录")

    # 加载本体
    ontology = load_ontology()
    xian_keywords = ontology.get("xian_city_keywords", [])
    noise_cities = [
        "北京", "上海", "广州", "深圳", "杭州", "南京", "成都",
        "重庆", "武汉", "天津", "苏州", "长沙", "厦门", "青岛"
    ]

    # 初始化LLM
    llm = None
    if use_llm:
        llm = LLMClient(config["llm"])
        logger.info(f"LLM已初始化: {config['llm']['provider']} / {config['llm']['model']}")

    # 输出路径
    out_dir = ROOT / Path(config["data"]["clean_poi"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 质量报告
    report = {
        "total_pois": len(pois_raw),
        "dq1_area": {"missing_before": 0, "missing_after": 0, "repaired": 0, "details": []},
        "dq2_date": {"missing_before": 0, "missing_after": 0, "repaired": 0, "details": []},
        "dq5_sentiment": {"total_notes": 0, "removed_notes": 0, "clean_notes": 0, "poi_stats": []},
        "dq6_type": {"reclassified": 0, "type_distribution": {}},
        "coord_check": {"valid": 0, "invalid": 0, "missing": 0},
    }

    clean_pois = []
    clean_sentiments = {}

    for idx, poi in enumerate(pois_raw):
        poi_name = poi.get("名称", poi.get("中文名", f"POI_{idx}"))
        logger.info(f"[{idx+1}/{len(pois_raw)}] 处理: {poi_name}")

        clean_poi = deepcopy(poi)

        # ── DQ-1: 面积修复 ──
        if use_llm:
            area_val, area_method = repair_area(poi, llm)
        else:
            area_val, area_method = repair_area.__wrapped__(poi) if hasattr(repair_area, '__wrapped__') else (None, "跳过LLM")
            # 仅做规则解析
            raw_area = poi.get("面积")
            if raw_area and str(raw_area).strip() not in ("", "null", "None", "未知", "—"):
                try:
                    area_str = str(raw_area).replace(",", "")
                    m = re.search(r"([\d.]+)", area_str)
                    if m:
                        area_val = float(m.group(1))
                        area_method = "规则解析"
                except (ValueError, TypeError):
                    area_val = None
                    area_method = "解析失败"

        original_area = poi.get("面积")
        has_original = original_area and str(original_area).strip() not in ("", "null", "None", "未知", "—")

        if not has_original:
            report["dq1_area"]["missing_before"] += 1

        if area_val is not None:
            clean_poi["面积_sqm"] = area_val
            clean_poi["面积_修复方法"] = area_method
            if not has_original:
                report["dq1_area"]["repaired"] += 1
        else:
            clean_poi["面积_sqm"] = None
            clean_poi["面积_修复方法"] = area_method
            if not has_original:
                report["dq1_area"]["missing_after"] += 1

        report["dq1_area"]["details"].append({
            "poi": poi_name,
            "original": str(original_area)[:50] if original_area else None,
            "repaired": area_val,
            "method": area_method
        })

        # ── DQ-2: 建成时间修复 ──
        if use_llm:
            year_val, era_val, date_method = repair_date(poi, llm)
        else:
            year_val, era_val, date_method = None, None, "跳过LLM"
            raw_time = poi.get("建成/发现时间") or poi.get("建成时间")
            if raw_time and str(raw_time).strip() not in ("", "null", "None", "未知", "—"):
                m = re.search(r"(\d{3,4})", str(raw_time))
                if m:
                    year_val = int(m.group(1))
                    era_val = str(raw_time)
                    date_method = "规则解析"

        original_time = poi.get("建成/发现时间") or poi.get("建成时间")
        has_original_time = original_time and str(original_time).strip() not in ("", "null", "None", "未知", "—")

        if not has_original_time:
            report["dq2_date"]["missing_before"] += 1

        if year_val is not None:
            clean_poi["建成年份"] = year_val
            clean_poi["建成朝代"] = era_val
            clean_poi["时间_修复方法"] = date_method
            if not has_original_time:
                report["dq2_date"]["repaired"] += 1
        else:
            clean_poi["建成年份"] = None
            clean_poi["建成朝代"] = None
            clean_poi["时间_修复方法"] = date_method
            if not has_original_time:
                report["dq2_date"]["missing_after"] += 1

        report["dq2_date"]["details"].append({
            "poi": poi_name,
            "original": str(original_time)[:50] if original_time else None,
            "year": year_val, "era": era_val,
            "method": date_method
        })

        # ── DQ-5: 舆情噪声过滤 ──
        kept_notes, removed_notes, sent_stats = filter_sentiment_noise(
            poi, xian_keywords, noise_cities
        )
        report["dq5_sentiment"]["total_notes"] += sent_stats["total"]
        report["dq5_sentiment"]["removed_notes"] += sent_stats["removed"]
        report["dq5_sentiment"]["clean_notes"] += sent_stats["kept"]

        if sent_stats["total"] > 0:
            report["dq5_sentiment"]["poi_stats"].append({
                "poi": poi_name,
                **sent_stats
            })

        # 存储清洗后的舆情
        clean_sentiments[poi_name] = {
            "poi_name": poi_name,
            "notes": kept_notes,
            "note_count": len(kept_notes),
            "removed_count": len(removed_notes)
        }
        # POI中也更新
        clean_poi["xiaohongshu_clean"] = kept_notes
        clean_poi["xiaohongshu_removed_count"] = len(removed_notes)

        # ── DQ-6: 类型统一 ──
        if use_llm:
            new_type, conf = classify_type(poi, llm)
        else:
            # 仅规则分类
            current = poi.get("类型", "")
            if current in VALID_TYPES:
                new_type, conf = current, 1.0
            else:
                new_type, conf = "其他文化设施类", 0.5
                rule_map = {
                    "遗址": "遗址遗迹类", "遗迹": "遗址遗迹类", "古墓": "遗址遗迹类",
                    "陵": "遗址遗迹类", "墓": "遗址遗迹类",
                    "博物": "博物馆纪念馆类", "纪念": "博物馆纪念馆类",
                    "寺": "宗教场所类", "庙": "宗教场所类",
                    "街": "历史街区类", "巷": "历史街区类",
                    "公园": "文化公园/广场类", "广场": "文化公园/广场类",
                }
                name = poi.get("名称", poi.get("中文名", ""))
                for keyword, typ in rule_map.items():
                    if keyword in name or keyword in current:
                        new_type, conf = typ, 0.9
                        break

        original_type = poi.get("类型", "")
        if new_type != original_type:
            report["dq6_type"]["reclassified"] += 1

        clean_poi["标准类型"] = new_type
        clean_poi["类型置信度"] = conf

        # ── 坐标校验 ──
        coord_result, coord_status = validate_coordinate(poi)
        if coord_result is not None:
            clean_poi["坐标_lat"] = coord_result[0]
            clean_poi["坐标_lon"] = coord_result[1]
            if "有效" in coord_status:
                report["coord_check"]["valid"] += 1
            else:
                report["coord_check"]["invalid"] += 1
                logger.warning(f"  坐标异常: {poi_name} - {coord_status}")
        else:
            clean_poi["坐标_lat"] = None
            clean_poi["坐标_lon"] = None
            report["coord_check"]["missing"] += 1

        clean_pois.append(clean_poi)

    # ── 统计类型分布 ──
    type_counter = Counter(p.get("标准类型", "未知") for p in clean_pois)
    report["dq6_type"]["type_distribution"] = dict(type_counter.most_common())

    # ── 质量门控 (Quality Checkpoint) ──
    total = report["total_pois"]
    area_missing_rate = report["dq1_area"]["missing_after"] / max(total, 1)
    date_missing_rate = report["dq2_date"]["missing_after"] / max(total, 1)
    total_notes = report["dq5_sentiment"]["total_notes"]
    clean_notes = report["dq5_sentiment"]["clean_notes"]
    city_consistency = clean_notes / max(total_notes, 1)

    report["quality_gates"] = {
        "area_missing_rate": round(area_missing_rate, 4),
        "area_gate_pass": area_missing_rate < 0.10,  # 目标 <10%
        "date_missing_rate": round(date_missing_rate, 4),
        "date_gate_pass": date_missing_rate < 0.15,  # 目标 <15%
        "city_consistency": round(city_consistency, 4),
        "city_gate_pass": city_consistency > 0.90,   # 目标 >90%
    }

    logger.info("=" * 60)
    logger.info("数据质量检测完成")
    logger.info(f"  面积缺失: {report['dq1_area']['missing_before']} → {report['dq1_area']['missing_after']} "
                f"(修复 {report['dq1_area']['repaired']})")
    logger.info(f"  时间缺失: {report['dq2_date']['missing_before']} → {report['dq2_date']['missing_after']} "
                f"(修复 {report['dq2_date']['repaired']})")
    logger.info(f"  舆情笔记: {total_notes} → {clean_notes} "
                f"(过滤 {report['dq5_sentiment']['removed_notes']})")
    logger.info(f"  类型重分: {report['dq6_type']['reclassified']}")
    logger.info(f"  坐标: 有效={report['coord_check']['valid']}, "
                f"异常={report['coord_check']['invalid']}, "
                f"缺失={report['coord_check']['missing']}")
    logger.info(f"  质量门控: area_pass={report['quality_gates']['area_gate_pass']}, "
                f"date_pass={report['quality_gates']['date_gate_pass']}, "
                f"city_pass={report['quality_gates']['city_gate_pass']}")
    logger.info("=" * 60)

    # ── 保存输出 ──
    clean_poi_path = ROOT / config["data"]["clean_poi"]
    clean_poi_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clean_poi_path, "w", encoding="utf-8") as f:
        json.dump(clean_pois, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存清洗POI: {clean_poi_path}")

    clean_sent_path = ROOT / config["data"]["clean_sentiment"]
    with open(clean_sent_path, "w", encoding="utf-8") as f:
        json.dump(clean_sentiments, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存清洗舆情: {clean_sent_path}")

    report_path = ROOT / config["data"]["quality_report"]
    # 精简 details 用于报告（避免文件过大）
    report_slim = deepcopy(report)
    report_slim["dq1_area"]["details"] = report_slim["dq1_area"]["details"][:10]  # 仅保留前10条示例
    report_slim["dq2_date"]["details"] = report_slim["dq2_date"]["details"][:10]
    report_slim["dq5_sentiment"]["poi_stats"] = report_slim["dq5_sentiment"]["poi_stats"][:10]

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_slim, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存质量报告: {report_path}")

    return report


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据质量检测与修复")
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过LLM调用，仅使用规则修复与统计")
    parser.add_argument("--config", default=None,
                        help="自定义配置文件路径")
    args = parser.parse_args()

    config = load_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f))

    report = run_quality_check(config, use_llm=not args.no_llm)

    # 打印摘要（使用 ASCII 安全字符，避免 Windows GBK 编码错误）
    import sys, io
    _out = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'buffer') else sys.stdout
    def _p(s):
        try:
            _out.write(s + "\n"); _out.flush()
        except Exception:
            print(s.encode('ascii', errors='replace').decode('ascii'))

    _p("\n" + "=" * 60)
    _p("数据质量检测摘要")
    _p("=" * 60)
    _p(f"总POI数: {report['total_pois']}")
    _p(f"面积缺失率: {report['quality_gates']['area_missing_rate']:.1%} "
       f"({'[PASS]' if report['quality_gates']['area_gate_pass'] else '[FAIL]'})")
    _p(f"时间缺失率: {report['quality_gates']['date_missing_rate']:.1%} "
       f"({'[PASS]' if report['quality_gates']['date_gate_pass'] else '[FAIL]'})")
    _p(f"舆情城市一致性: {report['quality_gates']['city_consistency']:.1%} "
       f"({'[PASS]' if report['quality_gates']['city_gate_pass'] else '[FAIL]'})")
    _p(f"类型分布: {report['dq6_type']['type_distribution']}")
    _p("=" * 60)
