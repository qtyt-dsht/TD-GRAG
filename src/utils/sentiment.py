"""
情感分析模块 - 基于 SnowNLP 的中文情感分析 + 城市相关性过滤
"""
import re
from typing import List, Dict, Any, Optional, Tuple

from loguru import logger

try:
    from snownlp import SnowNLP
    HAS_SNOWNLP = True
except ImportError:
    HAS_SNOWNLP = False
    logger.warning("snownlp 未安装，将使用简化情感分析")


class SentimentAnalyzer:
    """中文情感分析器"""

    def __init__(self, method: str = "snownlp", city_keywords: Optional[List[str]] = None):
        """
        初始化情感分析器

        Args:
            method: 'snownlp' 或 'llm'
            city_keywords: 城市相关关键词列表
        """
        self.method = method
        self.city_keywords = city_keywords or [
            "西安", "长安", "碑林", "雁塔", "莲湖", "新城区", "未央", "灞桥",
            "长安区", "临潼", "阎良", "高陵", "鄠邑", "蓝田", "周至",
            "曲江", "高新区", "经开区", "浐灞", "大雁塔", "钟楼", "鼓楼",
            "城墙", "兵马俑", "华清池", "大明宫", "秦始皇", "陕西",
        ]
        # 负面关键词列表（增强判断）
        self.negative_keywords = [
            "差", "烂", "糟糕", "失望", "不好", "脏", "乱", "破", "坑",
            "不推荐", "避雷", "踩雷", "无语", "垃圾", "不值", "后悔",
            "太贵", "拥挤", "排队", "商业化", "过度开发",
        ]
        self.positive_keywords = [
            "好", "棒", "推荐", "值得", "赞", "美", "漂亮", "震撼",
            "惊艳", "壮观", "必去", "喜欢", "舒服", "不错", "满意",
            "精彩", "宝藏", "绝美", "治愈", "免费",
        ]

    def score(self, text: str) -> float:
        """
        计算文本情感分数

        Args:
            text: 中文文本

        Returns:
            情感分数，范围 [-1, 1]，正值为积极，负值为消极
        """
        if not text or not text.strip():
            return 0.0

        if self.method == "snownlp" and HAS_SNOWNLP:
            try:
                s = SnowNLP(text)
                # SnowNLP 返回 [0, 1]，映射到 [-1, 1]
                raw_score = s.sentiments
                return round(raw_score * 2 - 1, 4)
            except Exception:
                pass

        # 回退到关键词方法
        return self._keyword_score(text)

    def _keyword_score(self, text: str) -> float:
        """基于关键词的简化情感分析"""
        pos_count = sum(1 for kw in self.positive_keywords if kw in text)
        neg_count = sum(1 for kw in self.negative_keywords if kw in text)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return round((pos_count - neg_count) / total, 4)

    def is_city_relevant(self, text: str, poi_name: str = "", threshold: float = 0.3) -> bool:
        """
        判断文本是否与目标城市(西安)相关

        Args:
            text: 文本内容
            poi_name: POI名称（如果文本提到POI名称也算相关）
            threshold: 相关性阈值

        Returns:
            是否相关
        """
        if not text:
            return False

        full_text = text.lower()

        # 如果提到POI名称
        if poi_name and poi_name in full_text:
            return True

        # 检查城市关键词
        hit_count = sum(1 for kw in self.city_keywords if kw in full_text)
        # 检查是否提到其他城市（噪声检测）
        other_cities = [
            "北京", "上海", "广州", "深圳", "成都", "杭州", "南京",
            "武汉", "重庆", "天津", "苏州", "长沙", "郑州",
        ]
        other_count = sum(1 for c in other_cities if c in full_text)

        # 如果提到其他城市但没提到西安相关词汇
        if other_count > 0 and hit_count == 0:
            return False

        # 如果有任何西安关键词匹配
        if hit_count > 0:
            return True

        # 默认保留（可能是POI特有内容）
        return True

    def analyze_notes(
        self,
        notes: List[Dict[str, Any]],
        poi_name: str = "",
        filter_noise: bool = True,
    ) -> Dict[str, Any]:
        """
        分析一组小红书笔记的情感

        Args:
            notes: 笔记列表 [{"标题": ..., "笔记内容": ..., "点赞": ..., "评论": [...]}]
            poi_name: POI名称
            filter_noise: 是否过滤跨城噪声

        Returns:
            {
                "total_notes": int,
                "clean_notes": int,
                "filtered_notes": int,
                "avg_sentiment": float,
                "sentiment_std": float,
                "activity": int,
                "avg_likes": float,
                "note_sentiments": [{"title": str, "sentiment": float, "relevant": bool}]
            }
        """
        results = {
            "total_notes": len(notes),
            "clean_notes": 0,
            "filtered_notes": 0,
            "avg_sentiment": 0.0,
            "sentiment_std": 0.0,
            "activity": 0,
            "avg_likes": 0.0,
            "note_sentiments": [],
        }

        sentiments = []
        likes_sum = 0

        for note in notes:
            title = note.get("标题", "")
            content = note.get("笔记内容", "")
            likes_str = note.get("点赞", "0")
            comments = note.get("评论", [])

            # 解析点赞数
            try:
                likes = int(str(likes_str).replace(",", ""))
            except (ValueError, TypeError):
                likes = 0

            full_text = f"{title} {content}"

            # 城市相关性过滤
            relevant = self.is_city_relevant(full_text, poi_name)
            if filter_noise and not relevant:
                results["filtered_notes"] += 1
                results["note_sentiments"].append({
                    "title": title,
                    "sentiment": 0.0,
                    "relevant": False,
                })
                continue

            # 情感分析
            sent = self.score(full_text)

            # 评论情感也参与（权重较低）
            comment_sents = []
            for c in comments:
                c_text = c.get("内容", "")
                if c_text:
                    comment_sents.append(self.score(c_text))

            if comment_sents:
                # 笔记情感权重0.7 + 评论平均情感权重0.3
                avg_comment = sum(comment_sents) / len(comment_sents)
                sent = 0.7 * sent + 0.3 * avg_comment

            sentiments.append(sent)
            likes_sum += likes
            results["clean_notes"] += 1
            results["note_sentiments"].append({
                "title": title,
                "sentiment": round(sent, 4),
                "relevant": True,
            })

        if sentiments:
            import numpy as np
            results["avg_sentiment"] = round(float(np.mean(sentiments)), 4)
            results["sentiment_std"] = round(float(np.std(sentiments)), 4)
            results["activity"] = len(sentiments)
            results["avg_likes"] = round(likes_sum / len(sentiments), 2)

        return results

    def batch_analyze(
        self,
        pois: List[Dict[str, Any]],
        filter_noise: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量分析所有POI的舆情

        Args:
            pois: POI列表（每个包含 "xiaohongshu" 字段）
            filter_noise: 是否过滤噪声

        Returns:
            每个POI的情感分析结果列表
        """
        results = []
        total_filtered = 0

        for poi in pois:
            poi_name = poi.get("名称", poi.get("name", ""))
            notes = poi.get("xiaohongshu", [])
            analysis = self.analyze_notes(notes, poi_name, filter_noise)
            analysis["poi_name"] = poi_name
            analysis["region"] = poi.get("区", "")
            analysis["type"] = poi.get("类型", "")
            results.append(analysis)
            total_filtered += analysis["filtered_notes"]

        logger.info(
            f"情感分析完成: {len(results)} POI, "
            f"总计过滤 {total_filtered} 条噪声笔记"
        )
        return results
