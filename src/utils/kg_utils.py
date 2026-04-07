#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Knowledge-graph relation normalization helpers."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable


RELATION_ALIASES = {
    "has_type": "has_type",
    "has_类型": "has_type",
    "has_类别": "has_type",
    "has_所属": "has_type",
    "locate_in": "locate_in",
    "has_区": "locate_in",
    "has_所属地区": "locate_in",
    "has_下辖地区": "locate_in",
    "has_address": "has_address",
    "has_地理位置": "has_address",
    "has_coordinates": "has_coordinates",
    "has_坐标": "has_coordinates",
    "has_建成/发现时间": "has_built_time",
    "has_建成时间": "has_built_time",
    "has_建立时间": "has_built_time",
    "has_建造时间": "has_built_time",
    "has_竣工时间": "has_built_time",
    "has_始建时间": "has_built_time",
    "has_始建年代": "has_built_time",
    "has_投用时间": "has_built_time",
    "has_成立时间": "has_built_time",
    "has_面积": "has_area",
    "has_占地面积": "has_area",
    "has_建筑面积": "has_area",
    "has_展厅面积": "has_area",
    "has_总面积": "has_area",
    "has_总建面积": "has_area",
    "has_水域面积": "has_area",
    "has_门票价格": "has_ticket_price",
    "has_门票": "has_ticket_price",
    "has_景点级别": "has_level",
    "has_景区级别": "has_level",
    "has_景区类型": "has_level",
    "has_博物馆等级": "has_level",
    "has_museum_level": "has_level",
    "has_grade": "has_level",
    "has_保护级别": "has_level",
    "has_文保级别": "has_level",
    "has_文物级别": "has_level",
    "has_文物保护等级": "has_level",
    "has_森林公园级别": "has_level",
    "has_开放时间": "has_open_time",
    "has_适宜游玩季节": "has_best_season",
    "has_建议游玩时长": "has_recommended_duration",
    "has_feature": "has_feature",
    "has_attribute": "has_feature",
    "has_特点": "has_feature",
    "has_遗址特点": "has_feature",
    "has_景点特色": "has_feature",
    "has_核心景观": "has_feature",
    "has_文化古迹": "has_feature",
    "has_人文景观": "has_feature",
    "has_景观": "has_feature",
    "has_百科摘要": "has_summary",
    "has_百科历史信息": "has_summary",
    "has_历史信息": "has_summary",
    "has_百科标签": "has_tags",
    "has_honor": "has_honor",
    "has_荣誉": "has_honor",
    "located_next_to": "nearby",
}


RELATION_DISPLAY = {
    "has_type": "Type / Category",
    "locate_in": "Located In / District",
    "has_address": "Address / Location Text",
    "has_coordinates": "Coordinates",
    "has_built_time": "Built / Found Time",
    "has_area": "Area",
    "has_open_time": "Open Time",
    "has_ticket_price": "Ticket Price",
    "has_level": "Level / Grade",
    "has_feature": "Feature / Attribute",
    "has_best_season": "Best Season",
    "has_recommended_duration": "Recommended Duration",
    "has_summary": "Summary / History",
    "has_tags": "Tags",
    "has_honor": "Honor / Award",
    "nearby": "Nearby",
}


def normalize_relation_name(relation: str) -> str:
    relation = str(relation).strip()
    return RELATION_ALIASES.get(relation, relation)


def merge_relation_distribution(distribution: Dict[str, int] | Counter | Iterable[str]) -> Dict[str, int]:
    counter = Counter()
    if isinstance(distribution, dict):
        for relation, value in distribution.items():
            counter[normalize_relation_name(relation)] += int(value)
    else:
        for relation in distribution:
            counter[normalize_relation_name(str(relation))] += 1
    return dict(counter.most_common())


def relation_display_name(relation: str) -> str:
    return RELATION_DISPLAY.get(relation, relation)
