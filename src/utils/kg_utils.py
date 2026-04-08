#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Knowledge-graph relation normalization helpers."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable

from src.utils.triplet_utils import normalize_relation_name


RELATION_DISPLAY = {
    "locatedIn": "Located In",
    "managedBy": "Managed By",
    "providesService": "Provides Service",
    "hasCapacity": "Has Capacity",
    "receivedAward": "Received Award",
    "partnersWith": "Partners With",
    "servesDemographic": "Serves Demographic",
    "regulatedBy": "Regulated By",
}


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
