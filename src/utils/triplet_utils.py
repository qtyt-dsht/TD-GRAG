#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Triplet normalization and backward-compatible loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RELATION_SIGNATURES: Dict[str, Dict[str, List[str]]] = {
    "locatedIn": {"head": ["VenueOrSite"], "tail": ["District"]},
    "managedBy": {"head": ["VenueOrSite"], "tail": ["Institution"]},
    "providesService": {"head": ["VenueOrSite"], "tail": ["Service"]},
    "hasCapacity": {"head": ["VenueOrSite"], "tail": ["Service"]},
    "receivedAward": {"head": ["VenueOrSite"], "tail": ["Award"]},
    "partnersWith": {"head": ["VenueOrSite"], "tail": ["Organization"]},
    "servesDemographic": {"head": ["VenueOrSite"], "tail": ["DemographicGroup"]},
    "regulatedBy": {"head": ["District", "VenueOrSite"], "tail": ["PolicyDocument"]},
}

PAPER_RELATIONS = set(RELATION_SIGNATURES.keys())


RELATION_ALIASES: Dict[str, str] = {
    "locatedIn": "locatedIn",
    "位于": "locatedIn",
    "locate_in": "locatedIn",
    "located_in": "locatedIn",
    "managedBy": "managedBy",
    "managed_by": "managedBy",
    "运营方": "managedBy",
    "管理方": "managedBy",
    "providesService": "providesService",
    "provides_service": "providesService",
    "提供服务": "providesService",
    "serviceProvided": "providesService",
    "hasCapacity": "hasCapacity",
    "has_capacity": "hasCapacity",
    "capacity": "hasCapacity",
    "容量": "hasCapacity",
    "服务能力": "hasCapacity",
    "receivedAward": "receivedAward",
    "received_award": "receivedAward",
    "has_honor": "receivedAward",
    "has_grade": "receivedAward",
    "has_museum_level": "receivedAward",
    "荣誉": "receivedAward",
    "奖项": "receivedAward",
    "等级": "receivedAward",
    "partnersWith": "partnersWith",
    "partners_with": "partnersWith",
    "associated_with": "partnersWith",
    "合作机构": "partnersWith",
    "合作方": "partnersWith",
    "servesDemographic": "servesDemographic",
    "serves_demographic": "servesDemographic",
    "服务人群": "servesDemographic",
    "面向人群": "servesDemographic",
    "regulatedBy": "regulatedBy",
    "regulated_by": "regulatedBy",
    "受约束于": "regulatedBy",
    "governedBy": "regulatedBy",
}


ENTITY_TYPE_ALIASES: Dict[str, str] = {
    "VenueOrSite": "VenueOrSite",
    "POI": "VenueOrSite",
    "poi": "VenueOrSite",
    "poi_name": "VenueOrSite",
    "Venue": "VenueOrSite",
    "Site": "VenueOrSite",
    "district": "District",
    "District": "District",
    "行政区": "District",
    "district_name": "District",
    "Institution": "Institution",
    "institution": "Institution",
    "Service": "Service",
    "service": "Service",
    "CapacityValue": "Service",
    "Capacity": "Service",
    "capacity": "Service",
    "Award": "Award",
    "award": "Award",
    "Honor": "Award",
    "Recognition": "Award",
    "Organization": "Organization",
    "organization": "Organization",
    "DemographicGroup": "DemographicGroup",
    "demographic": "DemographicGroup",
    "PopulationGroup": "DemographicGroup",
    "PolicyDocument": "PolicyDocument",
    "policy": "PolicyDocument",
    "Policy": "PolicyDocument",
    "Plan": "PolicyDocument",
    "Regulation": "PolicyDocument",
}


def normalize_relation_name(relation: Any) -> str:
    key = str(relation or "").strip()
    return RELATION_ALIASES.get(key, key)


def normalize_entity_type(entity_type: Any) -> str:
    key = str(entity_type or "").strip()
    return ENTITY_TYPE_ALIASES.get(key, key)


def relation_dimension(relation: str) -> str:
    if relation == "locatedIn":
        return "anchor"
    if relation in ("managedBy", "providesService", "hasCapacity"):
        return "supply"
    if relation in ("servesDemographic", "regulatedBy"):
        return "demand"
    if relation in ("receivedAward", "partnersWith"):
        return "quality"
    return ""


def _safe_float(value: Any, default: float = 0.8) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_types(relation: str, head_type: str = "", tail_type: str = "", head: str = "", tail: str = "") -> tuple[str, str]:
    signature = RELATION_SIGNATURES.get(relation, {})
    if not head_type:
        head_type = signature.get("head", [""])[0]
    if not tail_type:
        tail_type = signature.get("tail", [""])[0]
    if relation == "regulatedBy" and head_type == "VenueOrSite" and head and head == tail:
        head_type = "District"
    return head_type, tail_type


def canonicalize_triplet(
    item: Dict[str, Any],
    *,
    source_poi: str = "",
    poi_type: str = "",
    source: str = "",
    valid_relations: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    head = str(item.get("head", item.get("subject", "")) or "").strip()
    tail = str(item.get("tail", item.get("object", item.get("target", ""))) or "").strip()
    relation = normalize_relation_name(item.get("relation", item.get("predicate", item.get("relation_type", ""))))

    if not head or not tail or not relation:
        return None
    if valid_relations is not None and relation not in set(valid_relations):
        return None

    head_type = normalize_entity_type(item.get("head_type", item.get("subject_type", "")))
    tail_type = normalize_entity_type(item.get("tail_type", item.get("object_type", item.get("target_type", ""))))
    head_type, tail_type = _infer_types(relation, head_type, tail_type, head, tail)

    triplet = {
        "head": head,
        "head_type": head_type,
        "relation": relation,
        "tail": tail,
        "tail_type": tail_type,
        "confidence": round(_safe_float(item.get("confidence", 0.8), 0.8), 3),
        "source_poi": str(item.get("source_poi", source_poi or head)).strip(),
        "poi_type": str(item.get("poi_type", poi_type or "")).strip(),
        "source": str(item.get("source", source or "")).strip(),
    }

    evidence_text = str(item.get("evidence_text", item.get("evidence", item.get("source_text", ""))) or "").strip()
    source_corpus = str(item.get("source_corpus", item.get("corpus", "")) or "").strip()
    diagnostic_dimension = str(item.get("diagnostic_dimension", relation_dimension(relation)) or "").strip()

    if evidence_text:
        triplet["evidence_text"] = evidence_text
    if source_corpus:
        triplet["source_corpus"] = source_corpus
    if diagnostic_dimension:
        triplet["diagnostic_dimension"] = diagnostic_dimension

    return triplet


def flatten_triplet_payload(
    payload: Any,
    *,
    valid_relations: Optional[Iterable[str]] = None,
    default_source: str = "",
) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        return []

    flat: List[Dict[str, Any]] = []
    allowed = set(valid_relations) if valid_relations is not None else None
    for item in payload:
        if not isinstance(item, dict):
            continue

        if any(key in item for key in ("head", "subject", "tail", "object")):
            triplet = canonicalize_triplet(item, source=default_source, valid_relations=allowed)
            if triplet:
                flat.append(triplet)
            continue

        if "relations" not in item:
            continue

        anchor_name = str(item.get("entity_name", item.get("name", item.get("source_poi", ""))) or "").strip()
        anchor_type = str(item.get("entity_type", item.get("poi_type", "")) or "").strip()
        for rel in item.get("relations", []):
            if not isinstance(rel, dict):
                continue
            candidate = {
                "head": anchor_name,
                "head_type": anchor_type,
                "relation": rel.get("relation", rel.get("relation_type", "")),
                "tail": rel.get("tail", rel.get("target", rel.get("entity_name", ""))),
                "tail_type": rel.get("tail_type", rel.get("entity_type", "")),
                "confidence": rel.get("confidence", 0.8),
                "source_poi": item.get("source_poi", anchor_name),
                "poi_type": item.get("poi_type", item.get("entity_type", "")),
                "source": default_source or "legacy_triplets"
            }
            triplet = canonicalize_triplet(candidate, valid_relations=allowed)
            if triplet:
                flat.append(triplet)

    return flat


def load_triplets_file(
    path: Path,
    *,
    valid_relations: Optional[Iterable[str]] = None,
    default_source: str = "",
) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text("utf-8"))
    return flatten_triplet_payload(payload, valid_relations=valid_relations, default_source=default_source)
