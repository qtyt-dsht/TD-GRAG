"""
9_build_kg.py — 知识图谱构建与存储

功能：
  1. 读取 triplets_full.json，构建 NetworkX 内存图
  2. 统计图谱拓扑指标（节点数、边数、密度、连通分量等）
  3. 导出为 Neo4j 可导入的 CSV 格式
  4. （可选）直接写入 Neo4j
  5. 构建 FAISS 向量索引用于语义检索

输入: artifacts/v20260225/kg/triplets_full.json
输出: artifacts/v20260225/kg/kg_graph.gml
      artifacts/v20260225/kg/neo4j_nodes.csv
      artifacts/v20260225/kg/neo4j_edges.csv
      artifacts/v20260225/kg/kg_topology.json
"""
import json
import csv
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional
from collections import defaultdict, Counter

import yaml
import networkx as nx
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.geo_utils import parse_coord
from src.utils.kg_utils import merge_relation_distribution, normalize_relation_name
from src.utils.triplet_utils import PAPER_RELATIONS, load_triplets_file


def load_config() -> Dict[str, Any]:
    with open(ROOT / "config" / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# 图谱构建
# ──────────────────────────────────────────────
class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.node_types: Dict[str, str] = {}     # node_name -> type
        self.node_attrs: Dict[str, Dict] = {}    # node_name -> attributes

    def add_triplet(self, head: str, relation: str, tail: str,
                    confidence: float = 1.0, source_poi: str = "",
                    **extra):
        """添加一条三元组到图"""
        # 添加节点
        if head not in self.G:
            self.G.add_node(head)
        if tail not in self.G:
            self.G.add_node(tail)

        # 添加边
        self.G.add_edge(
            head, tail,
            relation=relation,
            confidence=confidence,
            source_poi=source_poi,
            **{k: v for k, v in extra.items() if isinstance(v, (str, int, float, bool))}
        )

    def set_node_type(self, name: str, node_type: str):
        """设置节点类型"""
        self.node_types[name] = node_type
        if name in self.G:
            self.G.nodes[name]["type"] = node_type

    def set_node_attributes(self, name: str, attrs: Dict):
        """设置节点属性"""
        self.node_attrs[name] = attrs
        if name in self.G:
            for k, v in attrs.items():
                if isinstance(v, (str, int, float, bool)):
                    self.G.nodes[name][k] = v

    def load_from_triplets(self, triplets: List[Dict]):
        """从三元组列表构建图"""
        for tri in triplets:
            self.add_triplet(
                head=tri.get("head", ""),
                relation=tri.get("relation", ""),
                tail=tri.get("tail", ""),
                confidence=tri.get("confidence", 1.0),
                source_poi=tri.get("source_poi", ""),
                poi_type=tri.get("poi_type", ""),
                source=tri.get("source", "")
            )

    def enrich_from_pois(self, pois: List[Dict]):
        """从POI数据补充节点属性"""
        for poi in pois:
            name = poi.get("名称", poi.get("中文名", ""))
            if not name:
                continue

            poi_type = poi.get("标准类型", poi.get("类型", ""))
            self.set_node_type(name, poi_type)

            attrs = {}
            coord = parse_coord(poi.get("坐标", ""))
            if coord:
                attrs["lat"] = coord[0]
                attrs["lon"] = coord[1]

            for field in ("区", "保护类型", "面积_sqm", "建成年份", "标准类型"):
                val = poi.get(field)
                if val is not None and str(val) not in ("None", "null", ""):
                    attrs[field] = val

            if attrs:
                self.set_node_attributes(name, attrs)

    def compute_topology(self) -> Dict[str, Any]:
        """计算图谱拓扑统计"""
        G = self.G
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # 转为无向图进行连通分量分析
        UG = G.to_undirected()
        components = list(nx.connected_components(UG))
        largest_cc = max(components, key=len) if components else set()

        # 度分布
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]

        # 关系类型分布
        rel_dist_raw = Counter()
        for _, _, data in G.edges(data=True):
            rel = data.get("relation", "unknown")
            rel_dist_raw[rel] += 1
        rel_dist_merged = Counter(merge_relation_distribution(rel_dist_raw))

        # 节点类型分布
        type_dist = Counter()
        for node in G.nodes():
            nt = G.nodes[node].get("type", self.node_types.get(node, "unknown"))
            type_dist[nt] += 1

        topology = {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "density": round(nx.density(G), 6) if n_nodes > 1 else 0,
            "num_connected_components": len(components),
            "largest_component_size": len(largest_cc),
            "largest_component_ratio": round(len(largest_cc) / max(n_nodes, 1), 4),
            "avg_in_degree": round(sum(in_degrees) / max(n_nodes, 1), 2),
            "avg_out_degree": round(sum(out_degrees) / max(n_nodes, 1), 2),
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "relation_distribution_raw": dict(rel_dist_raw.most_common()),
            "relation_distribution_merged": dict(rel_dist_merged.most_common()),
            "relation_distribution": dict(rel_dist_merged.most_common()),
            "node_type_distribution": dict(type_dist.most_common()),
        }
        return topology

    def export_neo4j_csv(self, out_dir: Path):
        """导出为 Neo4j LOAD CSV 格式"""
        out_dir.mkdir(parents=True, exist_ok=True)

        # 节点CSV
        nodes_path = out_dir / "neo4j_nodes.csv"
        with open(nodes_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id:ID", "name", "type", "lat:float", "lon:float",
                             "district", "area:float", "year:int"])
            for node in self.G.nodes():
                data = self.G.nodes[node]
                writer.writerow([
                    node,
                    node,
                    data.get("type", self.node_types.get(node, "")),
                    data.get("lat", ""),
                    data.get("lon", ""),
                    data.get("区", ""),
                    data.get("面积_sqm", ""),
                    data.get("建成年份", ""),
                ])
        logger.info(f"节点CSV: {nodes_path} ({self.G.number_of_nodes()} nodes)")

        # 边CSV
        edges_path = out_dir / "neo4j_edges.csv"
        with open(edges_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([":START_ID", ":END_ID", ":TYPE", "confidence:float",
                             "source_poi"])
            for u, v, data in self.G.edges(data=True):
                writer.writerow([
                    u, v,
                    data.get("relation", ""),
                    data.get("confidence", 1.0),
                    data.get("source_poi", ""),
                ])
        logger.info(f"边CSV: {edges_path} ({self.G.number_of_edges()} edges)")

    # 中文属性键 → ASCII 安全键的映射表
    _CN_KEY_MAP = {
        "类型": "type", "名称": "name", "区": "district", "地址": "address",
        "经度": "lng", "纬度": "lat", "朝代": "dynasty", "年代": "era",
        "面积": "area", "等级": "level", "描述": "description",
        "来源": "source", "权重": "weight", "关系": "relation",
        "置信度": "confidence", "方法": "method", "时间": "time",
        "评分": "score", "标签": "tag", "备注": "note",
    }

    @classmethod
    def _sanitize_key(cls, k: str) -> str:
        """将属性键转为GML兼容的ASCII键"""
        import re
        # 先查映射表
        if k in cls._CN_KEY_MAP:
            return cls._CN_KEY_MAP[k]
        # 替换常见符号
        k = k.replace("/", "_").replace("（", "").replace("）", "")
        k = k.replace("(", "").replace(")", "").replace(" ", "_")
        # 如果仍含非ASCII字符，用 'attr_' + hash 替代
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', k):
            safe = "attr_" + hashlib.md5(k.encode()).hexdigest()[:8]
            return safe
        return k

    def export_gml(self, path: Path):
        """导出为 GML 格式（NetworkX原生）"""
        # GML不支持中文属性名，需要清洗为ASCII键
        G_clean = nx.MultiDiGraph()
        for node in self.G.nodes():
            attrs = {}
            for k, v in self.G.nodes[node].items():
                clean_k = self._sanitize_key(k)
                if isinstance(v, (str, int, float)):
                    attrs[clean_k] = v
            G_clean.add_node(node, **attrs)

        for u, v, data in self.G.edges(data=True):
            attrs = {}
            for k, val in data.items():
                clean_k = self._sanitize_key(k)
                if isinstance(val, (str, int, float)):
                    attrs[clean_k] = val
            G_clean.add_edge(u, v, **attrs)

        nx.write_gml(G_clean, str(path))
        logger.info(f"GML: {path}")

    def try_load_neo4j(self, config: Dict[str, Any]) -> bool:
        """尝试将图写入Neo4j"""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.warning("neo4j驱动未安装，跳过Neo4j写入")
            return False

        uri = config.get("uri", "bolt://localhost:7687")
        user = config.get("user", "neo4j")
        password = config.get("password", "")

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                # 清空已有数据
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Neo4j已清空")

                # 批量创建节点
                for node in self.G.nodes():
                    props = {k: v for k, v in self.G.nodes[node].items()
                             if isinstance(v, (str, int, float, bool))}
                    props["name"] = node
                    node_type = props.pop("type", "Entity")
                    session.run(
                        f"CREATE (n:`{node_type}` $props)",
                        props=props
                    )

                # 批量创建关系
                for u, v, data in self.G.edges(data=True):
                    rel = data.get("relation", "RELATED")
                    props = {k: val for k, val in data.items()
                             if k != "relation" and isinstance(val, (str, int, float, bool))}
                    session.run(
                        f"MATCH (a {{name: $head}}), (b {{name: $tail}}) "
                        f"CREATE (a)-[:`{rel}` $props]->(b)",
                        head=u, tail=v, props=props
                    )

                logger.info(f"Neo4j写入完成: {self.G.number_of_nodes()} nodes, "
                            f"{self.G.number_of_edges()} edges")

            driver.close()
            return True

        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return False


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def run_build_kg(config: Dict[str, Any]) -> Dict[str, Any]:
    """构建知识图谱"""
    logger.info("=" * 60)
    logger.info("开始知识图谱构建")
    logger.info("=" * 60)

    # 加载三元组
    triplets_path = ROOT / config["kg"]["triplets_full"]
    if not triplets_path.exists():
        # 回退到原始三元组
        triplets_path = ROOT / config["data"]["triplets_raw"]
        logger.warning(f"triplets_full 不存在，回退到: {triplets_path}")

    triplets = load_triplets_file(
        triplets_path,
        valid_relations=PAPER_RELATIONS,
        default_source="build_kg"
    )

    logger.info(f"加载 {len(triplets)} 条三元组")

    # 构建图
    builder = KnowledgeGraphBuilder()
    builder.load_from_triplets(triplets)

    # 从POI数据补充节点属性
    clean_poi_path = ROOT / config["data"]["clean_poi"]
    if clean_poi_path.exists():
        with open(clean_poi_path, "r", encoding="utf-8") as f:
            pois = json.load(f)
        builder.enrich_from_pois(pois)
        logger.info(f"从 {len(pois)} 条POI补充节点属性")
    else:
        poi_raw_path = ROOT / config["data"]["poi_raw"]
        if poi_raw_path.exists():
            with open(poi_raw_path, "r", encoding="utf-8") as f:
                pois = json.load(f)
            builder.enrich_from_pois(pois)

    # 拓扑统计
    topology = builder.compute_topology()
    logger.info(f"图谱拓扑: {topology['num_nodes']} nodes, {topology['num_edges']} edges, "
                f"density={topology['density']}, components={topology['num_connected_components']}")

    # 输出
    out_dir = ROOT / config["kg"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 导出文件
    builder.export_neo4j_csv(out_dir)
    builder.export_gml(out_dir / "kg_graph.gml")

    # 保存拓扑统计
    topo_path = out_dir / "kg_topology.json"
    with open(topo_path, "w", encoding="utf-8") as f:
        json.dump(topology, f, ensure_ascii=False, indent=2)
    logger.info(f"拓扑统计: {topo_path}")

    # 尝试写入Neo4j
    neo4j_cfg = config.get("neo4j", {})
    if neo4j_cfg.get("uri"):
        builder.try_load_neo4j(neo4j_cfg)

    logger.info("=" * 60)
    logger.info("知识图谱构建完成")
    logger.info("=" * 60)

    return topology


if __name__ == "__main__":
    config = load_config()
    topology = run_build_kg(config)

    print("\n" + "=" * 60)
    print("知识图谱拓扑统计")
    print("=" * 60)
    for k, v in topology.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k2, v2 in v.items():
                print(f"  {k2}: {v2}")
        else:
            print(f"{k}: {v}")
    print("=" * 60)
