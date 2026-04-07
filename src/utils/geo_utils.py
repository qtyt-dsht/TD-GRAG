"""
地理空间工具模块 - 提供坐标计算、空间查询、GIS分析功能
"""
import json
import math
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
from loguru import logger

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.ops import unary_union
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    logger.warning("geopandas/shapely 未安装，空间分析功能不可用")


EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两个经纬度点之间的 Haversine 距离 (km)"""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def parse_coord(coord_str: str) -> Optional[Tuple[float, float]]:
    """解析坐标字符串 'lat,lon' 格式，返回 (lat, lon) 或 None"""
    if not coord_str or not isinstance(coord_str, str):
        return None
    try:
        parts = coord_str.strip().split(",")
        if len(parts) == 2:
            lat, lon = float(parts[0]), float(parts[1])
            if 25.0 <= lat <= 45.0 and 100.0 <= lon <= 120.0:
                return (lat, lon)
    except (ValueError, TypeError):
        pass
    return None


def find_neighbors(
    pois: List[Dict[str, Any]],
    distance_km: float = 2.0,
) -> List[Tuple[str, str, float]]:
    """
    查找互为邻近的 POI 对

    Args:
        pois: POI列表，每项需包含 '名称' 和 '坐标'
        distance_km: 邻近距离阈值 (km)

    Returns:
        列表 [(poi_name_a, poi_name_b, distance_km)]
    """
    pairs = []
    coords = []
    for poi in pois:
        c = parse_coord(poi.get("坐标", ""))
        coords.append(c)

    n = len(pois)
    for i in range(n):
        if coords[i] is None:
            continue
        for j in range(i + 1, n):
            if coords[j] is None:
                continue
            dist = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if dist <= distance_km:
                name_i = pois[i].get("名称", pois[i].get("name", f"POI_{i}"))
                name_j = pois[j].get("名称", pois[j].get("name", f"POI_{j}"))
                pairs.append((name_i, name_j, round(dist, 3)))

    logger.info(f"发现 {len(pairs)} 对邻近POI（阈值 {distance_km}km）")
    return pairs


class GeoUtils:
    """地理空间分析工具类"""

    def __init__(self, geojson_path: Optional[str] = None):
        """
        初始化地理工具

        Args:
            geojson_path: 城市边界 GeoJSON 文件路径
        """
        self.city_boundary = None
        self.districts = {}

        if geojson_path and HAS_GEO:
            try:
                gdf = gpd.read_file(geojson_path)
                self.city_boundary = gdf
                logger.info(f"加载城市边界: {len(gdf)} 个要素")
            except Exception as e:
                logger.warning(f"加载GeoJSON失败: {e}")

    def point_in_city(self, lat: float, lon: float) -> bool:
        """判断点是否在城市范围内"""
        if self.city_boundary is None:
            # 简化判断: 西安市大致范围
            return 33.5 <= lat <= 34.8 and 107.4 <= lon <= 109.8
        try:
            point = Point(lon, lat)  # shapely 使用 (lon, lat)
            return any(self.city_boundary.contains(point))
        except Exception:
            return False

    def compute_density(
        self,
        pois: List[Dict[str, Any]],
        region_areas: Dict[str, float],
    ) -> Dict[str, float]:
        """
        计算各区域设施密度 D_k = N_k / A_k

        Args:
            pois: POI列表
            region_areas: 区域面积字典 {区名: 面积(km²)}

        Returns:
            {区名: 密度值}
        """
        region_counts = {}
        for poi in pois:
            region = poi.get("区", "")
            if region:
                region_counts[region] = region_counts.get(region, 0) + 1

        density = {}
        for region, count in region_counts.items():
            area = region_areas.get(region, 1.0)
            density[region] = round(count / area, 4) if area > 0 else 0
        return density

    def compute_coverage(
        self,
        pois: List[Dict[str, Any]],
        radius_m: float = 1000,
        grid_resolution: float = 0.01,
    ) -> Dict[str, float]:
        """
        计算各区域服务覆盖率 C_k (向量化版本)

        Args:
            pois: POI列表
            radius_m: 服务半径 (米)
            grid_resolution: 网格分辨率 (度), 默认0.01≈1km

        Returns:
            {区名: 覆盖率}
        """
        radius_km = radius_m / 1000.0
        # 近似: 1度纬度≈111km, 1度经度≈111*cos(lat)km
        # 用度数近似阈值加速预筛选
        deg_threshold = radius_km / 111.0 * 1.5  # 留余量

        # 按区分组
        region_pois = {}
        for poi in pois:
            coord = parse_coord(poi.get("坐标", ""))
            region = poi.get("区", "")
            if coord and region:
                region_pois.setdefault(region, []).append(coord)

        coverage = {}
        for region, coords_list in region_pois.items():
            if not coords_list:
                coverage[region] = 0.0
                continue

            poi_arr = np.array(coords_list)  # shape (N, 2)
            lats, lons = poi_arr[:, 0], poi_arr[:, 1]
            min_lat, max_lat = lats.min() - 0.02, lats.max() + 0.02
            min_lon, max_lon = lons.min() - 0.02, lons.max() + 0.02

            grid_lats = np.arange(min_lat, max_lat, grid_resolution)
            grid_lons = np.arange(min_lon, max_lon, grid_resolution)
            # 网格坐标矩阵 shape (M, 2)
            glat_grid, glon_grid = np.meshgrid(grid_lats, grid_lons, indexing='ij')
            grid_pts = np.column_stack([glat_grid.ravel(), glon_grid.ravel()])
            total_points = len(grid_pts)

            if total_points == 0:
                coverage[region] = 0.0
                continue

            # 向量化: 用度数近似快速判断覆盖
            # 对每个网格点, 检查是否有POI在 deg_threshold 内
            covered = 0
            # 分批处理避免内存爆炸 (每批1000个网格点)
            batch_size = 1000
            for i in range(0, total_points, batch_size):
                batch = grid_pts[i:i + batch_size]  # (B, 2)
                # 计算与所有POI的度数差 — 广播 (B, 1, 2) - (1, N, 2)
                diff = batch[:, np.newaxis, :] - poi_arr[np.newaxis, :, :]  # (B, N, 2)
                # 近似距离(度)
                approx_dist = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)  # (B, N)
                # 每个网格点的最小距离
                min_dist = approx_dist.min(axis=1)  # (B,)
                covered += int((min_dist <= deg_threshold).sum())

            coverage[region] = round(covered / max(total_points, 1), 4)

        return coverage

    def compute_accessibility(
        self,
        pois: List[Dict[str, Any]],
        population_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """
        计算各区域可达性指标（简化版：基于区域内POI间平均距离的倒数）

        Args:
            pois: POI列表
            population_points: 人口分布点（可选）

        Returns:
            {区名: 可达性指数}
        """
        region_coords = {}
        for poi in pois:
            coord = parse_coord(poi.get("坐标", ""))
            region = poi.get("区", "")
            if coord and region:
                region_coords.setdefault(region, []).append(coord)

        accessibility = {}
        for region, coords in region_coords.items():
            if len(coords) < 2:
                accessibility[region] = 0.5  # 单点默认中等可达性
                continue

            # 计算区域内POI间平均距离
            dists = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dists.append(haversine(coords[i][0], coords[i][1],
                                           coords[j][0], coords[j][1]))
            avg_dist = np.mean(dists) if dists else 10.0
            # 可达性 = 1 / (1 + avg_dist)，距离越小可达性越高
            accessibility[region] = round(1.0 / (1.0 + avg_dist), 4)

        return accessibility

    @staticmethod
    def get_region_areas() -> Dict[str, float]:
        """
        获取西安市各区面积 (km²)（基于公开数据）
        """
        return {
            "碑林区": 23.37,
            "雁塔区": 152.0,
            "莲湖区": 38.32,
            "新城区": 29.98,
            "未央区": 262.14,
            "灞桥区": 332.0,
            "长安区": 1583.0,
            "临潼区": 915.0,
            "阎良区": 244.0,
            "高陵区": 294.0,
            "鄠邑区": 1282.0,
            "蓝田县": 1969.0,
            "周至县": 2956.0,
            "曲江新区": 51.83,
            "高新区": 307.0,
            "经开区": 86.0,
            "浐灞生态区": 129.0,
            "西咸新区": 882.0,
        }
