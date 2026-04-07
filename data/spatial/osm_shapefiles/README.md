# OSM Shapefiles for Xi'an

## 说明

由于文件过大(总计约 250 MB),OSM shapefile 数据未包含在此仓库中。

## 如何获取数据

### 方法 1: 从 Geofabrik 下载

1. 访问 Geofabrik 下载页面: https://download.geofabrik.de/asia/china.html
2. 下载陕西省数据: `shaanxi-latest-free.shp.zip`
3. 解压到此目录

### 方法 2: 使用 OSM 数据提取工具

使用 `osmium` 或 `ogr2ogr` 工具从 OSM PBF 文件提取西安市数据:

```bash
# 下载陕西省 PBF 文件
wget https://download.geofabrik.de/asia/china/shaanxi-latest.osm.pbf

# 提取西安市边界内的数据
osmium extract -b 107.40,33.42,109.49,34.45 shaanxi-latest.osm.pbf -o xian.osm.pbf

# 转换为 shapefile
ogr2ogr -f "ESRI Shapefile" osm_shapefiles xian.osm.pbf
```

### 方法 3: 联系作者

如需完整的预处理数据,请联系作者。

## 所需文件列表

完整的 shapefile 数据应包含以下文件:

- `gis_osm_buildings_a_free_1.*` (建筑物, ~37 MB)
- `gis_osm_landuse_a_free_1.*` (土地利用, ~51 MB)
- `gis_osm_roads_free_1.*` (道路网络, ~57 MB)
- `gis_osm_pois_free_1.*` (兴趣点)
- `gis_osm_water_a_free_1.*` (水体)
- `gis_osm_waterways_free_1.*` (水系)
- `gis_osm_natural_a_free_1.*` (自然要素)
- `gis_osm_places_free_1.*` (地名)
- `gis_osm_transport_a_free_1.*` (交通设施)
- `gis_osm_railways_free_1.*` (铁路)
- `gis_osm_traffic_a_free_1.*` (交通信号)
- `gis_osm_pofw_a_free_1.*` (宗教场所)

每个图层包含 5 个文件: `.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`

## 数据说明

- **数据源**: OpenStreetMap (OSM)
- **覆盖范围**: 陕西省
- **坐标系**: WGS84 (EPSG:4326)
- **更新时间**: 2025-12-15
- **许可证**: ODbL (Open Database License)

## 使用示例

```python
import geopandas as gpd

# 读取建筑物数据
buildings = gpd.read_file('data/spatial/osm_shapefiles/gis_osm_buildings_a_free_1.shp')

# 读取道路数据
roads = gpd.read_file('data/spatial/osm_shapefiles/gis_osm_roads_free_1.shp')

# 筛选西安市范围
xian_bounds = gpd.read_file('data/spatial/xian_city.geojson')
buildings_xian = gpd.sjoin(buildings, xian_bounds, predicate='within')
```

## 注意事项

1. 下载的数据可能需要根据西安市边界进行裁剪
2. 确保所有 shapefile 组件文件(.shp, .shx, .dbf, .prj, .cpg)都在同一目录
3. 数据量较大,建议使用 GeoPandas 或 QGIS 进行处理
