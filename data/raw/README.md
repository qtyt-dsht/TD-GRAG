# POI Data Files

## 说明

由于文件过大,完整的 POI 数据未包含在此仓库中。

## 文件说明

- `poi_info_sample.json` - 示例数据 (前 100 条记录)
- `poi_info_final.json` - 完整 POI 数据 (7.9 MB, 未包含)
- `poi_info_original.json` - 原始 POI 数据 (3.1 MB, 未包含)

## 如何获取完整数据

### 方法 1: 联系作者

如需完整的 POI 数据,请联系作者。

### 方法 2: 自行采集

使用高德地图 API 或百度地图 API 采集西安市文化用地 POI 数据:

```python
import requests

# 示例: 使用高德地图 API
def get_pois(keyword, city='西安'):
    url = 'https://restapi.amap.com/v3/place/text'
    params = {
        'key': 'YOUR_API_KEY',
        'keywords': keyword,
        'city': city,
        'output': 'json'
    }
    response = requests.get(url, params=params)
    return response.json()

# 文化用地关键词
keywords = ['博物馆', '图书馆', '文化馆', '美术馆', '剧院', '历史遗迹']
```

## 数据格式

POI 数据格式示例:

```json
{
  "name": "陕西历史博物馆",
  "type": "博物馆",
  "address": "西安市雁塔区小寨东路91号",
  "location": {
    "lng": 108.964888,
    "lat": 34.224377
  },
  "description": "...",
  "rating": 4.8
}
```

## 数据统计

- **总记录数**: ~2,000 条
- **覆盖范围**: 西安市全域
- **数据类型**: 文化设施、历史遗迹、公共文化空间
- **更新时间**: 2025-12
