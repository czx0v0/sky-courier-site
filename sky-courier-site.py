import streamlit as st
import pandas as pd
import numpy as np
import requests
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import math
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from geopy.distance import geodesic
import time
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# 页面配置
st.set_page_config(layout="wide", page_title="无人机机场选址智能分析系统")
st.title("🥡 无人机机场选址智能分析系统")

# 初始化session状态
if "selected_point" not in st.session_state:
    st.session_state.selected_point = None
if "poi_data" not in st.session_state:
    st.session_state.poi_data = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "pareto_candidates" not in st.session_state:
    st.session_state.pareto_candidates = None

# 常量
# API常量配置
MIN_RADIUS_KM = 1.0  # 最小允许半径

# POI类型配置
# 每日api限额100次，每次花费和种类相同的次数，节省限额所以大部分先注释掉
POI_TYPES = {
    "住宅区": "120000",  # 居住区
    #"商务楼宇": "120200",  # 商务住宅
    "购物中心": "060100",  # 购物相关
    "餐饮服务": "050000",  # 餐饮
    #"学校": "141200",  # 教育
    #"交通枢纽": "150000",  # 交通设施
    #"医院": "090000",  # 医疗
    "写字楼": "120201"  # 写字楼
}
# 类型距离映射配置
POI_DISTANCE_RULES = {
    # 取货点类型 (500km覆盖)
    "餐饮服务": 500,  # 餐饮服务
    "购物中心": 500,  # 购物中心
    # 送货点类型 (2.5km覆盖)
    "住宅区": 4000,  # 住宅区
    "写字楼": 4000,  # 写字楼
    # 其他默认
    "default": 2500
}
# 类型权重配置
POI_WEIGHTS = {
    "餐饮服务": 1.3,  # 餐饮高权重
    "购物中心": 1.5,
    "住宅区": 1.0,
    "写字楼": 1.0,
    "default": 0.8
}
# 点类型配置
POI_TYPE_ICONS = {
    "推荐点": "flag",
    "中心点": "star",
    #"周边点": "cloud"
}


def get_circle_boundary(lat, lng, radius_km=15, points=36):
    """
    通过数学计算生成圆形边界
    :param lat: 中心点纬度
    :param lng: 中心点经度
    :param radius_km: 半径(公里)
    :param points: 边界点数
    :return: 边界坐标列表
    """
    # 动态调整
    dynamic_points = max(points, int(radius_km * 10))

    boundary = []
    for i in range(dynamic_points):
        angle = math.pi * 2 * i / dynamic_points
        dx = (radius_km * math.cos(angle)) / (111.32 * math.cos(math.radians(lat)))
        dy = (radius_km * math.sin(angle)) / 111.32
        boundary.append([lat + dy, lng + dx])
    return boundary

def safe_normalize(data):
    """防御性标准化（处理全零值）"""
    data = np.asarray(data)
    if np.all(data == 0) or len(data) == 0:
        return np.zeros_like(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
def generate_candidate_points(poi_df, n_clusters=10):
    """通过K-Means聚类生成候选点位"""
    coords = poi_df[['lng', 'lat']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(coords)
    # 返回聚类中心作为候选点
    return kmeans.cluster_centers_

def calculate_kde_scores(poi_df, candidate_points):
    """计算每个候选点的核密度得分"""
    # 提取POI坐标和权重（商家权重=1.5，居民区=1.0）
    poi_coords = poi_df[['lng', 'lat']].values.T  # (2, N)
    weights = poi_df['type'].map({'餐饮服务': 1.3,"购物中心":1.5, '住宅区': 1.0, '写字楼': 1.0}).fillna(1.0).values

    # 计算带权重的KDE
    kde = gaussian_kde(poi_coords, weights=weights)

    # 评估候选点密度
    scores = kde.evaluate(candidate_points.T)  # candidate_points形状为(N, 2)
    return pd.DataFrame({'lng': candidate_points[:, 0], 'lat': candidate_points[:, 1], 'kde_score': scores})


def calculate_geo_distance_matrix(candidates, pois):
    """地理距离矩阵（单位：米）"""
    # 转换为 (lat, lng) 元组列表
    candidates_points = [(row['lat'], row['lng']) for _, row in candidates.iterrows()]
    pois_points = [(row['lat'], row['lng']) for _, row in pois.iterrows()]

    # 预分配矩阵
    distance_matrix = np.zeros((len(candidates_points), len(pois_points)))

    # 双重循环计算真实地理距离
    for i, c_point in enumerate(candidates_points):
        for j, p_point in enumerate(pois_points):
            distance_matrix[i][j] = geodesic(c_point, p_point).meters

    return distance_matrix

def optimize_pareto_front(candidates, poi_df, top_n=3):
    """动态距离阈值+加权覆盖的Pareto优化"""
    # 预处理
    candidates = candidates.dropna(subset=['lng', 'lat']).copy()
    poi_df = poi_df.dropna(subset=['lng', 'lat']).copy()

    # 获取每个POI的距离规则
    poi_types = poi_df['type'].apply(lambda x: x if x in POI_DISTANCE_RULES else 'default')
    poi_distances = poi_types.map(POI_DISTANCE_RULES).values
    # st.write(poi_distances)
    poi_weights = poi_types.map(POI_WEIGHTS).values
    # st.write(poi_weights)

    # 计算距离矩阵
    # candidate_coords = candidates[['lng', 'lat']].values
    # poi_coords = poi_df[['lng', 'lat']].values
    #distance_matrix = cdist(candidate_coords, poi_coords)  # 单位：米
    distance_matrix = calculate_geo_distance_matrix(candidates, poi_df)
    # st.write(distance_matrix)
    # 获取每个候选点最近的5个POI索引
    nearest_indices = np.argpartition(distance_matrix, 5, axis=1)[:, :5]

    # 动态覆盖计算
    # 生成覆盖掩码（考虑类型距离规则）
    coverage_mask = distance_matrix < poi_distances
    # st.write(coverage_mask)

    # 计算加权覆盖度
    weighted_coverage = (coverage_mask * poi_weights).sum(axis=1)

    # 有效距离计算
    # 仅统计在覆盖范围内的距离
    valid_distances = np.where(coverage_mask, distance_matrix, np.nan)
    avg_distance = np.nanmean(valid_distances, axis=1)
    avg_distance = np.nan_to_num(avg_distance, nan=5000)  # 无覆盖时设为最大

    # 标准化
    norm_coverage = safe_normalize(weighted_coverage)
    norm_distance = 1 - safe_normalize(avg_distance)  # 距离越小得分越高

    # 非支配排序
    objectives = np.column_stack([-norm_coverage, avg_distance])
    fronts = NonDominatedSorting().do(objectives, n_stop_if_ranked=top_n)

    # 结果选择
    selected_indices = []
    for front in fronts:
        remaining = top_n - len(selected_indices)
        selected_indices.extend(front[:remaining])
        if remaining <= 0:
            break

    # 结果格式化
    result_df = candidates.iloc[selected_indices].copy()
    result_df['weighted_coverage'] = weighted_coverage[selected_indices]
    result_df['avg_distance'] = avg_distance[selected_indices]
    result_df['score'] = 0.6 * norm_coverage[selected_indices] + 0.4 * norm_distance[selected_indices]

    # return result_df.sort_values('score', ascending=False).head(top_n)
    # 不能直接返回DataFrame!
    return [
        {
            "lat": row['lat'],
            "lng": row['lng'],
            "kde_score":row['kde_score'],
            "weighted_coverage":row['weighted_coverage'],
            "avg_distance":row['avg_distance'],
            "score":row['score'],
            "nearest_pois": [
                {k:v for k,v in p.items() if k in ['name','type','lat','lng','distance']}
                for p in poi_df.iloc[nearest_indices[i]].to_dict('records')
            ]
        }
        for i, row in result_df.iterrows()
    ]
# 1. 地图交互模块
with st.expander("🗺️ 第一步：选择中心点", expanded=True):
    col_map, col_info = st.columns([3, 1])

    with col_map:
        # 创建地图底图
        m = folium.Map(location=[22.5411, 114.0588], zoom_start=12)
        folium.TileLayer(
            tiles='https://webst01.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}',
            attr='高德地图',
            name='高德地图'
        ).add_to(m)

        # 添加绘图工具（点击、圆圈、多边形）
        Draw(
            export=True,
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": True,
                "marker": True,
                "circlemarker": False,
                "rectangle": True
            },
            edit_options={"edit": True}
        ).add_to(m)

        # 显示地图并获取点击事件
        map_output = st_folium(
            m,
            width=800,
            height=600,
            returned_objects=["all_drawings", "last_active_drawing", "last_clicked"]
        )

        # 保存选择的点
        if map_output["last_active_drawing"]:
            selection = map_output["last_active_drawing"]
            st.session_state.draw_type = selection["geometry"]["type"]

            if selection["geometry"]["type"] == "Point":
                st.session_state.selected_point = {
                    "lat": selection["geometry"]["coordinates"][1],
                    "lng": selection["geometry"]["coordinates"][0]
                }
                st.session_state.analysis_area = {
                    "center": [selection["geometry"]["coordinates"][1],
                               selection["geometry"]["coordinates"][0]],
                    "boundary": None,
                    "radius": 15
                }
            elif selection["geometry"]["type"] in ["Circle", "Polygon"]:
                # 计算几何中心点
                if selection["geometry"]["type"] == "Circle":
                    coords = selection["geometry"]["coordinates"]
                    # 转换为km
                    raw_radius_m = selection["radius"]
                    radius_km = max(raw_radius_m / 1000, MIN_RADIUS_KM)
                    st.session_state.analysis_area = {
                        "center": [coords[1], coords[0]],
                        "boundary": None,
                        "radius": radius_km
                    }
                    # 显示
                    if raw_radius_m / 1000 < MIN_RADIUS_KM:
                        st.warning(f"已自动将半径调整为最小值 {MIN_RADIUS_KM} 公里")
                else:  # Polygon
                    coords = np.array(selection["geometry"]["coordinates"][0])
                    center = coords.mean(axis=0)
                    st.session_state.analysis_area = {
                        "center": [center[1], center[0]],
                        "boundary": selection["geometry"]["coordinates"][0],
                        "radius": 15
                    }

                st.session_state.selected_point = {
                    "lat": st.session_state.analysis_area["center"][0],
                    "lng": st.session_state.analysis_area["center"][1]
                }


    with col_info:
        st.subheader("选择信息")
        if st.session_state.selected_point:
            st.success("已选择中心点")
            st.write(f"纬度: {st.session_state.selected_point['lat']:.6f}")
            st.write(f"经度: {st.session_state.selected_point['lng']:.6f}")
            # 动态获取半径
            current_radius = st.session_state.analysis_area.get("radius", 15.0)
            # 类型检查与转换
            if isinstance(current_radius, (list, tuple)):
                # 如果存储的是多边形顶点，取第一个点的半径
                current_radius = float(current_radius[0])
            else:
                current_radius = float(current_radius)
            # 半径调整滑块
            new_radius = st.slider(
                "调整覆盖的半径 (km)",
                min_value=1.0,
                max_value=15.0,
                value=current_radius,
                step=0.5,
                key="radius_selector"
            )
            # 半径调整
            if new_radius != current_radius:
                # 转换为float
                st.session_state.analysis_area["radius"] = float(new_radius)
                st.rerun()

            st.write(f"当前半径: **{new_radius} km**")
            st.write(f"覆盖面积: **{math.pi * new_radius ** 2:.1f} km²**")

            boundary = get_circle_boundary(
                st.session_state.selected_point['lat'],
                st.session_state.selected_point['lng'],
                radius_km=15
            )

            # 保存到session
            st.session_state.analysis_area = {
                "center": [st.session_state.selected_point['lat'], st.session_state.selected_point['lng']],
                "boundary": boundary
            }

        else:
            st.warning("请在地图上点击选择中心点")

def get_coordinates(poi_df):
    """统一处理不同格式的位置数据"""
    coordinates = []
    for _, row in poi_df.iterrows():
        try:
            # 情况1：有独立的lat和lng列
            if 'lat' in row and 'lng' in row:
                lat, lng = row['lat'], row['lng']
            # 情况2：location是"lng,lat"字符串
            elif 'location' in row and isinstance(row['location'], str):
                lng, lat = map(float, row['location'].split(','))
            # 情况3：location是字典
            elif 'location' in row and isinstance(row['location'], dict):
                lng, lat = row['location']['lng'], row['location']['lat']
            else:
                continue
            coordinates.append([lat, lng])  # 注意Folium使用(lat,lng)顺序
        except (KeyError, AttributeError) as e:
            st.warning(f"跳过无效位置数据: {row}")
    return coordinates

# 缓存装饰器减少API调用
@st.cache_data(ttl=3600, show_spinner="正在获取POI数据...")
def get_combined_poi(api_key, location, radius, types=None):
    """获取多种POI类型的组合数据"""
    if types is None:
        types = list(POI_TYPES.values())
    all_pois = []
    for type_code in types:
        page = 1
        while True:
            url = f"https://restapi.amap.com/v3/place/around?key={api_key}" \
                  f"&location={location}&radius={radius}&types={type_code}&offset=20&page={page}" # offset<=25
            try:
                response = requests.get(url, timeout=50)
                data = response.json()
                if data['status'] != '1':
                    break
                for poi in data["pois"]:
                    poi["poi_type"] = [k for k, v in POI_TYPES.items() if v == type_code][0]
                    all_pois.append(poi)
                page += 1

                time.sleep(0.8)
                if page > 30:  # 最多页数
                    break
            except Exception as e:
                st.warning(f"获取{type_code}类型POI失败: {str(e)}")
        st.write(f'{type_code}获取完成')
    return all_pois

# 在点击获取POI按钮时调用
if st.session_state.selected_point and st.button("获取周边POI数据"):
    location = f"{st.session_state.selected_point['lng']},{st.session_state.selected_point['lat']}"
    radius = float(st.session_state.analysis_area.get("radius", 15.0)) * 1000  # 转换为米

    with st.spinner(f"正在获取半径{radius / 1000}km内的多类型POI数据..."):
        all_pois = get_combined_poi(
            st.secrets["AMAP_KEY"],
            location,
            radius,
            types=list(POI_TYPES.values())  # 获取所有类型
        )

    # 处理数据
    processed_data = []
    for poi in all_pois:
        try:
            lng, lat = map(float, poi["location"].split(","))
            if not (-180 <= lng <= 180 and -90 <= lat <= 90):
                raise ValueError("坐标超出合理范围")
            address = str(poi["address"])  # 强制转换为字符串
            # 如果address是列表，转换为逗号分隔字符串
            if isinstance(poi["address"], list):
                address = ", ".join(poi["address"])
            processed_data.append({
                "name": str(poi.get("name", "")),
                "type": str(poi.get("poi_type", "其他")),
                "address": address,
                "lat": lat,
                "lng": lng,
                "distance": float(poi.get("distance", 0))
            })
        except Exception as e:
            st.error(f"解析POI数据失败：{str(e)}")
            continue
    # 创建DataFrame并清洗
    df = pd.DataFrame(processed_data)
    # 二次清洗
    df = df[
        (df['lat'].notnull()) &
        (df['lng'].notnull())
        ]
    df = df.astype({
        "name": "string",
        "type": "category",
        "address": "string",
        "lat": "float32",
        "lng": "float32",
        "distance": "float32"
    })
    st.session_state.poi_data = df

# 显示POI数据
if st.session_state.poi_data is not None:
    with st.expander("📊 POI数据预览"):
        st.dataframe(st.session_state.poi_data.head(20))
        # 可视化POI分布
        st.subheader("POI分布热力图")
        # 创建可视化地图
        viz_map = folium.Map(
            location=[st.session_state.selected_point['lat'], st.session_state.selected_point['lng']],
            zoom_start=12
        )
        heat_data = get_coordinates(st.session_state.poi_data)
        if heat_data:
            HeatMap(heat_data, radius=15).add_to(viz_map)

        # 添加分析范围
        folium.GeoJson(
            {
                "type": "Polygon",
                "coordinates": [st.session_state.analysis_area["boundary"]]
            },
            style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 2, "fillOpacity": 0.1}
        ).add_to(viz_map)

        # 添加中心点
        folium.Marker(
            [st.session_state.selected_point['lat'], st.session_state.selected_point['lng']],
            popup="中心点",
            icon=folium.Icon(color="green")
        ).add_to(viz_map)

        st_folium(viz_map, width=800, height=500)
if st.session_state.poi_data is not None:
    # 数据完整性检查
    st.subheader("数据健康检查")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("总POI数量", len(st.session_state.poi_data))
        st.metric("有效坐标数",
                  len(st.session_state.poi_data.dropna(subset=['lat', 'lng'])))

    with col2:
        invalid_coords = st.session_state.poi_data[
            (~st.session_state.poi_data['lat'].between(-90, 90)) |
            (~st.session_state.poi_data['lng'].between(-180, 180))
            ]
        st.metric("无效坐标数", len(invalid_coords))

## 后端算法
if st.session_state.poi_data is not None and st.button("开始优化选址"):
    poi_df = st.session_state.poi_data
    # 进度管理
    with st.status("🚀 优化进程", state="running", expanded=True) as status:
        # 阶段1: 生成候选点

        st.write("🧮 1/3 使用K-Means聚类识别高密度区域...生成候选点...")
        candidate_points = generate_candidate_points(poi_df)  # 使用清洗后的数据
        time.sleep(0.5)

        # 阶段2: 核密度估计
        st.write("🗺️ 2/3 计算带权重的空间密度分布...核密度估计...")
        kde_scores = calculate_kde_scores(poi_df, candidate_points)
        time.sleep(0.5)

        # 阶段3: 多目标优化
        st.write("💻 3/3 Pareto前沿筛选最优解...Pareto优化...")
        pareto_candidates = optimize_pareto_front(kde_scores, poi_df,top_n = 5)
        time.sleep(0.5)

        status.update(label="✅ 优化完成", state="complete")
    # 结果显示
    st.subheader("📊 数据质量报告")
    st.metric("有效候选点", len(pareto_candidates))
    st.write(pareto_candidates)
    st.session_state.pareto_candidates = pareto_candidates

#  DeepSeek API分析
if st.session_state.poi_data is not None and st.button("进行智能分析"):
    with st.spinner("AI分析中..."):
        try:
            # 数据准备
            # st.write(st.session_state.pareto_candidates)
            points = [{"lat": p["lat"], "lng": p["lng"],"score":p['score'],
                                "kde_score":p['kde_score'],"avg_distance":p['avg_distance'],
                                "weighted_coverage":p['weighted_coverage'],"nearest_pois":p['nearest_pois']} for p in st.session_state.pareto_candidates]
            # 构造专业prompt模板
            prompt = f"""
                        ## 角色定义：
                        你是一名城市规划专家，需评估外卖无人机机场配送站选址方案，提供外卖无人机机场选址AI分析报告。
                        ## 最优机场建设候选点数据：
                        - 推荐点详细信息:{st.session_state.pareto_candidates}
                            - "lat": 纬度
                            - "lng": 经度
                            - "score": 总得分
                            - "kde_score"：核密度评分
                            - "avg_distance": 周边poi平均距离
                            - "weighted_coverage": 加权poi覆盖度
                            - "nearest_pois": 周边主要poi
                        - （算法: K-Means聚类+核密度估计+Pareto优化）
                        ## 深度分析维度:  
                        1. 商业潜力对比（基于周边餐饮/购物中心密度）  
                        2. 交通可达性分析（道路网络+峰值时段）  
                        3. 空域合规性评估（限飞区搜索推测）
                        4. 商业成本和租金等（区域租金搜索推测）  
                        ## 输出要求: 
                        请结合以上维度用对比表格呈现前{len(points)} 名候选点优劣，最后给出综合推荐。  
                       """
            # 调用DeepSeek API
            deepseek_key = st.secrets["DEEPSEEK_KEY"]
            headers = {
                "Authorization": f"Bearer {deepseek_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=(30,50)
            )
            # 结果可视化
            if response.status_code == 200:
                analysis = response.json()["choices"][0]["message"]["content"]
                # 存储
                st.session_state.analysis_result = analysis
        except Exception as e:
            st.error(f"分析失败: {str(e)}")
    try:
        st.header("📌 智能分析结果")
        # 数据准备
        points = st.session_state.pareto_candidates
        # 地图生成
        map_center = [
            st.session_state.selected_point['lat'],
            st.session_state.selected_point['lng']
        ]
        m = folium.Map(location=map_center, zoom_start=12)

        # 中心点
        folium.Marker(
            map_center,
            tooltip="<b>原始中心点<b>",
            icon=folium.Icon(icon=POI_TYPE_ICONS['中心点'],
                             color="red",
                             )
        ).add_to(m)

        # 候选点
        for idx, point in enumerate(points, 1):
            folium.Marker(
                [point["lat"], point["lng"]],
                tooltip=f"""
                <b>推荐点#{idx},<b><br>
                "lat": {point["lat"]:.8f},<br>
                "lng": {point["lng"]:.8f},<br>
                "score":{point['score']:.6f},<br>
                "kde_score":{point['kde_score']:.6f},<br>
                "avg_distance":{point['avg_distance']:.6f},<br>
                "weighted_coverage":{point['weighted_coverage']:.6f},
                """,
                icon = folium.Icon(
                    icon=POI_TYPE_ICONS['推荐点'],
                    color="green",
                    angle=45
                )
            ).add_to(m)

            # 绘制POI连线
            for poi in point['nearest_pois']:
                folium.Marker(
                    [poi['lat'], poi['lng']],
                    popup=poi["name"], # 点击查看
                    icon=folium.DivIcon(
                        icon_size=(15, 15),
                        icon_anchor=(10, 10),
                        html='<div style="background:rgba(150,150,150,0.7); width:15px; height:15px; border-radius:50%;"></div>'
                    ),
                ).add_to(m)
                folium.PolyLine(
                    locations=[[point['lat'], point['lng']], [poi['lat'], poi['lng']]],
                    color="black",
                    weight = 2,
                    dash_array="5,3"
                ).add_to(m)

        st.markdown(st.session_state.analysis_result)
        st_folium(
            m,
            width=800,
            height=600,
            key="optimized_points_map",
            returned_objects=[]
        )
        # 添加下载按钮
        st.download_button(
            label="下载分析报告",
            data=st.session_state.analysis_result,
            file_name="选址分析报告.md",
            mime="text/markdown"
        )
    except Exception as e:
        st.error(f"可视化失败: {str(e)}")
        st.stop()

# 隐藏streamlit默认菜单和页脚
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
