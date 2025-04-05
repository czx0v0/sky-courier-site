import streamlit as st
import pandas as pd
import numpy as np
import requests
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import json
import math
from folium.plugins import HeatMap
import re

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
# 修改地图交互处理部分

# API常量配置
MIN_RADIUS_KM = 1.0  # 最小允许半径

# POI类型配置
# 每日api限额100次，每次花费和种类相同的次数，节省限额所以大部分先注释掉
POI_TYPES = {
    #"住宅区": "120000",  # 居住区
    "商务楼宇": "120200",  # 商务住宅
    #"购物中心": "060100",  # 购物相关
    #"餐饮服务": "050000",  # 餐饮
    #"学校": "141200",  # 教育
    #"交通枢纽": "150000",  # 交通设施
    #"医院": "090000",  # 医疗
    #"写字楼": "120201"  # 写字楼
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
        url = f"https://restapi.amap.com/v3/place/around?key={api_key}" \
              f"&location={location}&radius={radius}&types={type_code}&offset=120"
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            if data['status'] != '1':
                break
            for poi in data["pois"]:
                poi["poi_type"] = [k for k, v in POI_TYPES.items() if v == type_code][0]
                all_pois.append(poi)
        except Exception as e:
            st.warning(f"获取{type_code}类型POI失败: {str(e)}")

    return all_pois


# 在后端分析部分添加评分逻辑
def calculate_scores(poi_df, center_point):
    """计算选址评分"""
    scores = {
        "coverage": 0,  # 覆盖度
        "density": 0,  # POI密度
        "diversity": 0  # 类型多样性
    }

    # 1. 覆盖度评分 (5km范围内POI数量占比)
    nearby_pois = poi_df[poi_df["distance"] <= 5000]
    scores["coverage"] = min(len(nearby_pois) / 50, 1.0) * 100  # 标准化到0-100

    # 2. 密度评分 (按单位面积POI数量)
    area = math.pi * (5 ** 2)  # 5km半径圆面积
    scores["density"] = min(len(nearby_pois) / area * 100, 100)

    # 3. 多样性评分 (类型熵)
    type_counts = nearby_pois["type"].value_counts()
    proportions = type_counts / type_counts.sum()
    entropy = -sum(proportions * np.log(proportions))
    scores["diversity"] = min(entropy * 20, 100)  # 标准化

    # 综合评分
    weights = {"coverage": 0.4, "density": 0.3, "diversity": 0.3}
    total_score = sum(scores[k] * w for k, w in weights.items())

    return {
        **scores,
        "total_score": total_score,
        "poi_count": len(nearby_pois),
        "type_distribution": type_counts.to_dict()
    }

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
        lng, lat = poi["location"].split(",")
        processed_data.append({
            "name": poi["name"],
            "type": poi["poi_type"],
            "address": poi["address"],
            "lat": float(lat),
            "lng": float(lng),
            "distance": float(poi["distance"])
        })

    st.session_state.poi_data = pd.DataFrame(processed_data)


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

        #heat_data = [[row[1], row[0]] for row in st.session_state.poi_data["location"].str.split(",").tolist()]
        heat_data = get_coordinates(st.session_state.poi_data)
        if heat_data:
            #HeatMap(heat_data).add_to(m)
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

#  DeepSeek API分析
if st.session_state.poi_data is not None and st.button("进行智能分析"):
    with st.spinner("正在调用DeepSeek API进行分析..."):
        try:
            # 准备分析数据（示例）
            poi_stats = {
                "total_pois": len(st.session_state.poi_data),
                "avg_distance": st.session_state.poi_data["distance"].mean(),
                "type_distribution": st.session_state.poi_data["type"].value_counts().to_dict()
            }

            # 构造prompt
            prompt = f"""
            你是一个专业的外卖无人机机场选址评估AI顾问，请根据以下数据提供分析报告：

            - 中心点坐标: ({st.session_state.selected_point['lat']:.6f}, {st.session_state.selected_point['lng']:.6f})
            - 分析半径: 15公里
            - POI总数: {poi_stats['total_pois']}
            - 平均距离: {poi_stats['avg_distance']:.2f}米
            - 类型分布: {json.dumps(poi_stats['type_distribution'], ensure_ascii=False)}

            请提供：
            1. 3个最佳选址坐标（纬度,经度）及理由。
            对每个选址，请从以下维度给出1-5星评分：
                1. **商业潜力**（餐饮/写字楼密度）
                2. **运营成本**（根据区域推测租金水平）
                3. **抗风险能力**（天气适应性和政策支持）
                4. 和现有的无人机机场的关系
                5. 电力水平
                6. 运营时间段
            格式要求使用Markdown满足程度
            注意坐标要匹配"(\d+\.\d+),\s*(\d+\.\d+)"正则表达式格式。
            """
            # 在调用DeepSeek API前添加评分计算
            if st.session_state.poi_data is not None:
                score_result = calculate_scores(
                    st.session_state.poi_data,
                    st.session_state.selected_point
                )

                # 将评分结果加入prompt
                prompt += f"\n\n当前区域评分:\n"
                prompt += f"- 覆盖度: {score_result['coverage']:.1f}/100\n"
                prompt += f"- 密度: {score_result['density']:.1f}/100\n"
                prompt += f"- 多样性: {score_result['diversity']:.1f}/100\n"
                prompt += f"- 综合评分: {score_result['total_score']:.1f}/100\n"
                prompt += f"- POI类型分布: {score_result['type_distribution']}\n"
            # 调用DeepSeek API（替换为你的实际调用方式）
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

            if response.status_code == 200:
                result = response.json()
                analysis_text = result["choices"][0]["message"]["content"]

                # 提取坐标并计算评分
                coord_matches = re.findall(r"(\d+\.\d+),\s*(\d+\.\d+)", analysis_text)
                st.write("提取到的坐标匹配:", coord_matches)  # 调试用
                recommended_points = [{"lat": float(lat), "lng": float(lng)} for lat,lng  in coord_matches[:3]]

                # 保存结果
                st.session_state.analysis_result = {
                    "text": analysis_text,
                    "recommendations": recommended_points,
                    "score": score_result
                }

                st.success("分析完成！")
            else:
                st.error(f"DeepSeek API错误: {response.text}")
        except Exception as e:
            st.error(f"分析失败: {str(e)}")

# 4. 显示分析结果
if st.session_state.analysis_result:
    st.divider()
    st.header("📌 智能分析结果")

    col_result, col_viz = st.columns([2, 3])

    with col_result:
        # 显示Markdown格式的分析报告
        st.markdown(st.session_state.analysis_result["text"])

        # 添加下载按钮
        st.download_button(
            label="下载分析报告",
            data=st.session_state.analysis_result["text"],
            file_name="选址分析报告.md",
            mime="text/markdown"
        )

    with col_viz:
        # 创建结果可视化地图
        result_map = folium.Map(
            location=[st.session_state.selected_point['lat'], st.session_state.selected_point['lng']],
            zoom_start=12
        )

        # 添加分析范围
        folium.GeoJson(
            {
                "type": "Polygon",
                "coordinates": [st.session_state.analysis_area["boundary"]]
            },
            style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 2, "fillOpacity": 0.1}
        ).add_to(result_map)

        # 添加中心点
        folium.Marker(
            [st.session_state.selected_point['lat'], st.session_state.selected_point['lng']],
            popup="原始选择点",
            icon=folium.Icon(color='green', icon='flag')
        ).add_to(result_map)
        # 添加推荐点
        for idx, point in enumerate(st.session_state.analysis_result["recommendations"]):
            st.subheader(f"推荐点位{idx}")
            st.table(point)
            # print(idx,point["lat"], point["lng"])
            folium.CircleMarker(
                location=[point["lat"], point["lng"]],
                popup=folium.Popup(f"推荐点位{idx}"),
                icon=folium.Icon(color='red', icon='star'),
            ).add_to(result_map)
        st_folium(result_map, width=800, height=500)

# 隐藏streamlit默认菜单和页脚
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)