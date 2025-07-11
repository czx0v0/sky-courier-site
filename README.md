# sky-courier-site
## 项目运行说明
1. 克隆仓库
```bash
git clone https://github.com/czx0v0/sky-courier-site.git
```
2. 新建conda环境，python == 3.9.21
```bash
 conda create -n sky-courier python=3.9.21
```
3. 激活conda环境，安装所需包，需安装库见`requirement.txt`(可选手动安装) 
```bash
 conda activate sky-courier
 conda install -c conda-forge folium geopy numpy pandas pymoo requests scikit-learn streamlit streamlit-folium
```
4. 新建`.streamlit`文件夹，以及`.streamlit/secrets.toml`文件，在文件中写入API_KEY，其中AMAP_KEY和DEEPSEEK_KEY分别为高德地图和DeepSeek的API_KEY。
```toml
AMAP_KEY = "xxx"
DEEPSEEK_KEY = "xxx"
```
5. 进入目录
```bash
cd sky-courier-site
```
6. 🌟在根目录下，运行
```bash
streamlit run sky-courier-site.py
```
## 补充
- 项目如果出现卡顿可以尝试刷新一下。
- 地图选点时需要先选中左侧的定位工具再点击右侧地图。
- 在生成选址报告时不要缩放和移动现有地图，否则可能会有错误。

