import os

# =========================================================
# 🚀 核心修复：配置国内镜像加速
# 这两行代码必须放在最开头，用于解决 "connection error" 或 "model not found"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# =========================================================

import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. 设置页面配置
st.set_page_config(page_title="专业鸟类识别专家", page_icon="🦅")

st.title("🦅 专业鸟类识别专家")
st.markdown("### 🔍 只有国内网络也能用的版本")
st.write("上传鸟类照片，AI 将使用 **nateraw/vit-base-birds** 模型进行精准识别（支持555种鸟类）。")

# 2. 加载模型 
@st.cache_resource
def load_model():
    # 使用专门针对鸟类训练的模型
    try:
        # 第一次运行时，因为配置了 hf-mirror.com，下载速度会快很多
        classifier = pipeline("image-classification", model="nateraw/vit-base-birds")
        return classifier
    except Exception as e:
        # 如果报错，打印详细信息
        return None

# 加载时的提示信息
if 'classifier' not in st.session_state:
    with st.spinner('正在连接镜像站下载模型 (首次运行约需 1-3 分钟，请耐心等待)...'):
        st.session_state.classifier = load_model()

# 3. 创建文件上传组件
uploaded_file = st.file_uploader("请选择一张 JPG 或 PNG 图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示用户上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的图片', use_column_width=True)

    # 4. 开始识别按钮
    if st.button('开始鉴定'):
        classifier = st.session_state.classifier
        
        if classifier:
            with st.spinner('AI 专家正在观察特征...'):
                try:
                    # 模型推理
                    results = classifier(image)

                    # 5. 展示结果
                    st.success("鉴定完成！")
                    
                    # --- 处理最佳结果 ---
                    top_result = results[0]
                    english_name = top_result['label']
                    score = top_result['score']
                    
                    # 格式化名字 (例如 "bald_eagle" -> "Bald Eagle")
                    formatted_name = english_name.replace("_", " ").title()

                    st.subheader("鉴定结论")
                    # 显示大号的结果
                    st.metric(label="鸟类学名 (英文)", value=formatted_name, delta=f"置信度: {score:.2%}")
                    
                    # 💡 提示用户
                    st.info(f"👉 复制 **{formatted_name}** 去百度/谷歌搜索，即可看到中文详细介绍。")

                    # --- 展示概率分布 ---
                    st.write("---")
                    st.write("**其他可能的结果：**")
                    for res in results[1:4]: # 只看第2到第4名
                        name = res['label'].replace("_", " ").title()
                        st.write(f"{name}: {res['score']:.2%}")
                        st.progress(res['score'])
                        
                except Exception as e:
                    st.error(f"识别过程中出现错误: {e}")
        else:
            st.error("❌ 模型加载失败。可能是网络问题导致下载中断，请尝试重启程序。")
            st.warning("提示：请确保你的电脑已连接互联网。")

# 侧边栏说明
st.sidebar.header("关于")
st.sidebar.info(
    "✅ **已启用国内镜像加速**\n\n"
    "模型: nateraw/vit-base-birds\n"
    "能力: 识别 555 种鸟类"
)
