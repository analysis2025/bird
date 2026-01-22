import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. 设置页面配置
st.set_page_config(page_title="专业鸟类识别专家", page_icon="🦅")

st.title("🦅 专业鸟类识别专家")
st.write("上传鸟类照片，AI 将精准识别具体品种（支持500+种鸟类）")

# 2. 加载模型 
@st.cache_resource
def load_model():
    # 核心修改：这里换成了专门针对鸟类训练的 expert model
    # 模型名称：nateraw/vit-base-birds
    # 这个模型能识别 555 种鸟类，准确率远超通用模型
    try:
        # 第一次运行会下载约 340MB 的模型文件
        classifier = pipeline("image-classification", model="nateraw/vit-base-birds")
        return classifier
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

with st.spinner('正在召唤鸟类专家模型 (首次加载约需1分钟)...'):
    classifier = load_model()

# 3. 创建文件上传组件
uploaded_file = st.file_uploader("请选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的图片', use_column_width=True)

    # 4. 开始识别
    if st.button('开始鉴定'):
        if classifier:
            with st.spinner('专家正在仔细观察羽毛和特征...'):
                try:
                    # 模型推理
                    results = classifier(image)

                    # 5. 展示结果
                    st.success("鉴定完成！")
                    
                    # 取出置信度最高的结果
                    top_result = results[0]
                    english_name = top_result['label']
                    score = top_result['score']
                    
                    # 格式化一下名字（把下划线换成空格，首字母大写）
                    formatted_name = english_name.replace("_", " ").title()

                    st.subheader("鉴定结论：")
                    st.metric(label="鸟类英文学名", value=formatted_name, delta=f"置信度: {score:.2%}")
                    
                    st.info(f"💡 提示: 您可以将 '{formatted_name}' 复制到搜索引擎查看中文详情。")

                    # 展示概率分布
                    st.write("---")
                    st.write("其他可能：")
                    for res in results[1:4]: # 只显示前3个备选
                        name = res['label'].replace("_", " ").title()
                        st.write(f"**{name}**: {res['score']:.2%}")
                        st.progress(res['score'])
                        
                except Exception as e:
                    st.error(f"识别过程中出现错误: {e}")
        else:
            st.error("模型未加载，无法进行识别。")

# 侧边栏
st.sidebar.title("关于模型")
st.sidebar.info(
    "当前使用的模型: \n"
    "**nateraw/vit-base-birds**\n\n"
    "该模型在 CUB-200-2011 数据集上训练，"
    "覆盖全球 555 种常见鸟类。"
)
