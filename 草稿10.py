import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. 设置页面配置
st.set_page_config(page_title="智能鸟类识别助手", page_icon="🐦")

st.title("🐦 智能鸟类识别助手")
st.write("请上传一张鸟类的照片，我会告诉你它是什么！")


# 2. 加载模型 (使用缓存装饰器，避免每次刷新都重新加载模型)
@st.cache_resource
def load_model():
    # 这里我们使用 Google 的 ViT 模型，它在图像分类上表现非常出色
    # 你也可以换成专门针对鸟类微调过的模型，例如 "nateraw/vit-base-birds"
    try:
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        return classifier
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None


with st.spinner('正在加载 AI 模型，请稍候...'):
    classifier = load_model()

# 3. 创建文件上传组件
uploaded_file = st.file_uploader("选择一张 JPG 或 PNG 图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示用户上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的图片', use_column_width=True)

    # 4. 开始识别
    if st.button('开始识别'):
        if classifier:
            with st.spinner('AI 正在观察这张图片...'):
                # 模型推理
                results = classifier(image)

                # 5. 展示结果
                st.success("识别完成！")
                st.subheader("我认为它是：")

                # 取出置信度最高的结果
                top_result = results[0]
                label = top_result['label']
                score = top_result['score']

                st.metric(label="预测结果", value=label, delta=f"置信度: {score:.2%}")

                # 展示其他可能的结果
                st.write("---")
                st.write("详细概率分布：")
                for res in results:
                    st.progress(res['score'])
                    st.write(f"**{res['label']}**: {res['score']:.2%}")
        else:
            st.error("模型未加载，无法进行识别。")

# 添加侧边栏说明
st.sidebar.title("关于")
st.sidebar.info(
    "这个应用使用 Python 和 Hugging Face Transformers 构建。\n\n"
    "模型: Vision Transformer (ViT)"
)
