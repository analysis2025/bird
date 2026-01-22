import streamlit as st
import os
import shutil
from transformers import pipeline
from PIL import Image

# ==============================================================================
# 🛠️ 自动下载模块 (使用国内 ModelScope 镜像)
# ==============================================================================
def check_and_download_model():
    """
    检查本地是否存在 bird_model 文件夹。
    如果不存在，自动调用阿里云 ModelScope 进行国内极速下载。
    """
    local_model_path = "./bird_model"
    
    # 1. 检查文件是否存在 (检查关键的 .bin 文件)
    if os.path.exists(local_model_path) and \
       os.path.exists(os.path.join(local_model_path, "pytorch_model.bin")):
        return local_model_path
    
    # 2. 如果不存在，开始下载
    st.info("检测到本地缺少模型文件，正在通过国内镜像自动下载 (约300MB)...")
    
    try:
        from modelscope import snapshot_download
        
        # 创建一个临时进度条
        progress_text = "🚀 正在从阿里云下载模型，速度很快，请稍候..."
        my_bar = st.progress(0, text=progress_text)

        # 下载到临时缓存目录
        # nateraw/vit-base-birds 是模型ID
        cache_dir = snapshot_download('nateraw/vit-base-birds', cache_dir='./temp_download')
        
        my_bar.progress(90, text="下载完成，正在整理文件...")
        
        # 3. 将下载的文件移动到整洁的 ./bird_model 文件夹
        if not os.path.exists(local_model_path):
            os.makedirs(local_model_path)
            
        # 遍历下载目录，把文件移出来
        for file_name in os.listdir(cache_dir):
            full_file_name = os.path.join(cache_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, local_model_path)
        
        # 4. 清理临时文件夹
        shutil.rmtree('./temp_download')
        
        my_bar.progress(100, text="✅ 模型准备就绪！")
        my_bar.empty() # 隐藏进度条
        
        return local_model_path

    except Exception as e:
        st.error(f"❌ 下载失败: {e}")
        st.stop() # 停止程序运行

# ==============================================================================
# 🎨 Streamlit 应用程序主逻辑
# ==============================================================================

# 1. 页面配置
st.set_page_config(page_title="智能鸟类识别专家", page_icon="🦅")
st.title("🦅 智能鸟类识别专家")
st.markdown("### 国内极速版")
st.write("上传一张鸟类照片，AI 专家将为您鉴定（支持 555 种鸟类）。")

# 2. 加载模型 (带缓存)
@st.cache_resource
def load_pipeline():
    # 第一步：确保模型在本地
    model_path = check_and_download_model()
    
    # 第二步：加载模型
    try:
        classifier = pipeline("image-classification", model=model_path)
        return classifier
    except Exception as e:
        st.error(f"模型加载出错: {e}")
        return None

# 显示加载状态
with st.spinner('正在初始化 AI 引擎...'):
    classifier = load_pipeline()

# 3. 上传图片
uploaded_file = st.file_uploader("请选择图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示图片
    image = Image.open(uploaded_file)
    st.image(image, caption='您的照片', use_column_width=True)

    # 4. 识别按钮
    if st.button('🔍 开始鉴定', type="primary"):
        if classifier:
            with st.spinner('AI 正在分析羽毛特征...'):
                try:
                    # 推理
                    results = classifier(image)
                    
                    # 获取最佳结果
                    top_result = results[0]
                    english_name = top_result['label']
                    score = top_result['score']
                    
                    # 美化名字
                    formatted_name = english_name.replace("_", " ").title()

                    # 结果展示区
                    st.success("✅ 鉴定完成！")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("识别结果 (英文)", formatted_name)
                    with col2:
                        st.metric("置信度", f"{score:.1%}")
                    
                    st.info(f"💡 提示：您可以复制 **{formatted_name}** 去搜索引擎查询中文资料。")

                    # 更多可能性折叠面板
                    with st.expander("查看其他可能性"):
                        for res in results[1:4]:
                            name = res['label'].replace("_", " ").title()
                            st.write(f"**{name}**: {res['score']:.1%}")
                            st.progress(res['score'])

                except Exception as e:
                    st.error(f"识别过程发生错误: {e}")
