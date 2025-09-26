import streamlit as st
import os
from dotenv import load_dotenv
from rag_core import build_or_load_vector_store, get_qa_chain, get_answer
from etl.document_processor import load_and_process_documents
import tempfile

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="中医理疗智能问答系统",
    page_icon="🌿",
    layout="wide"
)

# 页面标题
st.title("中医理疗智能问答系统")

# 侧边栏
st.sidebar.header("配置选项")

# 文件上传器
uploaded_files = st.sidebar.file_uploader(
    "上传额外的PDF或TXT文件",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# 检索数量滑动条
top_k = st.sidebar.slider("检索文档数量", 1, 10, 4)

# 安全免责声明
st.sidebar.markdown("""
---
### ⚠️ 安全免责声明
本系统提供的所有信息仅供学习和参考之用，不能替代专业医生的诊断和治疗建议。
如有实际健康问题，请务必咨询合格的中医师或其他医疗专业人士。
本系统不对因使用或不正确使用本系统提供的信息而导致的任何后果负责。
""")

# 添加调试开关
debug_mode = st.sidebar.checkbox("调试模式")

# 初始化会话状态
if 'qa_chain' not in st.session_state:
    with st.spinner("正在初始化问答系统..."):
        # 处理数据目录中的文档
        data_path = "./data"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        # 加载和处理文档
        if debug_mode:
            st.write("正在加载文档...")
        
        documents = load_and_process_documents(data_path)
        
        if debug_mode:
            st.write(f"已加载 {len(documents)} 个文档片段")
        
        # 处理上传的文件
        if uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # 加载上传的文档
            uploaded_documents = load_and_process_documents(temp_dir)
            documents.extend(uploaded_documents)
        
        # 构建或加载向量库
        if debug_mode:
            st.write("正在构建向量库...")
            
        vector_store = build_or_load_vector_store(documents, "./chroma_db")
        
        # 创建QA链
        if debug_mode:
            st.write("正在创建问答链...")
            
        st.session_state.qa_chain = get_qa_chain(vector_store)
        
        if debug_mode:
            st.write("系统初始化完成!")

# 主界面 - 问题输入
question = st.chat_input("请输入您的问题...")

# 处理问题和显示答案
if question:
    with st.spinner("正在生成答案..."):
        if debug_mode:
            st.write(f"问题: {question}")
            
        result = get_answer(question, st.session_state.qa_chain)
        
        if debug_mode:
            st.write("答案生成完成")
        
        # 显示问题
        st.subheader("问题")
        st.write(question)
        
        # 显示答案
        st.subheader("答案")
        st.markdown(result["result"])
        
        # 显示参考资料
        with st.expander("参考资料"):
            for doc in result["source_documents"]:
                st.markdown(f"**来源**: {doc.metadata.get('source', 'Unknown')}")
                st.markdown(f"**内容**: {doc.page_content}")
                st.divider()