import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from log.logger import logger
from cache.cache import ttl_cache

# 导入向量库加载函数
from etl.vector_builder import load_vector_store
# 导入版本管理器
from etl.vector_version_manager import vector_version_manager


@ttl_cache(expire_time=600)  # 10分钟缓存
def load_vector_store_with_cache(documents, persist_directory: str):
    """
    从持久化目录加载Chroma向量库，使用缓存机制。
    
    Args:
        documents: 文档列表（此参数仅为保持接口兼容性，实际不使用）
        persist_directory (str): 向量库存储目录
        
    Returns:
        Chroma: 加载的向量库实例
    """
    # 使用版本管理器获取当前活动版本路径
    active_version_path = vector_version_manager.get_active_version_path()
    if active_version_path and os.path.exists(active_version_path):
        return load_vector_store(active_version_path)
    else:
        # 如果没有活动版本或路径不存在，回退到原来的persist_directory
        return load_vector_store(persist_directory)


def _create_prompt_template():
    """创建专业的Prompt模板"""
    template = """你是一位经验丰富的经络理疗专家。请严格根据提供的上下文信息来回答问题。

上下文信息：
{context}

用户问题：{question}

请根据以上上下文提供专业、准确的经络理疗建议。如果上下文信息不足以回答问题，请明确告知。

重要安全免责声明：
请注意，本系统提供的所有信息仅供学习和参考之用，不能替代专业医生的诊断和治疗建议。
如有实际健康问题，请务必咨询合格的经络师或其他医疗专业人士。
本系统不对因使用或不正确使用本系统提供的信息而导致的任何后果负责。

请用中文回答，回答要结构清晰、专业易懂。"""

    return PromptTemplate.from_template(template)


def get_qa_chain(vector_store, top_k: int = 4):
    """
    初始化DeepSeek LLM，创建并返回一个配置好的RetrievalQA链。
    """
    logger.info("初始化DeepSeek LLM...")
    # 初始化LLM，使用DeepSeek兼容的ChatOpenAI接口
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY 环境变量未设置")
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
        
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2000,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )
    logger.info("LLM初始化完成")
    
    # 创建Prompt模板
    prompt = _create_prompt_template()
    
    # 创建RetrievalQA链
    logger.info("创建QA链...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    logger.info("QA链创建完成")
    
    return qa_chain


def get_answer(question, qa_chain, top_k: int = 4):
    """
    一个工具函数，用于执行问答链并返回答案。
    """
    logger.info(f"开始处理问题: {question}")
    # 更新 retriever 的 k 值
    qa_chain.retriever.search_kwargs["k"] = top_k
    result = qa_chain.invoke({"query": question})
    logger.info("问题处理完成")
    return result