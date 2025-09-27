#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量库构建模块
负责构建和更新Chroma向量库
"""

import os
from functools import lru_cache

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from log.logger import logger

@lru_cache(maxsize=1)
def init_embedding():
    """
    初始化向量嵌入模型
    
    Returns:
        embedding model: 初始化的嵌入模型
    """
    try:
        # 使用本地模型路径
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-large-zh-v1.5")
        embedding = SentenceTransformerEmbeddings(model_name=model_path)
        logger.info("BGE嵌入模型初始化成功")
        return embedding
    except Exception as e:
        # 如果BGE模型加载失败，使用替代方案
        logger.warning(f"BGE模型加载失败，使用替代方案: {e}")
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("使用替代方案初始化嵌入模型成功")
        return embedding


def build_vector_store(documents, persist_directory: str):
    """
    构建Chroma向量库
    
    Args:
        documents (list): 文档列表
        persist_directory (str): 向量库存储目录
        
    Returns:
        Chroma: 构建好的向量库实例
    """
    logger.info(f"开始构建向量库，文档数: {len(documents)}")
    
    # 如果没有文档，创建一个测试文档
    if len(documents) == 0:
        logger.info("没有文档，创建测试文档...")
        from langchain.docstore.document import Document
        documents = [Document(page_content="这是测试文档内容，用于初始化向量库。", metadata={"source": "test"})]
    
    # 创建新的向量库
    logger.info("创建新的向量库...")
    try:
        embedding = init_embedding()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
        logger.info("新向量库创建完成")
    except Exception as e:
        # 如果BGE模型加载失败，使用替代方案
        logger.warning(f"BGE模型加载失败，使用替代方案: {e}")
        embedding = init_embedding()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
        logger.info("使用替代方案创建向量库完成")
    
    return vector_store


def update_vector_store(documents, persist_directory: str):
    """
    更新现有的Chroma向量库
    
    Args:
        documents (list): 要添加到向量库的新文档
        persist_directory (str): 向量库存储目录
        
    Returns:
        Chroma: 更新后的向量库实例
    """
    if not documents:
        logger.info("没有新文档需要添加到向量库")
        return load_vector_store(persist_directory)
    
    logger.info(f"开始更新向量库，新增文档数: {len(documents)}")
    
    try:
        # 加载现有向量库
        vector_store = load_vector_store(persist_directory)
        
        # 添加新文档
        vector_store.add_documents(documents)
        logger.info("向量库更新完成")
        return vector_store
    except Exception as e:
        logger.error(f"更新向量库时出错: {e}")
        # 如果更新失败，重新构建向量库
        return build_vector_store(documents, persist_directory)


def load_vector_store(persist_directory: str):
    """
    加载现有的Chroma向量库
    
    Args:
        persist_directory (str): 向量库存储目录
        
    Returns:
        Chroma: 加载的向量库实例
    """
    logger.info("加载现有向量库...")
    
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        raise FileNotFoundError(f"向量库目录 {persist_directory} 不存在或为空")
    
    try:
        embedding = init_embedding()
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        logger.info("现有向量库加载完成")
    except Exception as e:
        # 如果BGE模型加载失败，使用替代方案
        logger.warning(f"BGE模型加载失败，使用替代方案: {e}")
        embedding = init_embedding()
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    
    return vector_store