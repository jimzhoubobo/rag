#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量库构建模块
负责构建和更新Chroma向量库
"""

import os
from functools import lru_cache

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from log.logger import logger
from constant.constants import VECTOR_STORE_BATCH_SIZE
from util.tools import ListUtils

from constant.constants import VECTOR_STORE_BATCH_SIZE
from util.tools import ListUtils


@lru_cache(maxsize=1)
def init_embedding():
    """
    初始化向量嵌入模型
    
    Returns:
        embedding model: 初始化的嵌入模型
    """
    try:

        # 使用本地模型路径 1024 本地跑起来比较大
        # model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-large-zh-v1.5")
        # embedding = SentenceTransformerEmbeddings(model_name=model_path)

        # 使用多语言支持的嵌入模型
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("嵌入模型初始化成功")
        # 使用本地模型路径
        # model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-large-zh-v1.5")
        # embedding = SentenceTransformerEmbeddings(model_name=model_path)
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("BGE嵌入模型初始化成功")
        return embedding
    except Exception as e:
        # 如果主要模型加载失败，使用替代方案
        logger.warning(f"主要模型加载失败，使用替代方案: {e}")
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("使用替代方案初始化嵌入模型成功")
        return embedding


def build_vector_store(documents, persist_directory: str, batch_size: int = 50):
def build_vector_store(documents, persist_directory: str, batch_size: int = 50):
    """
    分批构建Chroma向量库
    分批构建Chroma向量库
    
    Args:
        documents (list): 文档列表
        persist_directory (str): 向量库存储目录
        batch_size (int): 每批处理的文档数量，默认使用常量VECTOR_STORE_BATCH_SIZE
        batch_size (int): 每批处理的文档数量，默认使用常量VECTOR_STORE_BATCH_SIZE
        
    Returns:
        Chroma: 构建好的向量库实例
    """
    logger.info(f"开始分批构建向量库，总文档数: {len(documents)}, 批大小: {batch_size}")
    logger.info(f"开始分批构建向量库，总文档数: {len(documents)}, 批大小: {batch_size}")
    try:
        embedding = init_embedding()
        vector_store = None
        # 使用enumerate和切片更优雅地处理批次
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(documents), batch_size), 1):
            batch_documents = documents[i:i + batch_size]
            
            logger.info(f"处理批次 {batch_num}/{total_batches}, 文档数: {len(batch_documents)}")
            
            if vector_store is None:
                # 第一批创建新的向量库
                vector_store = Chroma.from_documents(
                    documents=batch_documents,
                    embedding=embedding,
                    persist_directory=persist_directory
                )
                logger.info(f"第 {batch_num} 批向量库创建完成")
            else:
                # 后续批次添加到现有向量库
                vector_store.add_documents(batch_documents)
                logger.info(f"第 {batch_num} 批文档添加完成")
        
        logger.info("所有批次处理完成，向量库构建完成")
        return vector_store
        
        vector_store = None
        # 使用enumerate和切片更优雅地处理批次
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(documents), batch_size), 1):
            batch_documents = documents[i:i + batch_size]
            
            logger.info(f"处理批次 {batch_num}/{total_batches}, 文档数: {len(batch_documents)}")
            
            if vector_store is None:
                # 第一批创建新的向量库
                vector_store = Chroma.from_documents(
                    documents=batch_documents,
                    embedding=embedding,
                    persist_directory=persist_directory
                )
                logger.info(f"第 {batch_num} 批向量库创建完成")
            else:
                # 后续批次添加到现有向量库
                vector_store.add_documents(batch_documents)
                logger.info(f"第 {batch_num} 批文档添加完成")
        
        logger.info("所有批次处理完成，向量库构建完成")
        return vector_store
        
    except Exception as e:
        # 如果BGE模型加载失败，使用替代方案
        logger.warning(f"BGE模型加载失败，使用替代方案: {e}")
        embedding = init_embedding()
        vector_store = None
        
        # 使用enumerate和切片更优雅地处理批次
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for batch_num, batch_documents in enumerate(ListUtils.chunk_list(documents,batch_size), 1):
            logger.info(f"[备用方案] 处理批次 {batch_num}/{total_batches}, 文档数: {len(batch_documents)}")
            
            if vector_store is None:
                # 第一批创建新的向量库
                vector_store = Chroma.from_documents(
                    documents=batch_documents,
                    embedding=embedding,
                    persist_directory=persist_directory
                )
                logger.info(f"[备用方案] 第 {batch_num} 批向量库创建完成")
            else:
                # 后续批次添加到现有向量库
                vector_store.add_documents(batch_documents)
                logger.info(f"[备用方案] 第 {batch_num} 批文档添加完成")
        
        logger.info("[备用方案] 所有批次处理完成，向量库构建完成")
        return vector_store
        vector_store = None
        
        # 使用enumerate和切片更优雅地处理批次
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for batch_num, batch_documents in enumerate(ListUtils.chunk_list(documents,batch_size), 1):
            logger.info(f"[备用方案] 处理批次 {batch_num}/{total_batches}, 文档数: {len(batch_documents)}")
            
            if vector_store is None:
                # 第一批创建新的向量库
                vector_store = Chroma.from_documents(
                    documents=batch_documents,
                    embedding=embedding,
                    persist_directory=persist_directory
                )
                logger.info(f"[备用方案] 第 {batch_num} 批向量库创建完成")
            else:
                # 后续批次添加到现有向量库
                vector_store.add_documents(batch_documents)
                logger.info(f"[备用方案] 第 {batch_num} 批文档添加完成")
        
        logger.info("[备用方案] 所有批次处理完成，向量库构建完成")
        return vector_store


def update_vector_store(documents, persist_directory: str, batch_size: int = VECTOR_STORE_BATCH_SIZE):
def update_vector_store(documents, persist_directory: str, batch_size: int = VECTOR_STORE_BATCH_SIZE):
    """
    分批更新现有的Chroma向量库
    分批更新现有的Chroma向量库
    
    Args:
        documents (list): 要添加到向量库的新文档
        persist_directory (str): 向量库存储目录
        batch_size (int): 每批处理的文档数量，默认使用常量VECTOR_STORE_BATCH_SIZE
        batch_size (int): 每批处理的文档数量，默认使用常量VECTOR_STORE_BATCH_SIZE
        
    Returns:
        Chroma: 更新后的向量库实例
    """
    if not documents:
        logger.info("没有新文档需要添加到向量库")
        return load_vector_store(persist_directory)
    
    logger.info(f"开始分批更新向量库，新增文档数: {len(documents)}, 批大小: {batch_size}")
    logger.info(f"开始分批更新向量库，新增文档数: {len(documents)}, 批大小: {batch_size}")
    
    try:
        # 加载现有向量库
        vector_store = load_vector_store(persist_directory)
        
        # 使用enumerate和切片更优雅地处理批次
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for batch_num, batch_documents in enumerate(ListUtils.chunk_list(documents,batch_size), 1):
            logger.info(f"添加批次 {batch_num}/{total_batches}, 文档数: {len(batch_documents)}")
            vector_store.add_documents(batch_documents)
            logger.info(f"批次 {batch_num} 文档添加完成")
        
        # 使用enumerate和切片更优雅地处理批次
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for batch_num, batch_documents in enumerate(ListUtils.chunk_list(documents,batch_size), 1):
            logger.info(f"添加批次 {batch_num}/{total_batches}, 文档数: {len(batch_documents)}")
            vector_store.add_documents(batch_documents)
            logger.info(f"批次 {batch_num} 文档添加完成")
        
        logger.info("向量库更新完成")
        return vector_store
    except Exception as e:
        logger.error(f"更新向量库时出错: {e}")
        # 如果更新失败，重新构建向量库
        return build_vector_store(documents, persist_directory, batch_size)
        return build_vector_store(documents, persist_directory, batch_size)


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