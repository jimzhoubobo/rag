#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG核心模块测试脚本
用于直接测试rag_core.py中的各个功能模块，便于后台调试
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# 加载环境变量
load_dotenv()

from rag.logger import logger

def test_document_loading():
    """测试文档加载功能"""
    logger.info("=== 测试文档加载功能 ===")
    try:
        from etl.document_processor import load_and_process_documents
        data_path = "./data"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            logger.info(f"创建数据目录: {data_path}")
        
        documents = load_and_process_documents(data_path)
        logger.info(f"成功加载 {len(documents)} 个文档片段")
        if documents:
            logger.info("第一个文档片段预览:")
            logger.info(f"内容: {documents[0].page_content[:100]}...")
            logger.info(f"元数据: {documents[0].metadata}")
        return documents
    except Exception as e:
        logger.error(f"文档加载测试失败: {e}")
        traceback.print_exc()
        return None

def test_vector_store(documents):
    """测试向量库功能"""
    logger.info("=== 测试向量库功能 ===")
    try:
        from rag.rag_core import build_or_load_vector_store
        persist_directory = "./chroma_db"
        
        if documents is None:
            # 创建简单的测试文档
            from langchain.docstore.document import Document
            documents = [
                Document(page_content="这是一个测试文档，用于测试向量库功能。", metadata={"source": "test"}),
                Document(page_content="这是另一个测试文档，用于验证向量库的存储和检索功能。", metadata={"source": "test"})
            ]
            logger.info("使用测试文档进行向量库测试")
        
        vector_store = build_or_load_vector_store(documents, persist_directory)
        logger.info("向量库创建/加载成功")
        
        # 测试相似性搜索
        logger.info("测试相似性搜索...")
        search_results = vector_store.similarity_search("测试", k=1)
        logger.info(f"搜索到 {len(search_results)} 个结果")
        if search_results:
            logger.info(f"第一个结果: {search_results[0].page_content[:100]}...")
        
        return vector_store
    except Exception as e:
        logger.error(f"向量库测试失败: {e}")
        traceback.print_exc()
        return None

def test_qa_chain(vector_store):
    """测试问答链功能"""
    logger.info("=== 测试问答链功能 ===")
    try:
        from rag.rag_core import get_qa_chain
        qa_chain = get_qa_chain(vector_store)
        logger.info("问答链创建成功")
        return qa_chain
    except Exception as e:
        logger.error(f"问答链测试失败: {e}")
        traceback.print_exc()
        return None

def test_get_answer(qa_chain):
    """测试获取答案功能"""
    logger.info("=== 测试获取答案功能 ===")
    try:
        from rag.rag_core import get_answer
        test_question = "什么是中医理疗？"
        logger.info(f"测试问题: {test_question}")
        
        result = get_answer(test_question, qa_chain)
        logger.info("答案获取成功")
        logger.info(f"答案: {result.get('result', '未找到答案')}")
        
        # 显示源文档
        source_docs = result.get('source_documents', [])
        if source_docs:
            logger.info(f"找到 {len(source_docs)} 个源文档:")
            for i, doc in enumerate(source_docs[:2]):  # 只显示前2个
                logger.info(f"  源文档 {i+1}: {doc.page_content[:100]}...")
        else:
            logger.info("未找到相关源文档")
            
        return result
    except Exception as e:
        logger.error(f"获取答案测试失败: {e}")
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    logger.info("RAG核心模块测试开始")
    logger.info("=" * 50)
    
    # 测试1: 文档加载
    documents = test_document_loading()
    
    # 测试2: 向量库
    vector_store = test_vector_store(documents)
    
    # 测试3: 问答链
    if vector_store:
        qa_chain = test_qa_chain(vector_store)
        
        # 测试4: 获取答案
        if qa_chain:
            test_get_answer(qa_chain)
    
    logger.info("=" * 50)
    logger.info("RAG核心模块测试结束")

if __name__ == "__main__":
    main()