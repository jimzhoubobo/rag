#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG核心模块LCEL重构测试脚本
用于测试重构后的get_qa_chain函数
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# 加载环境变量
load_dotenv()

from log.logger import logger
from constant.constants import ProjectConstants
from langchain.docstore.document import Document
from rag.rag_core import load_vector_store_with_cache

def test_vector_store():
    """测试向量库功能"""
    logger.info("=== 测试向量库功能 ===")
    try:
        persist_directory = ProjectConstants.get_chroma_db_path()
        
        # 创建简单的测试文档
        documents = [
            Document(page_content="经络是中医理论的重要组成部分，它是运行气血、联络脏腑肢节、沟通上下内外的通路。", metadata={"source": "test"}),
            Document(page_content="针灸是通过对穴位的刺激来调节人体气血运行的一种治疗方法。", metadata={"source": "test"}),
            Document(page_content="拔罐是一种通过在皮肤上造成负压来促进血液循环的疗法。", metadata={"source": "test"})
        ]
        logger.info("使用测试文档进行向量库测试")
        
        vector_store = load_vector_store_with_cache(documents, persist_directory)
        logger.info("向量库加载成功")
        
        # 测试相似性搜索
        logger.info("测试相似性搜索...")
        search_results = vector_store.similarity_search("中医", k=1)
        logger.info(f"搜索到 {len(search_results)} 个结果")
        if search_results:
            logger.info(f"第一个结果: {search_results[0].page_content[:100]}...")
        
        return vector_store
    except Exception as e:
        logger.error(f"向量库测试失败: {e}")
        traceback.print_exc()
        return None


def test_qa_chain_with_relevant_question():
    """测试问答链功能 - 相关问题"""
    logger.info("=== 测试问答链功能 - 相关问题 ===")
    vector_store = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
    try:
        from rag.rag_core import get_qa_chain
        # 测试用户ID为group_0的用户（应该使用fallback_chain_a）
        qa_chain = get_qa_chain(vector_store, user_id="user_group_0")
        logger.info("问答链创建成功")
        
        # 测试相关问题
        test_question = "经络是什么？"
        logger.info(f"测试相关问题: {test_question}")
        
        result = qa_chain.invoke({"question": test_question})
        logger.info("答案获取成功")
        logger.info(f"答案: {result}")
        
        return True
    except Exception as e:
        logger.error(f"问答链测试失败: {e}")
        traceback.print_exc()
        return False


def test_qa_chain_with_irrelevant_question():
    """测试问答链功能 - 不相关问题"""
    logger.info("=== 测试问答链功能 - 不相关问题 ===")
    vector_store = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
    try:
        from rag.rag_core import get_qa_chain
        # 测试用户ID为group_1的用户（应该使用fallback_chain_b）
        qa_chain = get_qa_chain(vector_store, user_id="user_group_1")
        logger.info("问答链创建成功")
        
        # 测试不相关问题（应该触发回退机制）
        test_question = "今天天气怎么样？"
        logger.info(f"测试不相关问题: {test_question}")
        
        result = qa_chain.invoke({"question": test_question})
        logger.info("答案获取成功")
        logger.info(f"答案: {result}")
        
        return True
    except Exception as e:
        logger.error(f"问答链测试失败: {e}")
        traceback.print_exc()
        return False


def test_qa_chain_with_anonymous_user():
    """测试问答链功能 - 匿名用户"""
    logger.info("=== 测试问答链功能 - 匿名用户 ===")
    vector_store = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
    try:
        from rag.rag_core import get_qa_chain
        # 测试匿名用户（应该使用fallback_chain_a，即group_0）
        qa_chain = get_qa_chain(vector_store)
        logger.info("问答链创建成功")
        
        # 测试问题
        test_question = "今天天气怎么样？"
        logger.info(f"测试匿名用户问题: {test_question}")
        
        result = qa_chain.invoke({"question": test_question})
        logger.info("答案获取成功")
        logger.info(f"答案: {result}")
        
        return True
    except Exception as e:
        logger.error(f"问答链测试失败: {e}")
        traceback.print_exc()
        return False


def test_get_answer_function():
    """测试get_answer函数"""
    logger.info("=== 测试get_answer函数 ===")
    vector_store = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
    try:
        from rag.rag_core import get_qa_chain, get_answer
        qa_chain = get_qa_chain(vector_store)
        logger.info("问答链创建成功")
        
        test_question = "针灸是什么？"
        logger.info(f"测试问题: {test_question}")
        
        result = get_answer(test_question, qa_chain)
        logger.info("答案获取成功")
        logger.info(f"答案: {result.get('result', '未找到答案')}")
        
        return True
    except Exception as e:
        logger.error(f"获取答案测试失败: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_vector_store()
    test_qa_chain_with_relevant_question()
    test_qa_chain_with_irrelevant_question()
    test_qa_chain_with_anonymous_user()
    test_get_answer_function()
    logger.info("所有测试完成")