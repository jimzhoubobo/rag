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

from rag.rag_core import load_vector_store_with_cache

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# 加载环境变量
load_dotenv()

from log.logger import logger
from constant.constants import ProjectConstants
from langchain.docstore.document import Document



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
