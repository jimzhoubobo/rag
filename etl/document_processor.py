#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档处理模块
负责加载和处理PDF、TXT等文档
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from rag.logger import logger


def load_and_process_documents(data_path: str):
    """
    加载并处理指定路径下的所有文档
    
    Args:
        data_path (str): 文档目录路径
        
    Returns:
        list: 处理后的文档片段列表
    """
    logger.info(f"开始加载文档，路径: {data_path}")
    
    documents = []
    
    # 支持的文件类型
    supported_extensions = {'.pdf', '.txt'}
    
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            try:
                if file_extension == '.pdf':
                    logger.info(f"加载PDF文件: {file_path}")
                    loader = PyPDFLoader(file_path)
                    pdf_docs = loader.load()
                    logger.info(f"已加载PDF文件: {file_path}, 文档数: {len(pdf_docs)}")
                    documents.extend(pdf_docs)
                    
                elif file_extension == '.txt':
                    logger.info(f"加载TXT文件: {file_path}")
                    loader = TextLoader(file_path, encoding='utf-8')
                    txt_docs = loader.load()
                    logger.info(f"已加载TXT文件: {file_path}, 文档数: {len(txt_docs)}")
                    documents.extend(txt_docs)
                    
            except Exception as e:
                logger.error(f"加载文件 {file_path} 时出错: {e}")
                continue
    
    # 如果没有找到支持的文件
    if not documents:
        logger.warning(f"在路径 {data_path} 中未找到支持的文档文件")
        return []
    
    # 分割文档
    logger.info(f"开始分割文档，总文档数: {len(documents)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    logger.info(f"文档分割完成，总共 {len(splits)} 个片段")
    
    return splits


def extract_content_and_store(documents, storage_path: str = None):
    """
    提取文档内容并存储到指定位置
    
    Args:
        documents (list): 文档列表
        storage_path (str): 存储路径，如果为None则不存储到文件
        
    Returns:
        list: 处理后的文档内容列表
    """
    extracted_content = []
    
    for i, doc in enumerate(documents):
        content = {
            "id": i,
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        extracted_content.append(content)
        
        # 如果指定了存储路径，则将内容保存到文件
        if storage_path:
            # 确保存储路径存在
            os.makedirs(storage_path, exist_ok=True)
            # 创建文件名
            filename = f"document_{i}.json"
            file_path = os.path.join(storage_path, filename)
            
            # 写入内容和元数据
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=4)
    
    logger.info(f"已提取并处理 {len(extracted_content)} 个文档片段")
    return extracted_content


if __name__ == "__main__":
    # 测试代码
    data_path = "../data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    documents = load_and_process_documents(data_path)
    extracted = extract_content_and_store(documents, "../processed_data")
    logger.info(f"处理完成，共处理 {len(extracted)} 个文档片段")
