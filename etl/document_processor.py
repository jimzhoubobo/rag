#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档处理ETL模块
负责加载和处理指定目录下的所有.pdf和.txt文档
"""

import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document


def load_and_process_documents(data_path: str):
    """
    加载并处理指定目录下的所有.pdf和.txt文档，返回分块后的文档列表。
    
    Args:
        data_path (str): 文档目录路径
        
    Returns:
        list: 分块后的文档列表
    """
    documents = []
    
    # 加载PDF文件
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    print(f"找到 {len(pdf_files)} 个PDF文件")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            print(f"已加载PDF文件: {pdf_file}, 文档数: {len(docs)}")
        except Exception as e:
            print(f"加载PDF文件 {pdf_file} 时出错: {e}")
    
    # 加载TXT文件
    txt_files = glob.glob(os.path.join(data_path, "*.txt"))
    print(f"找到 {len(txt_files)} 个TXT文件")
    for txt_file in txt_files:
        try:
            loader = TextLoader(txt_file, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
            print(f"已加载TXT文件: {txt_file}, 文档数: {len(docs)}")
        except Exception as e:
            print(f"加载TXT文件 {txt_file} 时出错: {e}")
    
    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )
    
    split_documents = text_splitter.split_documents(documents)
    print(f"文档分割完成，总共 {len(split_documents)} 个片段")
    return split_documents


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
            filename = f"document_{i}.txt"
            file_path = os.path.join(storage_path, filename)
            
            # 写入内容和元数据
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Content:\n{doc.page_content}\n\n")
                f.write(f"Metadata:\n{str(doc.metadata)}\n")
    
    print(f"已提取并处理 {len(extracted_content)} 个文档片段")
    return extracted_content


if __name__ == "__main__":
    # 测试代码
    data_path = "../data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    documents = load_and_process_documents(data_path)
    extracted = extract_content_and_store(documents, "../processed_data")
    print(f"处理完成，共处理 {len(extracted)} 个文档片段")