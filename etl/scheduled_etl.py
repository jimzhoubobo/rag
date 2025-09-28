#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
定时ETL任务
每30分钟执行一次文档加载和处理任务
"""

import schedule
import time
import os
import sys
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from etl.document_processor import load_and_process_documents, extract_and_save_content
from etl.vector_builder import build_vector_store
from etl.vector_version_manager import vector_version_manager
from log.logger import logger
from constant.constants import ProjectConstants


def etl_job():
    """
    ETL任务函数
    """
    logger.info("开始执行ETL任务...")
    
    try:
        # 定义数据路径
        data_path = ProjectConstants.get_data_path()
        processed_data_path = ProjectConstants.get_processed_data_path()
        
        # 确保数据目录存在
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(processed_data_path, exist_ok=True)
        
        # 加载和处理文档
        documents = load_and_process_documents(data_path)
        
        # 提取内容并存储
        # extracted_content = extract_and_save_content(documents, processed_data_path)
        
        # 使用版本管理器创建新版本并向量库
        if documents:
            success = vector_version_manager.switch_to_new_version(documents)
            if success:
                logger.info(f"ETL任务执行完成，共处理 {len(documents)} 个文档片段，向量库版本已更新")
            else:
                logger.error("向量库版本更新失败")
        else:
            logger.info("没有文档需要处理")
        
    except Exception as e:
        logger.error(f"ETL任务执行出错: {e}")


def run_scheduler():
    """
    运行定时调度器
    """
    # 每30分钟执行一次
    schedule.every(30).minutes.do(etl_job)
    
    # 每天凌晨5点执行一次
    schedule.every().day.at("05:00").do(etl_job)
    
    # 立即执行一次
    etl_job()
    
    logger.info("定时任务已启动:")
    logger.info("  - 每30分钟执行一次")
    logger.info("  - 每天凌晨5点执行一次")
    
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    run_scheduler()