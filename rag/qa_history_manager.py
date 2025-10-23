#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问答历史管理模块
负责管理问答历史的保存、查询等功能
"""

import os
import sqlite3
import asyncio
import aiosqlite
from log.logger import logger


def _initialize_database():
    """初始化SQLite数据库和表"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qa_history.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建表来存储问答历史
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT NOT NULL,
            user_id TEXT,
            device_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    return db_path


async def _initialize_database_async():
    """异步初始化SQLite数据库和表"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qa_history.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS qa_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_name TEXT NOT NULL,
                user_id TEXT,
                device_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await db.commit()
    return db_path


def save_qa_history(group_name: str, user_id: str, device_id: str, question: str, answer: str):
    """
    保存问答历史到SQLite数据库
    
    Args:
        group_name (str): 分组名称
        user_id (str): 用户ID
        device_id (str): 设备ID
        question (str): 用户问题
        answer (str): 回答内容
    """
    try:
        db_path = _initialize_database()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO qa_history (group_name, user_id, device_id, question, answer)
            VALUES (?, ?, ?, ?, ?)
        ''', (group_name, user_id, device_id, question, answer))
        
        conn.commit()
        conn.close()
        logger.info("问答历史保存成功")
    except Exception as e:
        logger.error(f"保存问答历史时出错: {e}")


async def save_qa_history_async(group_name: str, user_id: str, device_id: str, question: str, answer: str):
    """
    异步保存问答历史到SQLite数据库
    
    Args:
        group_name (str): 分组名称
        user_id (str): 用户ID
        device_id (str): 设备ID
        question (str): 用户问题
        answer (str): 回答内容
    """
    try:
        db_path = await _initialize_database_async()
        async with aiosqlite.connect(db_path) as db:
            await db.execute('''
                INSERT INTO qa_history (group_name, user_id, device_id, question, answer)
                VALUES (?, ?, ?, ?, ?)
            ''', (group_name, user_id, device_id, question, answer))
            await db.commit()
        logger.info("问答历史异步保存成功")
    except Exception as e:
        logger.error(f"异步保存问答历史时出错: {e}")
        # 异常时尝试同步保存作为兜底方案
        save_qa_history(group_name, user_id, device_id, question, answer)


def handle_task_exception(task_name: str, task: asyncio.Task):
    """
    处理异步任务异常的回调函数
    
    Args:
        task_name (str): 任务名称
        task (asyncio.Task): 异步任务对象
    """
    try:
        # 获取任务结果，这会抛出任务中未处理的异常
        task.result()
    except Exception as e:
        logger.error(f"异步任务 {task_name} 执行出错: {e}")
        # 可以在这里添加额外的错误处理逻辑