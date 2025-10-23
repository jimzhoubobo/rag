import os
import hashlib
import sqlite3
import asyncio
import aiosqlite
import functools
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


def _get_group_name(user_id: str, device_id: str, num_groups: int = 2) -> str:
    """
    根据用户ID或设备ID获取分组名称
    
    Args:
        user_id (str): 用户ID
        device_id (str): 设备ID
        num_groups (int): 分组数量，默认为2组
        
    Returns:
        str: 分组名称
    """
    # 如果用户ID为空，则使用设备ID进行哈希分组
    if not user_id and device_id:
        hash_object = hashlib.md5(device_id.encode())
        # 根据哈希值和分组数量计算分组编号
        last_char = hash_object.hexdigest()[-1]
        group_num = int(last_char, 16) % num_groups
        return f"group_{group_num}"
    elif user_id:
        hash_object = hashlib.md5(user_id.encode())
        # 根据哈希值和分组数量计算分组编号
        last_char = hash_object.hexdigest()[-1]
        group_num = int(last_char, 16) % num_groups
        return f"group_{group_num}"
    else:
        return "group_0"  # 匿名用户默认分到组0


def _save_qa_history(group_name: str, user_id: str, device_id: str, question: str, answer: str):
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


async def _save_qa_history_async(group_name: str, user_id: str, device_id: str, question: str, answer: str):
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
        _save_qa_history(group_name, user_id, device_id, question, answer)


def _handle_task_exception(task_name: str, task: asyncio.Task):
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


def get_qa_chain(vector_store, top_k: int = 4, user_id: str = None, device_id: str = None):
    """
    初始化DeepSeek LLM，创建并返回一个配置好的RetrievalQA链。
    
    Args:
        vector_store: 向量存储实例
        top_k (int): 检索的文档数量
        user_id (str): 用户ID
        device_id (str): 设备ID
        
    Returns:
        RetrievalQA: 配置好的问答链
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
    
    # 添加用户ID和设备ID到qa_chain的元数据中
    qa_chain.metadata = {
        "user_id": user_id,
        "device_id": device_id
    }
    
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
    
    # 获取用户ID和设备ID
    user_id = getattr(qa_chain, 'metadata', {}).get('user_id', None)
    device_id = getattr(qa_chain, 'metadata', {}).get('device_id', hashlib.md5(os.urandom(16)).hexdigest())
    
    # 获取分组名称
    group_name = _get_group_name(user_id, device_id)
    
    # 异步保存问答历史
    try:
        # 创建异步任务
        task = asyncio.create_task(
            _save_qa_history_async(group_name, user_id, device_id, question, result.get('result', ''))
        )
        # 添加任务完成回调，用于处理异常
        task.add_done_callback(functools.partial(_handle_task_exception, "保存问答历史"))
    except RuntimeError:
        # 如果事件循环未运行，则使用同步方法
        _save_qa_history(group_name, user_id, device_id, question, result.get('result', ''))
    
    return result