import logging
import os
from logging.handlers import RotatingFileHandler

# 创建日志目录
log_dir = "/apps/logs/tcm-rag-qa"
os.makedirs(log_dir, exist_ok=True)

# 日志文件路径
log_file = os.path.join(log_dir, "tcm-rag-qa.log")

# 创建日志记录器
logger = logging.getLogger("tcm_rag_qa")
logger.setLevel(logging.INFO)

# 创建滚动文件处理器(最大10MB，保留10个备份文件)
handler = RotatingFileHandler(
    log_file, 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=10,
    encoding='utf-8'
)

# 创建格式化器
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

# 添加处理器到记录器
logger.addHandler(handler)

# 如果在调试模式下，也输出到控制台
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)