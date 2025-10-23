# 首先安装必要的库
# %pip install --upgrade --quiet langchain langchain-openai

import os
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 假设你已设置好API密钥，例如：os.environ["OPENAI_API_KEY"] = "your-api-key"

def func1(z):
    return z.num * 4


# 创建一个可并行运行的处理链
runnable = RunnableParallel(
    extra=RunnablePassthrough.assign(content=lambda x: x["num"] * 3,c2 = lambda x: x["num"] * 2), # 在保留原输入的基础上，新增一个'mult'键
    modified=lambda x: x["num"] + 1, # 直接对输入进行处理
)

# 调用这个链
result = runnable.invoke({"num": 1})
print(result)