import os

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn # 用于运行服务器
from dotenv import load_dotenv
load_dotenv()
# 创建FastAPI应用实例
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="一个简单的API服务器，使用LangChain的可运行接口"
)

# 构建一个简单的链：提示词模板 + 大模型
# 提示词模板中，"topic"是用户输入的变量
prompt_template = ChatPromptTemplate.from_template("请给我讲一个关于{topic}的笑话")
# model = ChatOpenAI(model="gpt-3.5-turbo") # 初始化OpenAI聊天模型，请确保已设置OPENAI_API_KEY环境变量
api_key = os.getenv("DEEPSEEK_API_KEY")
model = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2000,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )
chain = prompt_template | model # 使用LCEL（LangChain Expression Language）将提示词和模型组合成链

# 关键一步：将链添加为路由，并指定访问路径为"/joke"
add_routes(app, chain, path="/joke")

# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)