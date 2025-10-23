from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
# (假设 compression_retriever 已经如上文定义好了)

llm = ChatOpenAI(temperature=0)

# --- 1. 定义 RAG 链 (主流程) ---
# 这个 Prompt 强制模型必须基于上下文
rag_prompt = ChatPromptTemplate.from_template(
    """
请根据以下上下文来回答问题。
如果上下文中没有提到相关信息，请明确说明“根据提供的资料，我无法回答该问题”。

上下文:
{context}

问题: {question}
回答: 
"""
)

# 辅助函数：将文档列表格式化为字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 链本身
rag_chain = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["docs"]))) # 1. 格式化 docs
    | rag_prompt                                                          # 2. 插入 Prompt
    | llm                                                                 # 3. LLM 生成
    | StrOutputParser()                                                   # 4. 解析输出
)

# --- 2. 定义 Fallback 链 (回退流程) ---

# 方案 A: 回退到“模型自己回答”
fallback_prompt = ChatPromptTemplate.from_template(
    """
(系统提示：未能在知识库中检索到相关信息，请使用你自己的知识来回答)
问题: {question}
回答: 
"""
)
fallback_chain_A = fallback_prompt | llm | StrOutputParser()

# 方案 B: 回退到“说不知道”
fallback_prompt_B = ChatPromptTemplate.from_template(
    """
(系统提示：未能在知识库中检索到相关信息)
请直接回答：“抱歉，我不知道这个问题的答案。”
问题: {question}
回答: 
"""
)
# 注意：这里我们甚至可以不经过 LLM，直接返回一个固定答案
# from langchain_core.runnables import RunnableLambda
# fallback_chain_B = RunnableLambda(lambda x: "抱歉，根据所提供的资料，我无法回答该问题。")
# 或者让 LLM 说
fallback_chain_B = fallback_prompt_B | llm | StrOutputParser()


# --- 3. 使用 RunnableBranch 组装 ---

# (我们选择方案 B 作为回退)
fallback_chain = fallback_chain_B

# 关键：这个函数用来判断是走 RAG 还是走 Fallback
# 它检查 retrieval_step 的输出中 "docs" 列表的长度
def route(input_dict):
    if len(input_dict["docs"]) == 0:
        return fallback_chain # 如果 "docs" 为空，路由到 fallback_chain
    else:
        return rag_chain      # 否则，路由到 rag_chain

# 这是第一步：获取输入 {"question": ...}，然后调用压缩检索器
# 输出会是 {"question": ..., "docs": [...]}
# 我们使用上面定义的 compression_retriever
retrieval_step = RunnablePassthrough.assign(
    docs=compression_retriever
)

# --- 4. 构建完整的链 ---

# 1. 先执行 retrieval_step (获取问题，并进行“粗召回 + 精排序”)
# 2. 将 {"question": ..., "docs": [...]} 传递给 route 函数
# 3. route 函数根据 docs 是否为空，选择执行 rag_chain 还是 fallback_chain
full_robust_chain = retrieval_step | RunnableLambda(route)

# --- 5. 执行 ---

question1 = "数据库中存在的问题..." # 假设能检索到
question2 = "今天天气怎么样？"      # 假设完全检索不到

# 场景1：能检索到
# retrieval_step -> docs 列表不为空 -> route() 选择 rag_chain
print(full_robust_chain.invoke({"question": question1}))

# 场景2：完全不相关
# retrieval_step (CohereRerank 返回 []) -> docs 列表为空 -> route() 选择 fallback_chain
print(full_robust_chain.invoke({"question": question2}))