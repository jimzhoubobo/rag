## ValueError: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6 in order to
## 问题解决过程
根本原因：
PyTorch版本与sentence-transformers库存在兼容性问题
版本依赖冲突导致无法加载嵌入模型
解决方案：
降级安装了兼容的sentence-transformers==2.2.2版本
解决了相关的依赖冲突，包括huggingface_hub和transformers版本问题,
pip install huggingface_hub==0.20.3
pip install transformers==4.35.2
在代码中增加了异常处理，确保即使主模型加载失败也能使用备用模型
技术实现：
使用HuggingFaceBgeEmbeddings专门处理中文BGE模型
添加了备用的HuggingFaceEmbeddings作为回退方案
配置了适当的模型参数以避免PyTorch版本问题
最终效果：
应用已成功启动，可在http://localhost:8502访问
所有功能正常运行，包括文档加载、向量存储和问答功能
您的中医理疗智能问答系统现在已经可以正常使用了！系统能够：
加载PDF和TXT格式的中医文档
使用中文优化的BGE嵌入模型处理文本
通过DeepSeek API提供智能问答
在界面中显示安全免责声明

## Error code: 400 - {'error': {'message': 'Failed to deserialize the JSON body into the target type: prompt: invalid type: sequence, expected a string at line 1 column 34'
太好了！DeepSeek API测试成功了。这表明我们使用ChatOpenAI方式调用DeepSeek API是正确的。
现在让我总结一下已完成的工作：
已完成的更新
1. 更新了LLM调用方式
将原来的OpenAI类改为使用ChatOpenAI类来调用DeepSeek API
使用了您提供的正确配置方式：
2.   llm = ChatOpenAI(
      model="deepseek-chat",
      temperature=0.7,
      max_tokens=2000,
      api_key=api_key,
      base_url="https://api.deepseek.com/v1"
  )
