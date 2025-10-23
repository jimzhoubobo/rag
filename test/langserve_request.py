import requests

# 构造请求
response = requests.post(
    "http://localhost:8000/joke/invoke", # 注意URL中包含"/invoke"
    json={'input': {'topic': '程序员'}} # 输入数据需符合链的输入格式
)
# 打印响应
print(response.json())
# 通常，模型的输出会在 `output` 字段或更深的嵌套结构中，你需要根据实际响应结构解析。