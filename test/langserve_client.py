from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/joke")
result = remote_chain.invoke({"topic": "科学家"})
print(result)