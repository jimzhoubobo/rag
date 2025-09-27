from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./bge-large-zh-v1.5')  # 使用本地路径
embeddings = model.encode(["这是一个测试句子。"], normalize_embeddings=True)
print(embeddings.shape)  # 应输出类似 (1, 1024)，表示1个句子，每个向量1024维