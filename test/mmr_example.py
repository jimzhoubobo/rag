import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class MMR:
    def __init__(self, lambda_param: float = 0.5):
        """
        MMR 算法实现

        Args:
            lambda_param: 权衡参数，范围 [0, 1]
                        - 接近1: 更看重相关性
                        - 接近0: 更看重多样性
        """
        self.lambda_param = lambda_param

    def compute_mmr_scores(self,
                           query_vector: np.ndarray,
                           doc_vectors: np.ndarray,
                           selected_indices: List[int]) -> np.ndarray:
        """
        计算所有候选文档的 MMR 分数
        """
        # 计算与查询的相关性
        relevance_scores = cosine_similarity(query_vector, doc_vectors)[0]

        # 如果没有已选文档，只返回相关性分数
        if not selected_indices:
            return relevance_scores

        # 计算与已选文档的最大相似度
        selected_vectors = doc_vectors[selected_indices]
        max_similarity = np.zeros(len(doc_vectors))

        for i in range(len(doc_vectors)):
            if i not in selected_indices:  # 只计算未选中文档
                similarities = cosine_similarity([doc_vectors[i]], selected_vectors)[0]
                max_similarity[i] = np.max(similarities) if len(similarities) > 0 else 0

        # 计算 MMR 分数
        mmr_scores = (self.lambda_param * relevance_scores -
                      (1 - self.lambda_param) * max_similarity)

        return mmr_scores

    def select_documents(self,
                         query_vector: np.ndarray,
                         doc_vectors: np.ndarray,
                         top_k: int = 5) -> List[int]:
        """
        使用 MMR 选择文档

        Args:
            query_vector: 查询向量
            doc_vectors: 所有文档向量
            top_k: 要选择的文档数量

        Returns:
            选择的文档索引列表
        """
        selected_indices = []
        remaining_indices = list(range(len(doc_vectors)))

        for _ in range(min(top_k, len(doc_vectors))):
            mmr_scores = self.compute_mmr_scores(query_vector, doc_vectors, selected_indices)

            # 从未选中的文档中选择最高 MMR 分数的文档
            candidate_scores = [(i, mmr_scores[i]) for i in remaining_indices
                                if i not in selected_indices]

            if not candidate_scores:
                break

            best_idx, best_score = max(candidate_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return selected_indices


# 示例使用
def demo_mmr():
    """演示 MMR 的不同 lambda 参数效果"""

    # 模拟数据：10个文档，每个文档5维向量
    np.random.seed(42)
    n_docs = 10
    n_features = 5

    # 生成文档向量（模拟文档嵌入）
    doc_vectors = np.random.randn(n_docs, n_features)

    # 生成查询向量
    query_vector = np.random.randn(1, n_features)

    print("文档与查询的原始相关性分数:")
    relevance_scores = cosine_similarity(query_vector, doc_vectors)[0]
    for i, score in enumerate(relevance_scores):
        print(f"文档 {i}: {score:.3f}")

    print("\n" + "=" * 50)

    # 测试不同的 lambda 参数
    lambda_values = [0.2, 0.5, 0.8]

    for lambda_val in lambda_values:
        print(f"\nλ = {lambda_val} 的结果:")
        mmr = MMR(lambda_param=lambda_val)
        selected_indices = mmr.select_documents(query_vector, doc_vectors, top_k=5)

        print(f"选择的文档索引: {selected_indices}")
        print("选择的文档相关性分数:")
        for idx in selected_indices:
            print(f"  文档 {idx}: {relevance_scores[idx]:.3f}")

        # 计算选择的文档之间的平均相似度（多样性指标）
        if len(selected_indices) > 1:
            selected_vectors = doc_vectors[selected_indices]
            similarity_matrix = cosine_similarity(selected_vectors)
            # 取上三角矩阵（不包括对角线）
            upper_tri = similarity_matrix[np.triu_indices(len(selected_indices), k=1)]
            avg_similarity = np.mean(upper_tri) if len(upper_tri) > 0 else 0
            print(f"选择文档间的平均相似度: {avg_similarity:.3f}")


if __name__ == "__main__":
    demo_mmr()