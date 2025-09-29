class ListUtils:
    """
    列表工具类
    """
    @staticmethod
    def chunk_list(lst, chunk_size):
        """
        将列表切分为指定大小的子列表，性能最优的实现

        Args:
            lst (list): 要切分的列表
            chunk_size (int): 子列表的大小

        Returns:
            list: 切分后的子列表 
        """
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]