### 1. 模块级单例模式
```commandline
_embedding_instance = None

def init_embedding():
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance
    
    # 模型加载逻辑
    _embedding_instance = SentenceTransformerEmbeddings(model_name=model_path)
    return _embedding_instance
```
### 2. 装饰器缓存模式
maxsize=1 的含义
缓存大小限制：表示缓存最多只能存储1个函数调用的结果
LRU淘汰策略：当缓存满时，会淘汰最近最少使用的项

```
from functools import lru_cache

@lru_cache(maxsize=1)
def init_embedding():
    # 只会执行一次，结果被缓存
    return SentenceTransformerEmbeddings(model_name=model_path)

# 第一次调用：执行函数，结果缓存
embedding1 = init_embedding()

# 第二次调用：直接返回缓存结果，不执行函数
embedding2 = init_embedding()

# 两个变量指向同一个对象
assert embedding1 is embedding2

```
### 3. 类级单例模式   
类级单例模式中 cls 参数说明
1. cls 参数的含义
类对象引用：cls 是类方法的第一个参数，指向调用该方法的类
自动传递：由Python解释器自动传递，调用方无需显式传入
类似 self：与实例方法中的 self 类似，但指向类而非实例
```commandline
class EmbeddingManager:
    _instance = None
    
    @classmethod
    def get_embedding(cls):
        if cls._instance is None:
            cls._instance = SentenceTransformerEmbeddings(model_name=model_path)
        return cls._instance

```
3. 使用要点
通过类名调用：ClassName.method_name()
无需传参：cls 由Python自动传入
访问类属性：通过 cls.attribute_name 访问类属性
创建实例：可在方法内通过 cls() 创建类实例
这种模式简化了单例模式的实现，调用方只需正常调用类方法即可。
### 4. 依赖注入模式
```commandline
class VectorStoreBuilder:
    def __init__(self):
        self.embedding = self._init_embedding()
    
    def _init_embedding(self):
        # 模型初始化逻辑
        return SentenceTransformerEmbeddings(model_name=model_path)

```
