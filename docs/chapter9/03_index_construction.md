# 第三節 Milvus索引構建

在圖RAG系統中，索引構建是連線圖資料和向量檢索的關鍵環節。本節介紹如何將圖資料轉換為可檢索的向量索引。

在第三章中，我們已經詳細介紹了Milvus的基本概念、部署方式和基礎操作。本節將在此基礎上，專門針對圖RAG場景進行深度應用。如果你對Milvus還不熟悉，建議先閱讀[Milvus介紹及多模態檢索實踐](https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter3/09_milvus.md)。

> [本節完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C9/rag_modules/milvus_index_construction.py)

## 一、索引構建概述

### 1.1 索引構建流程

圖RAG的索引構建需要將從圖資料庫構建的結構化文件轉換為向量表示，並儲存到向量資料庫中：

```mermaid
flowchart LR
    A[圖資料庫] --> B[文件構建]
    B --> C[文件分塊]
    C --> D[向量化]
    D --> E[Milvus索引]
    
    style A fill:#e1f5fe
    style E fill:#e8f5e8
```

### 1.2 核心元件

- **文件構建器**：從圖資料構建結構化文件
- **分塊處理器**：智慧分塊策略
- **向量化模型**：文字轉向量
- **Milvus索引**：高效能向量儲存和檢索

## 二、Milvus索引構建實現

### 2.1 索引構建器核心架構

```python
class MilvusIndexConstructionModule:
    """Milvus索引構建模組 - 負責向量化和Milvus索引構建"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 19530,
                 collection_name: str = "cooking_knowledge",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name

        self.client = None
        self.embeddings = None
        self.collection_created = False

        self._setup_client()
        self._setup_embeddings()
```

**程式碼解讀**：
- **模組化設計**：將Milvus操作封裝為獨立模組，便於複用和維護
- **配置靈活性**：支援自定義Milvus連線引數和嵌入模型
- **中文最佳化**：預設使用`BAAI/bge-small-zh-v1.5`，專門針對中文文字最佳化
- **延遲初始化**：在建構函式中設定連線，避免啟動時的阻塞

### 2.2 向量化處理

```python
def _vectorize_documents(self, documents: List[Document]) -> Tuple[List[List[float]], List[Dict]]:
    """文件向量化處理"""
    vectors = []
    metadatas = []
    
    for i, doc in enumerate(documents):
        try:
            # 向量化文件內容
            vector = self.embedding_model.embed_query(doc.page_content)
            vectors.append(vector)
            
            # 準備後設資料
            metadata = {
                "id": i,
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "parent_id": doc.metadata.get("parent_id", ""),
                # ... 其他後設資料
            }
            metadatas.append(metadata)
            
        except Exception as e:
            logger.error(f"文件 {i} 向量化失敗: {e}")
            continue
    
    return vectors, metadatas
```

### 2.3 圖RAG專用集合Schema設計

```python
def _create_collection_schema(self):
    """建立集合schema"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
        FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="difficulty", dtype=DataType.INT64),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="中式烹飪知識圖譜向量集合"
    )
    return schema
```

**Schema設計亮點**：
- **圖資料特化**：專門為烹飪知識圖譜設計的欄位結構
- **豐富後設資料**：包含菜譜名稱、節點型別、菜系、難度等圖譜特有資訊
- **長度最佳化**：根據實際資料特點設定合理的欄位長度限制
- **檢索友好**：所有關鍵欄位都可用於過濾和檢索條件

## 三、索引最佳化策略

### 3.1 批次插入最佳化

```python
def _batch_insert(self, vectors: List[List[float]], metadatas: List[Dict]):
    """批次插入最佳化"""
    batch_size = self.config.batch_size
    collection_name = self.config.milvus_collection_name
    
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        
        # 準備插入資料
        insert_data = [
            [meta["id"] for meta in batch_metadatas],           # id
            batch_vectors,                                       # vector
            [meta["content"] for meta in batch_metadatas],      # content
            [meta["source"] for meta in batch_metadatas],       # source
            [meta["chunk_id"] for meta in batch_metadatas],     # chunk_id
            [meta["parent_id"] for meta in batch_metadatas],    # parent_id
        ]
        
        # 執行插入
        self.milvus_client.insert(collection_name, insert_data)
        logger.info(f"批次 {i//batch_size + 1} 插入完成，數量: {len(batch_vectors)}")
```

### 3.2 索引建立

```python
def _create_index(self):
    """建立向量索引"""
    collection_name = self.config.milvus_collection_name
    
    # 索引引數
    index_params = {
        "metric_type": "COSINE",    # 餘弦相似度
        "index_type": "IVF_FLAT",   # 索引型別
        "params": {"nlist": 1024}   # 索引引數
    }
    
    # 建立索引
    self.milvus_client.create_index(
        collection_name=collection_name,
        field_name="vector",
        index_params=index_params
    )
    
    # 載入集合到記憶體
    self.milvus_client.load_collection(collection_name)
    
    logger.info("向量索引建立完成")
```

## 四、索引構建流程

### 4.1 核心向量構建流程

```python
def build_vector_index(self, chunks: List[Document]) -> bool:
    """構建向量索引"""
    logger.info(f"正在構建Milvus向量索引，文件數量: {len(chunks)}...")

    try:
        # 1. 建立集合（如果schema不相容則強制重新建立）
        if not self.create_collection(force_recreate=True):
            return False

        # 2. 準備資料
        logger.info("正在生成向量embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)

        # 3. 準備插入資料
        entities = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            entity = {
                "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                "vector": vector,
                "text": self._safe_truncate(chunk.page_content, 15000),
                "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                # ... 更多欄位
            }
            entities.append(entity)

        # 4. 批次插入資料
        batch_size = 100
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            self.client.insert(collection_name=self.collection_name, data=batch)
```

**關鍵技術點解讀**：

1. **強制重建策略**：`force_recreate=True`確保Schema一致性，避免欄位不匹配錯誤

2. **批次向量化**：一次性處理所有文件的向量化，提高效率
   ```python
   texts = [chunk.page_content for chunk in chunks]
   vectors = self.embeddings.embed_documents(texts)  # 批次處理
   ```

3. **安全截斷機制**：`_safe_truncate`方法防止欄位長度超限
   ```python
   def _safe_truncate(self, text: str, max_length: int) -> str:
       if text is None:
           return ""
       return str(text)[:max_length]
   ```

4. **圖資料後設資料保留**：完整保留圖譜中的結構化資訊，支援後續的複合檢索

### 4.2 索引驗證

```python
def verify_index(self) -> bool:
    """驗證索引構建結果"""
    try:
        collection_name = self.config.milvus_collection_name
        
        # 檢查集合狀態
        collection_info = self.milvus_client.describe_collection(collection_name)
        logger.info(f"集合資訊: {collection_info}")
        
        # 檢查資料量
        count = self.milvus_client.query(
            collection_name=collection_name,
            expr="",
            output_fields=["count(*)"]
        )
        logger.info(f"索引中文件數量: {count}")
        
        # 簡單檢索測試
        test_results = self.milvus_client.search(
            collection_name=collection_name,
            data=[[0.1] * self.config.embedding_dim],  # 測試向量
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=1
        )
        
        logger.info("索引驗證透過")
        return True

    except Exception as e:
        logger.error(f"索引驗證失敗: {e}")
        return False
```

## 五、為什麼從FAISS切換到Milvus？

在第八章中，使用的是FAISS作為向量儲存方案。雖然FAISS在研究和原型開發中表現出色，但在生產環境和複雜應用場景下，Milvus提供了更多優勢：

**FAISS的侷限性**：
- **純庫模式**：FAISS是一個向量搜尋庫，缺乏資料庫的完整功能
- **無持久化**：需要手動管理資料持久化和備份
- **單機限制**：難以實現分散式部署和水平擴充套件
- **後設資料支援有限**：無法高效儲存和查詢複雜的結構化後設資料
- **併發效能**：在高併發場景下效能受限

**Milvus的優勢**：
- **完整資料庫功能**：提供CRUD操作、事務支援、資料一致性保證
- **雲原生架構**：支援分散式部署、自動擴縮容、高可用性
- **豐富的後設資料支援**：支援複雜Schema設計，適合圖RAG的多維度資料
- **生產級特性**：監控、日誌、備份恢復等企業級功能
