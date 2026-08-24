# 第三節 向量資料庫

## 一、向量資料庫的作用

在前面我們學習瞭如何使用嵌入模型將文字、影象等非結構化資料轉換為高維向量。這些向量是 RAG 系統能夠進行語義理解的基礎。然而，當向量數量從幾百個增長到數百萬甚至數十億時，一個核心問題隨之而來：**如何快速、準確地從海量向量中找到與使用者查詢最相似的那幾個？**

### 1.1 向量資料庫主要功能

向量資料庫的核心價值在於其高效處理海量高維向量的能力。其主要功能可以概括為以下幾點：

- **高效的相似性搜尋**：這是向量資料庫最重要的功能。它利用專門的索引技術（如 HNSW, IVF），能夠在數十億級別的向量中實現毫秒級的近似最近鄰（ANN）查詢，快速找到與給定查詢最相似的資料。

- **高維資料儲存與管理**：專門為儲存高維向量（通常維度成百上千）而最佳化，支援對向量資料進行增、刪、改、查等基本操作。

- **豐富的查詢能力**：除了基本的相似性搜尋，還支援按標量欄位過濾查詢（例如，在搜尋相似圖片的同時，指定`年份 > 2023`）、範圍查詢和聚類分析等，滿足複雜業務需求。

- **可擴充套件與高可用**：現代向量資料庫通常採用分散式架構，具備良好的水平擴充套件能力和容錯性，能夠透過增加節點來應對資料量的增長，並確保服務的穩定可靠。

- **資料與模型生態整合**：與主流的 AI 框架（如 LangChain, LlamaIndex）和機器學習工作流無縫整合，簡化了從模型訓練到向量檢索的應用開發流程。

### 1.2 向量資料庫 vs 傳統資料庫

傳統的資料庫（如 MySQL）擅長處理結構化資料的精確匹配查詢（例如，`WHERE age = 25`），但它們並非為處理高維向量的相似性搜尋而設計的。在龐大的向量集合中進行暴力、線性的相似度計算，其計算成本和時間延遲無法接受。**向量資料庫 (Vector Database)** 很好的解決了這一問題，它是一種專門設計用於高效儲存、管理和查詢高維向量的資料庫系統。在 RAG 流程中，它扮演著“知識庫”的角色，是連線資料與大語言模型的關鍵橋樑。

向量資料庫與傳統資料庫的主要差異如下：

| **維度** | **向量資料庫** | **傳統資料庫 (RDBMS)** |
| :--- | :--- | :--- |
| **核心資料型別** | 高維向量 (Embeddings) | 結構化資料 (文字、數字、日期) |
| **查詢方式** | **相似性搜尋** (ANN) | **精確匹配** |
| **索引機制** | HNSW, IVF, LSH 等 ANN 索引 | B-Tree, Hash Index |
| **主要應用場景** | AI 應用、RAG、推薦系統、影象/語音識別 | 業務系統 (ERP, CRM)、金融交易、資料包表 |
| **資料規模** | 輕鬆應對千億級向量 | 通常在千萬到億級行資料，更大規模需複雜分庫分表 |
| **效能特點** | 高維資料檢索效能極高，計算密集型 | 結構化資料查詢快，高維資料查詢效能呈指數級下降 |
| **一致性** | 通常為最終一致性 | 強一致性 (ACID 事務) |

向量資料庫和傳統資料庫並非相互替代的關係，而是**互補關係**。在構建現代 AI 應用時，通常會將兩者結合使用：利用傳統資料庫儲存業務後設資料和結構化資訊，而向量資料庫則專門負責處理和檢索由 AI 模型產生的海量向量資料。

## 二、工作原理

向量資料庫的核心是高效處理高維向量的相似性搜尋。向量是一組有序的數值，可以表示文字、影象、音訊等複雜資料的特徵或屬性。在 RAG 系統中，向量一般透過嵌入模型將原始資料轉換為高維向量表示，比如上一節的圖文示例。向量資料庫通常採用四層架構，透過儲存層、索引層、查詢層和服務層的協同工作來實現高效相似性搜尋，其中儲存層負責儲存向量資料和後設資料，最佳化儲存效率並支援分散式儲存；索引層維護索引演算法（HNSW、LSH、PQ等），負責索引的建立與最佳化，並支援索引調整；查詢層處理查詢請求，支援混合查詢並實現查詢最佳化；服務層管理客戶端連線，提供監控和日誌能力，並實現安全管理。

主要技術手段包括：
- **基於樹的方法**：如 Annoy 使用的隨機投影樹，透過樹形結構實現對數複雜度的搜尋
- **基於雜湊的方法**：如 LSH（區域性敏感雜湊），透過雜湊函式將相似向量對映到同一“桶”
- **基於圖的方法**：如 HNSW（分層可導航小世界圖），透過多層鄰近圖結構實現快速搜尋
- **基於量化的方法**：如 Faiss 的 IVF 和 PQ，透過聚類和量化壓縮向量

## 三、主流向量資料庫介紹

![向量資料庫分類圖](./images/3_3_1.webp)

當前主流的向量資料庫產品包括：

[ **Pinecone** ](https://www.pinecone.io/)是一款完全託管的向量資料庫服務，採用Serverless架構設計。它提供儲存計算分離、自動擴充套件和負載均衡等企業級特性，並保證99.95%的SLA。Pinecone支援多種語言SDK，提供極高可用性和低延遲搜尋（<100ms），特別適合企業級生產環境、高併發場景和大規模部署。

[ **Milvus** ](https://github.com/milvus-io/milvus)是一款開源的分散式向量資料庫，採用分散式架構設計，支援GPU加速和多種索引演算法。它能夠處理億級向量檢索，提供高效能GPU加速和完善的生態系統。Milvus特別適合大規模部署、高效能要求的場景，以及需要自定義開發的開源專案。

[ **Qdrant** ](https://github.com/qdrant/qdrant)是一款高效能的開源向量資料庫，採用Rust開發，支援二進位制量化技術。它提供多種索引策略和向量混合搜尋功能，能夠實現極高的效能（RPS>4000）和低延遲搜尋。Qdrant特別適合效能敏感應用、高併發場景以及中小規模部署。

[ **Weaviate** ](https://github.com/weaviate/weaviate)是一款支援GraphQL的AI整合向量資料庫，提供20+AI模組和多模態支援。它採用GraphQL API設計，支援RAG最佳化，特別適合AI開發、多模態處理和快速開發場景。Weaviate具有活躍的社群支援和易於整合的特點。

[ **Chroma** ](https://github.com/chroma-core/chroma)是一款輕量級的開源向量資料庫，採用本地優先設計，無依賴。它提供零配置安裝、本地執行和低資源消耗等特性，特別適合原型開發、教育培訓和小規模應用。Chroma的部署簡單，適合快速原型開發。

**選擇建議**：
-   **新手入門/小型專案**：從 `ChromaDB` 或 `FAISS` 開始是最佳選擇。它們與 LangChain/LlamaIndex 緊密整合，幾行程式碼就能執行，且能滿足基本的儲存和檢索需求。
-   **生產環境/大規模應用**：當資料量超過百萬級，或需要高併發、實時更新、複雜後設資料過濾時，應考慮更專業的解決方案，如 `Milvus`、`Weaviate` 或雲服務 `Pinecone`。

## 四、本地向量儲存：以 FAISS 為例

FAISS (Facebook AI Similarity Search) 是一個由 Facebook AI Research 開發的高效能庫，專門用於高效的相似性搜尋和密集向量聚類。當與 LangChain 結合使用時，它可以作為一個強大的本地向量儲存方案，非常適合快速原型設計和中小型應用。

與 ChromaDB 等資料庫不同，FAISS 本質上是一個演算法庫，它將索引直接儲存為本地檔案（一個 `.faiss` 索引檔案和一個 `.pkl` 對映檔案），而非執行一個資料庫服務。這種方式輕量且高效。

### 4.1 環境準備

在開始之前，請確保已安裝所有必需的庫：

> 當前requirements.txt安裝的 `faiss-cpu` 是 CPU 版本。如果你的機器有 GPU，可以安裝 `faiss-gpu` 以獲得更好的效能。

### 4.2 基礎示例(FAISS)

下面的程式碼演示了使用 LangChain 和 FAISS 完成一個完整的“建立 -> 儲存 -> 載入 -> 查詢”流程。

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 示例文字和嵌入模型
texts = [
    "張三是法外狂徒",
    "FAISS是一個用於高效相似性搜尋和密集向量聚類的庫。",
    "LangChain是一個用於開發由語言模型驅動的應用程式的框架。"
]
docs = [Document(page_content=t) for t in texts]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 建立向量儲存並儲存到本地
vectorstore = FAISS.from_documents(docs, embeddings)

local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")

# 3. 載入索引並執行查詢
# 載入時需指定相同的嵌入模型，並允許反序列化
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 相似性搜尋
query = "FAISS是做什麼的？"
results = loaded_vectorstore.similarity_search(query, k=1)

print(f"\n查詢: '{query}'")
print("相似度最高的文件:")
for doc in results:
    print(f"- {doc.page_content}")
```
**執行結果與解讀**：

當你執行上述指令碼時，會看到類似以下的輸出：
```bash
FAISS index has been saved to ./faiss_index_store

查詢: 'FAISS是做什麼的？'
相似度最高的文件:
- FAISS是一個用於高效相似性搜尋和密集向量聚類的庫。
```

**索引建立實現細節**：
透過深入 LangChain 原始碼，可以發現索引建立是一個分層、解耦的過程，主要涉及以下幾個方法的巢狀呼叫：

1.  **`from_documents` (封裝層)**:
    *   這是我們直接呼叫的方法。它的職責很簡單：從輸入的 `Document` 物件列表中提取出純文字內容 (`page_content`) 和後設資料 (`metadata`)。
    *   然後，它將這些提取出的資訊傳遞給核心的 `from_texts` 方法。

2.  **`from_texts` (向量化入口)**:
    *   這個方法是面向使用者的入口。它接收文字列表，並執行關鍵的第一步：呼叫 `embedding.embed_documents(texts)`，將所有文字批次轉換為向量。
    *   完成向量化後，它並不直接處理索引構建，而是將生成的向量和其他所有資訊（文字、後設資料等）傳遞給一個內部的輔助方法 `__from`。

3.  **`__from` (構建索引框架)**:
    *   一個內部方法，負責搭建 FAISS 向量儲存的“空框架”。
    *   它會根據指定的距離策略（預設為 L2 歐氏距離）初始化一個空的 FAISS 索引結構（如 `faiss.IndexFlatL2`）。
    *   同時，它也準備好了用於儲存文件原文的 `docstore` 和用於連線 FAISS 索引與文件的 `index_to_docstore_id` 對映。
    *   最後，它呼叫另一個內部方法 `__add` 來完成資料的填充。

4.  **`__add` (填充資料)**:
    *   真正執行資料新增操作的核心。它接收到向量、文字和後設資料後，執行以下關鍵操作：
        *   **新增向量**: 將向量列表轉換為 FAISS 需要的 `numpy` 陣列，並呼叫 `self.index.add(vector)` 將其批次新增到 FAISS 索引中。
        *   **儲存文件**: 將文字和後設資料打包成 `Document` 物件，存入 `docstore`。
        *   **建立對映**: 更新 `index_to_docstore_id` 字典，建立起 FAISS 內部的整數 ID（如 0, 1, 2...）到我們文件唯一 ID 的對映關係。


## 練習

1. LlamaIndex預設會將資料儲存為透明可讀的JSON格式，執行[03_llamaindex_vector.py](https://github.com/datawhalechina/all-in-rag/blob/main/code/C3/03_llamaindex_vector.py)檔案，檢視儲存的json檔案內容。
2. 新建一個程式碼檔案實現對LlamaIndex儲存資料的載入和相似性搜尋。