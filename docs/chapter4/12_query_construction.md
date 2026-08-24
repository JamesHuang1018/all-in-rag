# 第二節 查詢構建

在前面的章節中，我們探討了如何透過向量嵌入和相似度搜尋來從非結構化資料中檢索資訊。然而，在實際應用中，我們常常需要處理更加複雜和多樣化的資料，包括結構化資料（如SQL資料庫）、半結構化資料（如帶有後設資料的文件）以及圖資料。使用者的查詢也可能不僅僅是簡單的語義匹配，而是包含複雜的過濾條件、聚合操作或關係查詢。

**查詢構建（Query Construction）**[^1] 正是應對這一挑戰的關鍵技術。它利用大語言模型（LLM）的強大理解能力，將使用者的自然語言查詢“翻譯”成針對特定資料來源的結構化查詢語言或帶有過濾條件的請求。這使得RAG系統能夠無縫地連線和利用各種型別的資料，從而極大地擴充套件了其應用場景和能力。

下圖展示了查詢構建在一個高階RAG流程中所處的位置：

![Advanced RAG Pipeline](./images/4_2_1.webp)

## 一、文字到後設資料過濾器

在構建向量索引時，常常會為文件塊（Chunks）附加後設資料（Metadata），例如文件來源、釋出日期、作者、章節、類別等。這些後設資料為我們提供了在語義搜尋之外進行精確過濾的可能。

**自查詢檢索器（Self-Query Retriever）** 是LangChain中實現這一功能的核心元件。它的工作流程如下：

1.  **定義後設資料結構**：首先，需要向LLM清晰地描述文件內容和每個後設資料欄位的含義及型別。
2.  **查詢解析**：當使用者輸入一個自然語言查詢時，自查詢檢索器會呼叫LLM，將查詢分解為兩部分：
    *   **查詢字串（Query String）**：用於進行語義搜尋的部分。
    *   **後設資料過濾器（Metadata Filter）**：從查詢中提取出的結構化過濾條件。
3.  **執行查詢**：檢索器將解析出的查詢字串和後設資料過濾器傳送給向量資料庫，執行一次同時包含語義搜尋和後設資料過濾的查詢。

例如，對於查詢“關於2022年釋出的機器學習的論文”，自查詢檢索器會將其解析為：
*   **查詢字串**: "機器學習的論文"
*   **後設資料過濾器**: `year == 2022`

### 程式碼示例

接下來以B站影片為例來看看如何使用`SelfQueryRetriever`。

```python
import os
from langchain_deepseek import ChatDeepSeek 
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO)

# 1. 初始化影片資料
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []
try:
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    for doc in docs:
        original = doc.metadata
        
        # 提取基本後設資料欄位
        metadata = {
            'title': original.get('title', '未知標題'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        
        doc.metadata = metadata
        bili.append(doc)
        
except Exception as e:
    print(f"載入BiliBili影片失敗: {str(e)}")

if not bili:
    print("沒有成功載入任何影片，程式退出")
    exit()

# 2. 建立向量儲存
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(bili, embed_model)
```

在上面的程式碼中，首先使用 `BiliBiliLoader` 載入了幾個B站影片的文件和後設資料。需要注意的是，由於 `BiliBiliLoader` 返回的原始後設資料結構較為複雜（例如，作者和觀看數資訊巢狀在其他字典中），所以進行了一些預處理工作：遍歷每個文件，手動提取需要的欄位（如`title`, `author`, `view_count`, `length`），並構建一個乾淨、扁平化的新 `metadata` 字典。這個過程確保了後續的自查詢檢索器能夠直接、可靠地訪問這些欄位。最後，將處理好的文件和後設資料存入 `Chroma` 向量資料庫中，為下一步的查詢構建做好準備。

```python
# 3. 配置後設資料欄位資訊
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="影片標題（字串）",
        type="string", 
    ),
    AttributeInfo(
        name="author",
        description="影片作者（字串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="影片觀看次數（整數）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="影片長度，以秒為單位的整數",
        type="integer"
    )
]

# 4. 建立自查詢檢索器
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
    )

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="記錄影片標題、作者、觀看次數等資訊的影片後設資料",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)

# 5. 執行查詢示例
queries = [
    "時間最短的影片",
    "時長大於600秒的影片"
]

for query in queries:
    print(f"\n--- 查詢: '{query}' ---")
    results = retriever.invoke(query)
    if results:
        for doc in results:
            title = doc.metadata.get('title', '未知標題')
            author = doc.metadata.get('author', '未知作者')
            view_count = doc.metadata.get('view_count', '未知')
            length = doc.metadata.get('length', '未知')
            print(f"標題: {title}")
            print(f"作者: {author}")
            print(f"觀看次數: {view_count}")
            print(f"時長: {length}秒")
            print("="*50)
    else:
        print("未找到匹配的影片")
```

這部分程式碼是實現自查詢檢索的核心。主要分為三個步驟：

1.  **配置後設資料欄位 (`metadata_field_info`)** ：這是與LLM溝通的藍圖。透過 `AttributeInfo` 為每個後設資料欄位定義名稱、型別和一份清晰的自然語言 `description`。LLM 將依賴這份描述來理解如何處理使用者的查詢，例如，它會根據“影片長度（整數）”的描述來解析關於“時長”的過濾和排序請求。因此，一份準確、無歧義的描述很重要。

2.  **建立自查詢檢索器 (`SelfQueryRetriever.from_llm`)** ：`from_llm` 方法在底層執行了兩個核心操作：
    *   **載入查詢構造器**：利用傳入的 `llm`、`document_contents` 和 `metadata_field_info`，建立一個專門的“查詢構造鏈”。這個鏈的核心職責是將使用者的自然語言查詢（如“時長大於600秒的影片”）轉換為一個通用的、結構化的查詢物件。
    *   **獲取內建翻譯器**：接著，檢查使用的向量資料庫（這裡是 `Chroma`），併為其匹配一個內建的“翻譯器”。這個翻譯器負責將上一步生成的通用查詢物件，翻譯成 `Chroma` 資料庫能夠原生理解和執行的過濾語法。

3.  **執行查詢 (`retriever.invoke`)** ：最後，用自然語言發起呼叫。檢索器內部會依次執行“構造”和“翻譯”兩個步驟，最終向 `Chroma` 發起一個同時包含語義搜尋和精確後設資料過濾的複合查詢，從而返回最相關的結果。

> **提示**：在程式碼中可以看到 `temperature` 引數被設定為 `0`。這個值是用於控制模型輸出的隨機性。值越高（如 0.8），輸出越隨機、越有創意；值越低，輸出越確定、越集中。設定為 `0` 可以讓模型的輸出變得完全確定，即對於相同的輸入，總是生成完全相同的輸出。在自查詢這種需要精確地將自然語言轉換為結構化查詢的場景下，可以確保轉換結果的穩定和可復現。

**輸出結果：**

```bash
--- 查詢: '時間最短的影片' ---
INFO:httpx:HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:langchain.retrievers.self_query.base:Generated Query: query=' ' filter=None limit=1
標題: 《吳恩達 x OpenAI Prompt課程》【專業翻譯，配套程式碼筆記】02.Prompt 的構建原則
作者: 二次元的Datawhale
觀看次數: 18788
時長: 1063秒
==================================================

--- 查詢: '時長大於600秒的影片' ---
INFO:httpx:HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:langchain.retrievers.self_query.base:Generated Query: query=' ' filter=Comparison(comparator=<Comparator.GT: 'gt'>, attribute='length', value=600) limit=None
WARNING:chromadb.segment.impl.vector.local_hnsw:Number of requested results 4 is greater than number of elements in index 3, updating n_results = 3
標題: 《吳恩達 x OpenAI Prompt課程》【專業翻譯，配套程式碼筆記】03.Prompt如何迭代最佳化
作者: 二次元的Datawhale
觀看次數: 7090
時長: 806秒
==================================================
標題: 《吳恩達 x OpenAI Prompt課程》【專業翻譯，配套程式碼筆記】02.Prompt 的構建原則
作者: 二次元的Datawhale
觀看次數: 18788
時長: 1063秒
```

## 二、文字到Cypher

除了處理扁平化的後設資料，查詢構建技術還能應用於更復雜的資料結構，如圖資料庫。

### 2.1 什麼是 Cypher？

Cypher 是圖資料庫（如 Neo4j）中最常用的查詢語言，其地位類似於 SQL 之於關聯式資料庫。它採用一種直觀的方式來匹配圖中的模式和關係，例如 `(:Person {name:"Tomaz"})-[:LIVES_IN]->(:Country {name:"Slovenia"})` 描述了一個人和一個國家以及他們之間的“居住在”關係。

### 2.2 “文字到Cypher”的原理

與“文字到後設資料過濾器”類似，“文字到Cypher”技術利用大語言模型（LLM）將使用者的自然語言問題直接翻譯成一句精準的 Cypher 查詢語句。LangChain 提供了相應的工具鏈（如 `GraphCypherQAChain`），其工作流程通常是：
1.  接收使用者的自然語言問題。
2.  LLM 根據預先提供的圖譜模式（Schema），將問題轉換為 Cypher 查詢。
3.  在圖資料庫上執行該查詢，獲取精確的結構化資料。
4.  (可選)將查詢結果再次交由 LLM，生成通順的自然語言答案。

由於生成有效的 Cypher 查詢是一項複雜的任務，通常使用效能較強的 LLM 來確保轉換的準確性。透過這種方式，使用者可以用最自然的方式與高度結構化的圖資料進行互動，極大地降低了資料查詢的門檻。

## 思考

- 為什麼本節的程式碼中查詢“時間最短的影片”時，得到的結果是錯誤的？

## 參考文獻

[^1]: [*LangChain Blog: Query Construction*](https://blog.langchain.ac.cn/query-construction/)
