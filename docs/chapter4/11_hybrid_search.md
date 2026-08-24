# 第一節 混合檢索

混合檢索（Hybrid Search）是一種結合了 **稀疏向量（Sparse Vectors）** 和 **密集向量（Dense Vectors）** 優勢的先進搜尋技術。旨在同時利用稀疏向量的關鍵詞精確匹配能力和密集向量的語義理解能力，以克服單一向量檢索的侷限性，從而在各種搜尋場景下提供更準確、更魯棒的檢索結果。

在本節中，我們將首先分析這兩種核心向量的特性，然後討論它們如何融合，最後透過milvus實現混合檢索。

## 一、稀疏向量 vs 密集向量

為了更好地理解混合檢索，首先需要釐清兩種向量的本質區別。

### 1.1 稀疏向量

稀疏向量，也常被稱為“詞法向量”，是基於詞頻統計的傳統資訊檢索方法的數學表示。它通常是一個維度極高（與詞彙表大小相當）但絕大多數元素為零的向量。它採用精準的“詞袋”匹配模型，將文件視為一堆詞的集合，不考慮其順序和語法，其中向量的每一個維度都直接對應一個具體的詞，非零值則代表該詞在文件中的重要性（權重）。這類向量的經典權重計算方法是 TF-IDF。在資訊檢索領域，BM25 則是基於這種稀疏表示的成功且應用廣泛的排序演算法之一，其核心公式如下：

  $$ Score(Q, D) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})} $$

  其中：
  - $IDF(q_i)$: 查詢詞 $q_i$ 的逆文件頻率，用於衡量一個詞的普遍程度。越常見的詞，IDF值越低。
  - $f(q_i, D)$: 查詢詞 $q_i$ 在文件 $D$ 中的詞頻。
  - $|D|$: 文件 $D$ 的長度。
  - $avgdl$: 集合中所有文件的平均長度。
  - $k_1, b$: 可調節的超引數。 $k_1$ 用於控制詞頻飽和度（一個詞在文件中出現10次和100次，其重要性增長並非線性），  $b$ 用於控制文件長度歸一化的程度。

這種方法的優點是可解釋性極強（每個維度都代表一個確切的詞），無需訓練，能夠實現關鍵詞的精確匹配，對於專業術語和特定名詞的檢索效果好。主要缺點是無法理解語義，例如它無法識別“汽車”和“轎車”是同義詞，存在“詞彙鴻溝”。

### 1.2 密集向量

密集向量，也常被稱為“語義向量”，是透過深度學習模型學習到的資料（如文字、影象）的低維、稠密的浮點數表示。這些向量旨在將原始資料對映到一個連續的、充滿意義的“語義空間”中來捕捉“語義”或“概念”。在理想的語義空間中，向量之間的距離和方向代表了它們所表示概念之間的關係。一個經典的例子是 `vector('國王') - vector('男人') + vector('女人')` 的計算結果在向量空間中非常接近 `vector('女王')`，這表明模型學會了“性別”和“皇室”這兩個維度的抽象概念。它的代表包括 Word2Vec、GloVe、以及所有基於 Transformer 的模型（如 BERT、GPT）生成的嵌入（Embeddings）。

其主要優點是能夠理解同義詞、近義詞和上下文關係，泛化能力強，在語義搜尋任務中表現卓越。但缺點也同樣明顯：可解釋性差（向量中的每個維度通常沒有具體的物理意義），需要大量資料和算力進行模型訓練，且對於未登入詞（OOV）[^1]的處理相對困難。

> **OOV（Out-of-Vocabulary）未登入詞**：指在模型訓練時沒有出現在詞彙表中，但在實際使用時遇到的新詞彙。例如，如果模型訓練時詞彙表中沒有"ChatGPT"這個詞，那麼在實際應用中遇到它時就是OOV。傳統的稀疏向量方法（如BM25）對OOV詞彙會完全忽略，而現代的密集向量方法透過子詞分割（如BPE、WordPiece）可以更好地處理OOV問題。

### 1.3 例項對比

**稀疏向量表示:**

稀疏向量的核心思想是隻儲存非零值。例如，一個8維的向量 `[0, 0, 0, 5, 0, 0, 0, 9]`，其大部分元素都是零。用稀疏格式表示，可以極大地節約空間。常見的稀疏表示法有兩種：

1.  **字典 / 鍵值對 (Dictionary / Key-Value):**
    這種方式將非零元素的 `索引` (0-based) 作為鍵，`值` 作為值。上面的向量可以表示為：
    ```json
    // {索引: 值}
    {
      "3": 5,
      "7": 9
    }
    ```

2.  **座標列表 (Coordinate list - COO):**
    這種方式通常用一個元組 `(維度, [索引列表], [值列表])` 來表示。上面的向量可以表示為：
    ```
    (8, [3, 7], [5, 9])
    ```
    這種格式在 `SciPy` 等科學計算庫中非常常見。

假設在一個包含5萬個詞的詞彙表中，“西紅柿”在第88位，“炒”在第666位，“蛋”在第999位，它們的BM25權重分別是1.2、0.8、1.5。那麼它的稀疏表示（採用字典格式）就是：
```json
// {索引: 權重}
{
  "88": 1.2,
  "666": 0.8,
  "999": 1.5
}
```
如果採用座標列表（COO）格式，它會是這樣：
```
(50000, [88, 666, 999], [1.2, 0.8, 1.5])
```
這兩種格式都清晰地記錄了文件的關鍵資訊，但它們的侷限性也很明顯：如果我們搜尋“番茄炒雞蛋”，由於“番茄”和“西紅柿”是不同的詞條（索引不同），模型將無法理解它們的語義相似性。

**密集向量表示:**

與稀疏向量不同，密集向量的所有維度都有值，因此使用**陣列 `[]`** 來表示是最直接的方式。一個預訓練好的語義模型在讀取“西紅柿炒蛋”後，會輸出一個低維的密集向量：

```json
// 這是一個低維（比如1024維）的浮點數向量
// 向量的每個維度沒有直接的、可解釋的含義
[0.89, -0.12, 0.77, ..., -0.45]
```

這個向量本身難以解讀，但它在語義空間中的位置可能與“番茄雞蛋麵”、“洋蔥炒雞蛋”等菜餚的向量非常接近，因為模型理解了它們共享“雞蛋類菜餚”、“家常菜”、“酸甜口味”等核心概念。因此，當我們搜尋“蛋白質豐富的家常菜”時，即使查詢中沒有出現任何原文關鍵詞，密集向量也很有可能成功匹配到這份菜譜。

## 二、混合檢索

透過上文可以看出稀疏向量和密集向量各有千秋，那麼將它們結合起來，實現優勢互補，就成了一個不錯的選擇。混合檢索便是基於這個思路，透過結合多種搜尋演算法（最常見的是稀疏與密集檢索）來提升搜尋結果相關性和召回率。

- **主要目標**：解決單一檢索技術的侷限性。例如，關鍵詞檢索無法理解語義，而向量檢索則可能忽略掉必須精確匹配的關鍵詞（如產品型號、函式名等）。混合檢索旨在同時利用稀疏向量的**精確性**和密集向量的**泛化性**，以應對複雜多變的搜尋需求。

### 2.1 技術原理與融合方法

混合檢索通常並行執行兩種檢索演算法，然後將兩組異構的結果集融合成一個統一的排序列表。以下是兩種主流的融合策略：

#### 2.1.1 倒數排序融合 (Reciprocal Rank Fusion, RRF)

RRF 不關心不同檢索系統的原始得分，只關心每個文件在各自結果集中的**排名**。其思想是：一個文件在不同檢索系統中的排名越靠前，它的最終得分就越高。

其計分公式為：

$$ RRF_{score}(d) = \sum_{i=1}^{k} \frac{1}{rank_i(d) + c} $$

其中：
- $d$ 是待評分的文件。
- $k$ 是檢索系統的數量（這裡是2，即稀疏和密集）。
- $rank_i(d)$ 是文件 $d$ 在第 $i$ 個檢索系統中的排名。
- $c$ 是一個常數（通常設為60），用於降低排名靠前文件的相對權重，實現更穩健的排名融合。

#### 2.1.2 加權線性組合

這種方法需要先將不同檢索系統的得分進行歸一化（例如，統一到 0-1 區間），然後透過一個權重引數 `α` 來進行線性組合。

$$ Hybrid_{score} = \alpha \cdot Dense_{score} + (1 - \alpha) \cdot Sparse_{score} $$

透過調整 `α` 的值，可以靈活地控制語義相似性與關鍵詞匹配在最終排序中的貢獻比例。例如，在電商搜尋中，可以調高關鍵詞的權重；而在智慧問答中，則可以側重於語義。

### 2.2 優勢與侷限

| 優勢 | 侷限 |
| :--- | :--- |
| **召回率與準確率高**：能同時捕獲關鍵詞和語義，顯著優於單一檢索。 | **計算資源消耗大**：需要同時維護和查詢兩套索引。 |
| **靈活性強**：可透過融合策略和權重調整，適應不同業務場景。 | **引數除錯複雜**：融合權重等超引數需要反覆實驗調優。 |
| **容錯性好**：關鍵詞檢索可部分彌補向量模型對拼寫錯誤或罕見詞的敏感性。 | **可解釋性仍是挑戰**：融合後的結果排序理由難以直觀分析。 |

## 三、程式碼實踐：透過 Milvus 實現混合檢索

接下來使用 Milvus 來實現一個完整的混合檢索流程，從定義 Schema、插入資料，到執行查詢。

### 3.1 步驟一：定義 Collection

在上一章中我們實現了多模態圖文檢索，現在還是同樣的步驟先建立一個 Collection。

```python
import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 初始化設定
COLLECTION_NAME = "dragon_hybrid_demo"
MILVUS_URI = "http://localhost:19530"  # 伺服器模式
DATA_PATH = "../../data/C4/metadata/dragon.json"  # 相對路徑
BATCH_SIZE = 50

# 2. 連線 Milvus 並初始化嵌入模型
print(f"--> 正在連線到 Milvus: {MILVUS_URI}")
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
print(f"--> 嵌入模型初始化完成。密集向量維度: {ef.dim['dense']}")

# 3. 建立 Collection
milvus_client = MilvusClient(uri=MILVUS_URI)
if milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在刪除已存在的 Collection '{COLLECTION_NAME}'...")
    milvus_client.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]

# 如果集合不存在，則建立它及索引
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在建立 Collection '{COLLECTION_NAME}'...")
    schema = CollectionSchema(fields, description="關於龍的混合檢索示例")
    # 建立集合
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print("--> Collection 建立成功。")

    # 建立索引
    print("--> 正在為新集合建立索引...")
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引建立成功。")

    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引建立成功。")

collection = Collection(COLLECTION_NAME)
collection.load()
print(f"--> Collection '{COLLECTION_NAME}' 已載入到記憶體。")
```

**fields欄位型別分析：**

- **pk**: 主鍵設計，`auto_id=True` 讓 Milvus 自動生成唯一標識，避免主鍵衝突
- **標量欄位**: 7個VARCHAR欄位用於儲存後設資料，`max_length` 根據實際資料分佈最佳化儲存
- **稀疏向量**: `SPARSE_FLOAT_VECTOR` 型別，儲存關鍵詞權重
- **密集向量**: `FLOAT_VECTOR` 型別，固定1024維，儲存語義特徵


### 3.2 步驟二：BGE-M3 雙向量生成

這裡使用 BGE-M3 作為向量生成器，它能夠同時生成稀疏向量和密集向量。

#### 3.2.1 資料載入與預處理

```python
if collection.is_empty:
    print(f"--> Collection 為空，開始插入資料...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    docs, metadata = [], []
    for item in dataset:
        parts = [
            item.get('title', ''),
            item.get('description', ''),
            item.get('location', ''),
            item.get('environment', ''),
        ]
        docs.append(' '.join(filter(None, parts)))
        metadata.append(item)
```

Collection 此時已載入到記憶體但為空狀態。透過 `is_empty` 檢查避免重複插入。多欄位文字合併中每個實體對應一個完整的資料記錄。

#### 3.2.2 向量生成

```python
print("--> 正在生成向量嵌入...")
embeddings = ef(docs)
print("--> 向量生成完成。")

# 獲取兩種向量
sparse_vectors = embeddings["sparse"]    # 稀疏向量：詞頻統計
dense_vectors = embeddings["dense"]      # 密集向量：語義編碼
```

#### 3.2.3 Collection 批次資料插入

```python
# 為每個欄位準備批次資料
img_ids = [doc["img_id"] for doc in metadata]
paths = [doc["path"] for doc in metadata]
titles = [doc["title"] for doc in metadata]
descriptions = [doc["description"] for doc in metadata]
categories = [doc["category"] for doc in metadata]
locations = [doc["location"] for doc in metadata]
environments = [doc["environment"] for doc in metadata]

# 插入資料
collection.insert([
    img_ids, paths, titles, descriptions, categories, locations, environments,
    sparse_vectors, dense_vectors
])
collection.flush()
```

- **欄位對映**: 嚴格按照 Schema 定義的欄位順序插入，9個欄位（7個標量+2個向量）
- **`flush()` 作用**: 強制將記憶體緩衝區資料寫入磁碟，使資料立即可搜尋
- **最終狀態**: Collection 包含6個Entity，索引層使用稀疏向量的 `SPARSE_INVERTED_INDEX` 和密集向量的 `AUTOINDEX`

### 3.3 步驟三：實現混合檢索

最後使用 milvus 中封裝好的 RRF 排序演算法來完成混合檢索：

#### 3.3.1 查詢向量生成

```python
# 6. 執行搜尋
search_query = "懸崖上的巨龍"
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
top_k = 5

print(f"\n{'='*20} 開始混合搜尋 {'='*20}")
print(f"查詢: '{search_query}'")
print(f"過濾器: '{search_filter}'")

# 生成查詢向量
query_embeddings = ef([search_query])
dense_vec = query_embeddings["dense"][0]
sparse_vec = query_embeddings["sparse"]._getrow(0)
```

嘗試列印向量資訊可以看到如下輸出：

```bash
=== 向量資訊 ===
密集向量維度: 1024
密集向量前5個元素: [-0.0035305   0.02043397 -0.04192593 -0.03036701 -0.02098157]
密集向量範數: 1.0000

稀疏向量維度: 250002
稀疏向量非零元素數量: 6
稀疏向量前5個非零元素:
  - 索引: 6, 值: 0.0659
  - 索引: 7977, 值: 0.1459
  - 索引: 14732, 值: 0.2959
  - 索引: 31433, 值: 0.1463
  - 索引: 141121, 值: 0.1587

稀疏向量密度: 0.00239998%
```

#### 3.3.2 混合檢索執行

使用 RRF 演算法進行混合檢索，透過 milvus 封裝的 RRFRanker 實現。RRFRanker 的核心引數是 `k` 值（預設60），用於控制 RRF 演算法中的排序平滑程度。

其中 `k` 值越大，排序結果越平滑；越小則高排名結果的權重越突出

```python
# 定義搜尋引數
search_params = {"metric_type": "IP", "params": {}}

# 先執行單獨的搜尋
print("\n--- [單獨] 密集向量搜尋結果 ---")
dense_results = collection.search(
    [dense_vec],
    anns_field="dense_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(dense_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路徑: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

print("\n--- [單獨] 稀疏向量搜尋結果 ---")
sparse_results = collection.search(
    [sparse_vec],
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(sparse_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路徑: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

print("\n--- [混合] 稀疏+密集向量搜尋結果 ---")
# 建立 RRF 融合器
rerank = RRFRanker(k=60)

# 建立搜尋請求
dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k)
sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", search_params, limit=top_k)

# 執行混合搜尋
results = collection.hybrid_search(
    [sparse_req, dense_req],
    rerank=rerank,
    limit=top_k,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

# 列印最終結果
for i, hit in enumerate(results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路徑: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")
```

最終輸出如下：
```bash
--- [單獨] 密集向量搜尋結果 ---
1. 懸崖上的白龍 (Score: 0.7219)
    路徑: ../../data/C3/dragon/dragon02.png
    描述: 一頭雄偉的白色巨龍棲息在懸崖邊緣，背景是金色的雲霞和遠方的海岸。它擁有巨大的翅膀和優雅的身姿，是典型的西方奇幻生物。...
2. 中華金龍 (Score: 0.5131)
    路徑: ../../data/C3/dragon/dragon06.png
    描述: 一條金色的中華龍在祥雲間盤旋，它身形矯健，龍鬚飄逸，展現了東方神話中龍的威嚴與神聖。...
3. 馴龍高手：無牙仔 (Score: 0.5119)
    路徑: ../../data/C3/dragon/dragon05.png
    描述: 在電影《馴龍高手》中，主角小嗝嗝騎著他的龍夥伴無牙仔在高空飛翔。他們飛向燦爛的太陽，下方是島嶼和海洋，畫面充滿了冒險與友誼。...

--- [單獨] 稀疏向量搜尋結果 ---
1. 懸崖上的白龍 (Score: 0.2319)
    路徑: ../../data/C3/dragon/dragon02.png
    描述: 一頭雄偉的白色巨龍棲息在懸崖邊緣，背景是金色的雲霞和遠方的海岸。它擁有巨大的翅膀和優雅的身姿，是典型的西方奇幻生物。...
2. 中華金龍 (Score: 0.0923)
    路徑: ../../data/C3/dragon/dragon06.png
    描述: 一條金色的中華龍在祥雲間盤旋，它身形矯健，龍鬚飄逸，展現了東方神話中龍的威嚴與神聖。...
3. 馴龍高手：無牙仔 (Score: 0.0691)
    路徑: ../../data/C3/dragon/dragon05.png
    描述: 在電影《馴龍高手》中，主角小嗝嗝騎著他的龍夥伴無牙仔在高空飛翔。他們飛向燦爛的太陽，下方是島嶼和海洋，畫面充滿了冒險與友誼。...

--- [混合] 稀疏+密集向量搜尋結果 ---
1. 懸崖上的白龍 (Score: 0.0328)
    路徑: ../../data/C3/dragon/dragon02.png
    描述: 一頭雄偉的白色巨龍棲息在懸崖邊緣，背景是金色的雲霞和遠方的海岸。它擁有巨大的翅膀和優雅的身姿，是典型的西方奇幻生物。...
2. 中華金龍 (Score: 0.0320)
    路徑: ../../data/C3/dragon/dragon06.png
    描述: 一條金色的中華龍在祥雲間盤旋，它身形矯健，龍鬚飄逸，展現了東方神話中龍的威嚴與神聖。...
3. 霸王龍的怒吼 (Score: 0.0318)
    路徑: ../../data/C3/dragon/dragon03.png
    描述: 史前時代的霸王龍張開血盆大口，發出震天的怒吼。在它身後，幾隻翼龍在陰沉的天空中盤旋，展現了白堊紀的原始力量。...
4. 奔跑的奶龍 (Score: 0.0313)
    路徑: ../../data/C3/dragon/dragon04.png
    描述: 一隻Q版的黃色小恐龍，有著大大的綠色眼睛和友善的微笑。是一部動畫中的角色，非常可愛。...
5. 馴龍高手：無牙仔 (Score: 0.0310)
    路徑: ../../data/C3/dragon/dragon05.png
    描述: 在電影《馴龍高手》中，主角小嗝嗝騎著他的龍夥伴無牙仔在高空飛翔。他們飛向燦爛的太陽，下方是島嶼和海洋，畫面充滿了冒險與友誼。...
```

> [本節完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C4/01_hybrid_search.py)

## 練習

- 分析程式碼為什麼在密集向量檢索和稀疏向量檢索中，排名第三的馴龍高手在混合檢索中反而排在了第五？
- 基於上一節的多模態檢索程式碼 `04_multi_milvus.py` ，結合本節的檢索程式碼加入多模態資訊融合的功能並嘗試使用混合檢索。（[參考程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C3/work_multimodal_dragon_search.py)）
