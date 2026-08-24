# 第四節 Milvus介紹及多模態檢索實踐

## 一、簡介

Milvus 是一個開源的、專為大規模向量相似性搜尋和分析而設計的向量資料庫。它誕生於 Zilliz 公司，並已成為 LF AI & Data 基金會的頂級專案，在AI領域擁有廣泛的應用。

與 FAISS、ChromaDB 等輕量級本地儲存方案不同，Milvus 從設計之初就瞄準了**生產環境**。其採用雲原生架構，具備高可用、高效能、易擴充套件的特性，能夠處理十億、百億甚至更大規模的向量資料。

**官網地址**: [https://milvus.io/](https://milvus.io/)

**GitHub**: [https://github.com/milvus-io/milvus](https://github.com/milvus-io/milvus)

## 二、 部署安裝

Milvus 提供了多種部署方式，這裡以 **Milvus Standalone (單機版)** 為例。

### 1. 環境準備

- **安裝 Docker 與 Docker Compose**: 確保系統中已安裝並正在執行 Docker 和 Docker Compose。如果你對 Docker 不熟悉，可以參考這篇詳細的教程：[Docker 萬字教程：從入門到掌握](https://mp.weixin.qq.com/s/u2es87JU5FNlGo3qDLY_ng)。

> codespace 環境自帶Docker Compose無需安裝

### 2. 下載並啟動 Milvus

在你選定的工作目錄下，開啟終端（Terminal）或命令列工具（PowerShell），執行以下步驟：

**第一步：下載配置檔案**

使用以下命令下載官方的 `docker-compose.yml` 檔案。這個檔案定義了 Milvus Standalone 及其執行所需的兩個核心依賴服務：`etcd` 用於儲存後設資料，`MinIO` 用於物件儲存（更多架構細節請參考[官方文件](https://milvus.io/docs/architecture_overview.md)）。

```bash
# macOS / Linux (使用 wget)
wget https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

```powershell
# Windows (使用 PowerShell)
Invoke-WebRequest -Uri "https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml" -OutFile "docker-compose.yml"
```

**第二步：啟動 Milvus 服務**

在 `docker-compose.yml` 檔案所在的目錄中，執行以下命令以後臺模式啟動 Milvus：

```bash
docker compose up -d
```

Docker 將會自動拉取所需的映象並啟動三個容器：`milvus-standalone`, `milvus-minio`, 和 `milvus-etcd`。這個過程可能需要幾分鐘，具體取決於你的網路狀況。

### 3. 驗證安裝

可以透過以下方式驗證 Milvus 是否成功啟動：

- **檢視 Docker 容器**: 開啟 Docker Desktop 的儀表盤 (Windows/macOS) 或在終端執行 `docker ps` 命令 (Linux)，確認三個 Milvus 相關容器（`milvus-standalone`, `milvus-minio`, `milvus-etcd`）都處於 `running` 或 `up` 狀態。
- **檢查服務埠**: Milvus Standalone 預設透過 `19530` 埠提供服務，這是後續程式碼連線時需要用到的地址。

### 4. 常用管理命令

- **停止服務**:
  ```bash
  docker compose down
  ```
  此命令會停止並移除容器，但保留儲存的資料卷。

- **徹底清理 (停止並刪除資料)**:
  如果想徹底刪除所有資料（包括向量、後設資料等），可以執行以下命令：
  ```bash
  docker compose down -v
  ```

## 三、核心元件

### 3.1 Collection (集合)

可以用一個圖書館的比喻來理解 Collection：

- **Collection (集合)**: 相當於一個**圖書館**，是所有資料的頂層容器。一個 Collection 可以包含多個 Partition，每個 Partition 可以包含多個 Entity。
- **Partition (分割槽)**: 相當於圖書館裡的**不同區域**（如“小說區”、“科技區”），將資料物理隔離，讓檢索更高效。
- **Schema (模式)**: 相當於圖書館的**圖書卡片規則**，定義了每本書（資料）必須登記哪些資訊（欄位）。
- **Entity (實體)**: 相當於**一本具體的書**，是資料本身。
- **Alias (別名)**: 相當於一個**動態的推薦書單**（如“本週精選”），它可以指向某個具體的 Collection，方便應用層呼叫，實現資料更新時的無縫切換。 

**Collection** 是 Milvus 中最基本的資料組織單位，類似於關係型資料庫中的一張**表 (Table)**。是我們儲存、管理和查詢向量及相關後設資料的容器。所有的資料操作，如插入、刪除、查詢等，都是圍繞 Collection 展開的。

一個 Collection 由其 **Schema** 定義，幷包含以下重要的子概念和特性：

#### 3.1.1 Schema

在建立 Collection 之前，必須先定義它的 **Schema**。 `Schema` 規定了 Collection 的資料結構，定義了其中包含的所有**欄位 (Field)** 及其屬性。一個設計良好的 Schema 是能夠保證資料一致性並提升查詢效能。

Schema 通常包含以下幾類欄位：

- **主鍵欄位 (Primary Key Field)**: 每個 Collection 必須有且僅有一個主鍵欄位，用於唯一標識每一條資料（實體）。它的值必須是唯一的，通常是整數或字串型別。
- **向量欄位 (Vector Field)**: 用於儲存核心的向量資料。一個 Collection 可以有一個或多個向量欄位，以滿足多模態等複雜場景的需求。
- **標量欄位 (Scalar Field)**: 用於儲存除向量之外的後設資料，如字串、數字、布林值、JSON 等。這些欄位可以用於過濾查詢，實現更精確的檢索。

![Schema 設計剖析](./images/3_4_1.webp)

上圖以一篇新聞文章為例，展示了一個典型的多模態、混合向量 Schema 設計。它將一篇文章拆解為：唯一的 `Article (ID)`、文字後設資料（如 `Title`、`Author Info`）、影象資訊（`Image URL`），併為影象和摘要內容分別生成了密集向量（`Image Embedding`, `Summary Embedding`）和稀疏向量（`Summary Sparse Embedding`）。

#### 3.1.2 Partition (分割槽)

**Partition** 是 Collection 內部的一個邏輯劃分。每個 Collection 在建立時都會有一個名為 `_default` 的預設分割槽。我們可以根據業務需求建立更多的分割槽，將資料按特定規則（如類別、日期等）存入不同分割槽。

**為什麼使用分割槽？**

- **提升查詢效能**: 在查詢時，可以指定只在一個或幾個分割槽內進行搜尋，從而大幅減少需要掃描的資料量，顯著提升檢索速度。
- **資料管理**: 便於對部分資料進行批次操作，如載入/解除安裝特定分割槽到記憶體，或者刪除整個分割槽的資料。

一個 Collection 最多可以有 1024 個分割槽。合理利用分割槽是 Milvus 效能最佳化的重要手段之一。

#### 3.1.3 Alias (別名)

**Alias** (別名) 是為 Collection 提供的一個“暱稱”。透過為一個 Collection 設定別名，我們可以在應用程式中使用這個別名來執行所有操作，而不是直接使用真實的 Collection 名稱。

**為什麼使用別名？**

- **安全地更新資料**：想象一下，你需要對一個線上服務的 Collection 進行大規模的資料更新或重建索引。直接在原 Collection 上操作風險很高。正確的做法是：
    1. 建立一個新的 Collection (`collection_v2`) 並匯入、索引好所有新資料。
    2. 將指向舊 Collection (`collection_v1`) 的別名（例如 `my_app_collection`）原子性地切換到新 Collection (`collection_v2`) 上。
- **程式碼解耦**：整個切換過程對上層應用完全透明，無需修改任何程式碼或重啟服務，實現了資料的平滑無縫升級。

### 3.2 索引 (Index)

如果說 Collection 是 Milvus 的骨架，那麼**索引 (Index)** 就是其加速檢索的神經系統。從宏觀上看，索引本身就是一種**為了加速查詢而設計的複雜資料結構**。對向量資料建立索引後，Milvus 可以極大地提升向量相似性搜尋的速度，代價是會佔用額外的儲存和記憶體資源。

![Milvus 索引結構與工作原理](./images/3_4_2.webp)

上圖清晰地展示了 Milvus 向量索引的內部元件及其工作流程：
- **資料結構**：這是索引的骨架，定義了向量的組織方式（如 HNSW 中的圖結構）。
- **量化**(可選)：資料壓縮技術，透過降低向量精度來減少記憶體佔用和加速計算。
- **結果精煉**(可選)：在找到初步候選集後，進行更精確的計算以最佳化最終結果。

Milvus 支援對標量欄位和向量欄位分別建立索引。

- **標量欄位索引**：主要用於加速後設資料過濾，常用的有 `INVERTED`、`BITMAP` 等。通常使用推薦的索引型別即可。
- **向量欄位索引**：這是 Milvus 的核心。選擇合適的向量索引是在查詢效能、召回率和記憶體佔用之間做出權衡的藝術。

#### 3.2.1 主要向量索引型別

Milvus 提供了多種向量索引演算法，以適應不同的應用場景。以下是幾種最核心的型別：

- **FLAT (精確查詢)**
  - **原理**：暴力搜尋（Brute-force Search）。它會計算查詢向量與集合中所有向量之間的實際距離，返回最精確的結果。
  - **優點**：100% 的召回率，結果最準確。
  - **缺點**：速度慢，記憶體佔用大，不適合海量資料。
  - **適用場景**：對精度要求極高，且資料規模較小（百萬級以內）的場景。

- **IVF 系列 (倒排檔案索引)**
  - **原理**：類似於書籍的目錄。它首先透過聚類將所有向量分成多個“桶”(`nlist`)，查詢時，先找到最相似的幾個“桶”，然後只在這幾個桶內進行精確搜尋。`IVF_FLAT`、`IVF_SQ8`、`IVF_PQ` 是其不同變體，主要區別在於是否對桶內向量進行了壓縮（量化）。
  - **優點**：透過縮小搜尋範圍，極大地提升了檢索速度，是效能和效果之間很好的平衡。
  - **缺點**：召回率不是100%，因為相關向量可能被分到了未被搜尋的桶中。
  - **適用場景**：通用場景，尤其適合需要高吞吐量的大規模資料集。

- **HNSW (基於圖的索引)**
  - **原理**：構建一個多層的鄰近圖。查詢時從最上層的稀疏圖開始，快速定位到目標區域，然後在下層的密集圖中進行精確搜尋。
  - **優點**：檢索速度極快，召回率高，尤其擅長處理高維資料和低延遲查詢。
  - **缺點**：記憶體佔用非常大，構建索引的時間也較長。
  - **適用場景**：對查詢延遲有嚴格要求（如實時推薦、線上搜尋）的場景。

- **DiskANN (基於磁碟的索引)**
  - **原理**：一種為在 SSD 等高速磁碟上執行而最佳化的圖索引。
  - **優點**：支援遠超記憶體容量的海量資料集（十億級甚至更多），同時保持較低的查詢延遲。
  - **缺點**：相比純記憶體索引，延遲稍高。
  - **適用場景**：資料規模巨大，無法全部載入到記憶體的場景。

#### 3.2.2 如何選擇索引？

選擇索引沒有唯一的“最佳答案”，需要根據業務場景在**資料規模、記憶體限制、查詢效能和召回率**之間進行權衡。

| 場景 | 推薦索引 | 備註 |
| :--- | :--- | :--- |
| 資料可完全載入記憶體，追求低延遲 | **HNSW** | 記憶體佔用較大，但查詢效能和召回率都很優秀。 |
| 資料可完全載入記憶體，追求高吞吐 | **IVF_FLAT / IVF_SQ8** | 效能和資源消耗的平衡之選。 |
| 資料量巨大，無法載入記憶體 | **DiskANN** | 在 SSD 上效能優異，專為海量資料設計。 |
| 追求 100% 準確率，資料量不大 | **FLAT** | 暴力搜尋，確保結果最精確。 |

在實際應用中，通常需要透過測試來找到最適合自己資料和查詢模式的索引型別及其引數。

### 3.3 檢索

#### 3.3.1 基礎向量檢索 (ANN Search)

擁有了資料容器 (Collection) 和檢索引擎 (Index) 後，最後一步就是從海量資料中高效地檢索資訊。這是 Milvus 的核心功能之一，**近似最近鄰 (Approximate Nearest Neighbor, ANN) 檢索**。與需要計算全部資料的暴力檢索（Brute-force Search）不同，ANN 檢索利用預先構建好的索引，能夠極速地從海量資料中找到與查詢向量最相似的 Top-K 個結果。這是一種在速度和精度之間取得極致平衡的策略。

- **主要引數**:
  - `anns_field`: 指定要在哪個向量欄位上進行檢索。
  - `data`: 傳入一個或多個查詢向量。
  - `limit` (或 `top_k`): 指定需要返回的最相似結果的數量。
  - `search_params`: 指定檢索時使用的引數，例如距離計算方式 (`metric_type`) 和索引相關的查詢引數。

#### 3.3.2 增強檢索

在基礎的 ANN 檢索之上，Milvus 提供了多種增強檢索功能，以滿足更復雜的業務需求。

**過濾檢索 (Filtered Search)**

在實際應用中，我們很少只進行單純的向量檢索。更常見的需求是“在滿足特定條件的向量中，查詢最相似的結果”，這就是過濾檢索。它將**向量相似性檢索**與**標量欄位過濾**結合在一起。

- **工作原理**：先根據提供的過濾表示式 (`filter`) 篩選出符合條件的實體，然後僅在這個子集內執行 ANN 檢索。這極大地提高了查詢的精準度。
- **應用示例**：
  - **電商**："檢索與這件紅色連衣裙最相似的商品，但只看價格低於500元且有庫存的。"
  - **知識庫**："查詢與‘人工智慧’相關的文件，但只從‘技術’分類下、且釋出於2023年之後的文章中尋找。"

**範圍檢索 (Range Search)**

有時我們關心的不是最相似的 Top-K 個結果，而是“所有與查詢向量的相似度在特定範圍內的結果”。

- **工作原理**：範圍檢索允許定義一個距離（或相似度）的閾值範圍。Milvus 會返回所有與查詢向量的距離落在這個範圍內的實體。
- **應用示例**：
  - **人臉識別**："查詢所有與目標人臉相似度超過 0.9 的人臉"，用於身份驗證。
  - **異常檢測**："查詢所有與正常樣本向量距離過大的資料點"，用於發現異常。

**多向量混合檢索 (Hybrid Search)**

這是 Milvus 提供的一種極其強大的高階檢索模式，它允許在一個請求中同時檢索**多個向量欄位**，並將結果智慧地融合在一起。

- **工作原理**：
  1. **並行檢索**：應用針對不同的向量欄位（如一個用於文字語義的密集向量，一個用於關鍵詞匹配的稀疏向量，一個用於影象內容的多模態向量）分別發起 ANN 檢索請求。
  2. **結果融合 (Rerank)**：Milvus 使用一個重排策略（Reranker）將來自不同檢索流的結果合併成一個統一的、更高質量的排序列表。常用的策略有 `RRFRanker`（平衡各方結果）和 `WeightedRanker`（可為特定欄位結果加權）。

- **應用示例**：
  - **多模態商品檢索**：使用者輸入文字“安靜舒適的白色耳機”，系統可以同時檢索商品的**文字描述向量**和**圖片內容向量**，返回最匹配的商品。
  - **增強型 RAG**: 結合**密集向量**（捕捉語義）和**稀疏向量**（精確匹配關鍵詞），實現比單一向量更精準的文件檢索效果。

**分組檢索 (Grouping Search)**

分組檢索解決了一個常見的痛點：檢索結果多樣性不足。想象一下，你檢索“機器學習”，返回的前10篇文章都來自同一本教科書不同章節。這顯然不是理想的結果。

- **工作原理**：分組檢索允許指定一個欄位（如 `document_id`）對結果進行分組。Milvus 會在檢索後，確保返回的結果中每個組（每個 `document_id`）只出現一次（或指定的次數），且返回的是該組內與查詢最相似的那個實體。
- **應用示例**：
  - **影片檢索**：檢索“可愛的貓咪”，確保返回的影片來自不同的博主。
  - **文件檢索**：檢索“資料庫索引”，確保返回的結果來自不同的書籍或來源。

透過這些靈活的檢索功能組合，開發者可以構建出滿足各種複雜業務需求的向量檢索應用。

## 四、milvus多模態實踐

在本節中，我們將透過一個完整的示例，演示如何使用 Milvus 和 Visualized-BGE 模型構建一個端到端的圖文多模態檢索引擎。

### 4.1 初始化與工具定義

首先匯入所有必需的庫，定義好模型路徑、資料目錄等常量。為了程式碼的整潔和複用，將 Visualized-BGE 模型的載入和編碼邏輯封裝在一個 `Encoder` 類中，並定義了一個 `visualize_results` 函式用於後續的結果視覺化。

```python
import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

# 1. 初始化設定
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://localhost:19530"

# 2. 定義工具 (編碼器和視覺化函式)
class Encoder:
    """編碼器類，用於將影象和文字編碼為向量。"""
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]

def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300, img_width: int = 300, row_count: int = 3) -> np.ndarray:
    """從檢索到的影象列表建立一個全景圖用於視覺化。"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 處理查詢影象
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 處理檢索到的影象
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width
        
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
        
        # 新增索引號
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return np.hstack([query_display_area, panoramic_image])
```

### 4.2 建立 Collection

這是與 Milvus 互動的開始。首先初始化 Milvus 客戶端，然後定義 Collection 的 Schema，它規定了集合的資料結構。

```python
# 3. 初始化客戶端
print("--> 正在初始化編碼器和Milvus客戶端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 4. 建立 Milvus Collection
print(f"\n--> 正在建立 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已刪除已存在的 Collection: '{COLLECTION_NAME}'")

image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 影象。")
dim = len(encoder.encode_image(image_list[0]))

fields = [
    # 主鍵欄位，設定自增 (auto_id=True)
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 向量欄位，維度與模型的輸出向量維度一致
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    # 儲存原影象路徑的標量欄位，最大長度限制 512 字元
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]

# 建立集合 Schema
schema = CollectionSchema(fields, description="多模態圖文檢索")
print("Schema 結構:")
print(schema)

# 建立集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功建立 Collection: '{COLLECTION_NAME}'")
print("Collection 結構:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))
```

**輸出結果：**
```bash
--> 正在建立 Collection 'multimodal_demo'

Schema 結構:
{
    'auto_id': True, 
    'description': '多模態圖文檢索', 
    'fields': [
        {'name': 'id', 'description': '', 'type': <DataType.INT64: 5>, 'is_primary': True, 'auto_id': True}, 
        {'name': 'vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 768}}, 
        {'name': 'image_path', 'description': '', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 512}}
    ], 
    'enable_dynamic_field': False
}

成功建立 Collection: 'multimodal_demo'

Collection 結構:
{
    'collection_name': 'multimodal_demo', 
    'auto_id': True, 
    'num_shards': 1, 
    'description': '多模態圖文檢索', 
    'fields': [
        {'field_id': 100, 'name': 'id', 'description': '', 'type': <DataType.INT64: 5>, 'params': {}, 'auto_id': True, 'is_primary': True}, 
        {'field_id': 101, 'name': 'vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 768}}, 
        {'field_id': 102, 'name': 'image_path', 'description': '', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 512}}
    ], 
    'functions': [], 
    'aliases': [], 
    'collection_id': 459243798405253751, 
    'consistency_level': 2, 
    'properties': {}, 
    'num_partitions': 1, 
    'enable_dynamic_field': False, 
    'created_timestamp': 459249546649403396, 
    'update_timestamp': 459249546649403396
}
```

上面的輸出詳細展示了剛剛建立的 `multimodal_demo` Collection 的完整結構。其 **Schema** 包含了三個核心欄位（**Field**）：一個自增的 `id` 作為**主鍵**，一個 768 維的 `vector` **向量欄位**用於儲存影象嵌入，以及一個 `image_path` **標量欄位**來記錄原始圖片路徑。

### 4.3 準備並插入資料

建立好 Collection 後，需要將資料填充進去。透過遍歷指定目錄下的所有圖片，將它們逐一編碼成向量，然後與圖片路徑一起組織成符合 Schema 結構的格式，最後批次插入到 Collection 中。

```python
# 5. 準備並插入資料
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入資料")
data_to_insert = []
for image_path in tqdm(image_list, desc="生成影象嵌入"):
    vector = encoder.encode_image(image_path)
    data_to_insert.append({"vector": vector, "image_path": image_path})

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 條資料。")
```

### 4.4 建立索引

為了實現快速檢索，需要為向量欄位建立索引。這裡選擇 `HNSW` 索引，它在召回率和查詢效能之間有著很好的平衡。建立索引後，必須呼叫 `load_collection` 將集合載入到記憶體中才能進行搜尋。

```python
# 6. 建立索引
print(f"\n--> 正在為 '{COLLECTION_NAME}' 建立索引")
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功為向量欄位建立 HNSW 索引。")
print("索引詳情:")
print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已載入 Collection 到記憶體中。")
```

**輸出結果：**
```bash
--> 正在為 'multimodal_demo' 建立索引
成功為向量欄位建立 HNSW 索引。
索引詳情:
{'M': '16', 'efConstruction': '256', 'metric_type': 'COSINE', 'index_type': 'HNSW', 'field_name': 'vector', 'index_name': 'vector', 'total_rows': 0, 'indexed_rows': 0, 'pending_index_rows': 0, 'state': 'Finished'}
已載入 Collection 到記憶體中。
```

可以看出，索引建立成功，在 `vector` 欄位上成功建立了 `HNSW` 索引，並使用 `COSINE` 作為距離度量。`M: '16'` 和 `efConstruction: '256'` 是 HNSW 索引的兩個關鍵引數，分別控制著圖中每個節點的最大連線數和索引構建時的搜尋範圍，這些引數直接影響檢索的效能和準確性。`state: 'Finished'` 狀態表明索引已成功構建。

### 4.5 執行多模態檢索

這裡透過定義一個包含圖片和文字的組合查詢，將其編碼為查詢向量，然後呼叫 `search` 方法在 Milvus 中執行近似最近鄰搜尋。

```python
# 7. 執行多模態檢索
print(f"\n--> 正在 '{COLLECTION_NAME}' 中執行檢索")
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一條龍"
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    output_fields=["image_path"],
    limit=5,
    # ef: 搜尋時的節點遍歷深度，值越大召回率越高但耗時越長
    search_params={"metric_type": "COSINE", "params": {"ef": 128}}
)[0]

retrieved_images = []
print("檢索結果:")
for i, hit in enumerate(search_results):
    print(f"  Top {i+1}: ID={hit['id']}, 距離={hit['distance']:.4f}, 路徑='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])
```

**輸出結果：**

```bash
--> 正在 'multimodal_demo' 中執行檢索
檢索結果:
  Top 1: ID=459243798403756667, 距離=0.9411, 路徑='../../data/C3\dragon\query.png'
  Top 2: ID=459243798403756668, 距離=0.5818, 路徑='../../data/C3\dragon\dragon02.png'
  Top 3: ID=459243798403756671, 距離=0.5731, 路徑='../../data/C3\dragon\dragon05.png'
  Top 4: ID=459243798403756670, 距離=0.4894, 路徑='../../data/C3\dragon\dragon04.png'
  Top 5: ID=459243798403756669, 距離=0.4100, 路徑='../../data/C3\dragon\dragon03.png'
```

這段輸出展示了與圖文組合查詢最相似的5個**實體 (Entity)**。`distance` 欄位代表了**餘弦相似度**，值越接近 1 表示越相似。可以看到，`Top 1` 結果正是查詢圖片本身，其相似度得分最高（0.9411），這說明了檢索的有效性。其餘結果也都是龍的圖片，並按相似度從高到低精確排列。

> **為什麼查詢圖片本身，其餘弦相似度不是 1.0 而是 0.9411 呢？**
> 在插入資料時，資料庫裡存放的是透過 `encode_image` 得到的**純影象嵌入向量**（只包含視覺特徵）；而在檢索時，輸入的查詢向量是透過 `encode_query` 生成的**圖文聯合嵌入向量**（融合了文字“一條龍”的資訊和影象的特徵）。由於查詢向量被文字語義進行了“偏置”，所以它與資料庫中的純視覺影象向量之間會有細微的特徵差異。

### 4.6 視覺化與清理

最後，將檢索到的圖片路徑用於視覺化，生成一張直觀的結果對比圖。在完成所有操作後，應該釋放 Milvus 中的資源，包括從記憶體中解除安裝 Collection 和刪除整個 Collection。

```python
# 8. 視覺化與清理
if not retrieved_images:
    print("沒有檢索到任何影象。")
else:
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"結果影象已儲存到: {combined_image_path}")
    Image.open(combined_image_path).show()

milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已從記憶體中釋放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已刪除 Collection: '{COLLECTION_NAME}'")
```

![檢索結果視覺化](./images/3_4_3.png)

透過上圖可以看出，這個多模態檢索引擎成功地理解了“一條龍”這個圖文組合查詢的意圖，並從相簿中找到了最相關的幾張圖片並進行排序。

> [本節完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C3/04_multi_milvus.py)
