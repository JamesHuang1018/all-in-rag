# 多模態 Omni Embedding 實踐（Jina v5-omni）

## 一、這個專題要解決什麼問題

主線教程（`docs/chapter3/07_multimodal_embedding.md`）已經講清楚了 CLIP 雙編碼器和 BGE-Visualized 圖文檢索的基礎正規化。本專題接著往前一步，驗證下一代"統一多模態語義空間"的實際效果：

> **文字、影象、音訊、影片、PDF 能不能被對映到同一個向量空間裡，並按語義遠近排序？**

本專題只做嵌入與相似度層面的驗證，不做端到端 RAG pipeline。

**驗收標準**：指令碼跑通，輸出一組相似度矩陣，並且你能說出"為什麼海灘影象與海灘描述文字的相似度，高於與無關音訊的相似度"。

**本專題不覆蓋的內容**：跨模態檢索的生產調優、大規模向量庫接入、不同模型之間的效能對比。

---

## 二、技術演進：從 CLIP 到 v5-omni

理解本專題的實驗前，有必要先看一眼這條技術路線是怎麼走過來的：

| 階段 | 模型代表 | 支援模態 | 核心思路 |
|------|----------|----------|----------|
| 第一代 | CLIP | 圖 + 文 | 雙編碼器 + 對比學習，對齊圖文向量空間 |
| 第二代 | BGE-Visualized-M3 | 圖 + 文 | 在強文字底座上接入 patch token，聯合建模 |
| 第三代 | Jina v5-omni | 圖 + 文 + 音訊 + 影片 + PDF | GELATO：凍結文字主幹，僅訓練輕量投影層 |

關鍵變化是：v5-omni 不需要重訓整個模型就能擴充套件到新模態，而且已有文字向量的系統升級後**文字路徑行為不變**，可以直接接入不需要重建全量索引。

---

## 三、GELATO 設計與向量空間

### 3.1 核心思路

GELATO（Geometry-preserving Embeddings via Locked Aligned TOwers）的核心是"凍結主幹、只訓練連線"：

- 文字主幹（jina-embeddings-v5-text-nano）和非文字編碼器均**凍結**
- 只訓練投影層（`fc_vision_2`、`fc_audio`）和模態分隔符 token embedding
- 可訓練引數約佔總權重 **0.35%**
- 文字輸入的 embedding 行為與原文字底座完全一致

這套設計讓"支援新模態"和"保持已有文字幾何穩定"可以在同一個模型裡同時實現。

### 3.2 nano 版模型組成

| 元件 | 來源 | 引數量（載入視覺時） |
|------|------|------|
| 文字主幹 | jina-embeddings-v5-text-nano | 0.24B |
| 視覺編碼器 | SigLIP2 Base，來自 Qwen3.5-0.8B | 合計約 354M |
| 音訊編碼器 | Whisper-large-v3，來自 Qwen2.5-Omni-7B | 合計約 916M |
| 輸出維度 | — | 768 維 |

載入時透過 `modality` 引數控制例項化哪些編碼器塔，不需要的模態不會載入，節省視訊記憶體。

### 3.3 任務介面卡

模型內建 4 個任務介面卡（LoRA + 任務專屬投影權重），推理時透過 `default_task` 選擇：

- `retrieval`：資訊檢索
- `text-matching`：語義相似度
- `clustering`：聚類
- `classification`：分類

本專題使用 `retrieval` 任務。

### 3.4 nano 與 small 的工程選型

兩個版本均支援全部模態，差別主要體現在引數規模和 embedding 維度上。工程上通常先用 nano 跑通，再根據召回評測結果決定是否升級到 small。

---

## 四、實驗場景

### 4.1 場景設定

假設你在做一個**海邊內容檢索**演示：語料庫裡有不同語言的文字描述、兩張實拍圖片、一段音訊和一段影片，以及一份與海灘無關的學術論文摘錄（作為"負樣本"參照）。

目標是：用一個關於"海灘夕陽"的文字 query，驗證不同模態的內容能否按語義遠近正確排序。

### 4.2 data/ 檔案說明

```
data/
├── beach1.jpg                        # 海灘夕陽實拍照片（約 50 KB）
├── beach2.jpg                        # 另一張海灘照片（約 78 KB）
├── example-audio-clip.wav            # 音訊片段（約 4.6 MB）
├── example-video-clip.mp4            # 影片片段（約 968 KB）
└── paper_2506.18902_excerpt_2pages.pdf  # 論文摘錄（與海灘無關，約 117 KB）
```

### 4.3 三輪 query 設計

指令碼內建 3 輪語義相近但表述各異的 query，用於觀察向量幾何的穩定性：

| 輪次 | query 文字 |
|------|-----------|
| R1 | `sunset on the beach` |
| R2 | `waves and sunset on coast` |
| R3 | `beach scene with warm orange sky` |

---

## 五、目錄結構

```
Extra-chapter/multimodal-embedding-omni-practice/
├── readme.md                         # 本文
├── images/
│   └── 3_2_3.png                     # v5-omni 架構圖
├── code/
│   ├── 08_jina_embedding_omni.py     # 可執行指令碼
│   └── pyproject.toml                # 依賴配置（uv）
└── data/
    ├── beach1.jpg
    ├── beach2.jpg
    ├── example-audio-clip.wav
    ├── example-video-clip.mp4
    └── paper_2506.18902_excerpt_2pages.pdf
```

---

## 六、最小實現骨架

下面是指令碼主幹的虛擬碼，對應 `code/08_jina_embedding_omni.py` 的執行順序：

```python
# 1) 載入模型（本地優先，否則從 HF 下載）
raw_model = AutoModel.from_pretrained(model_path, default_task="retrieval", modality="vision")
processor = AutoProcessor.from_pretrained(model_path)
st_model  = SentenceTransformer(model_path, model_kwargs={"default_task": "retrieval"})
# raw_model + processor 用於文字和影象編碼
# st_model 用於音訊、影片、PDF 編碼（自動按副檔名識別模態）

# 2) 構建語料庫（corpus，9 個向量）
docs  = raw_model.embed(processor(text=[4 種語言描述], padding=True, ...))
img1  = raw_model.embed(processor(images=[beach1.jpg], text="<image>", ...))
img2  = raw_model.embed(processor(images=[beach2.jpg], text="<image>", ...))
audio = st_model.encode("example-audio-clip.wav")
video = st_model.encode("example-video-clip.mp4")
pdf   = st_model.encode("paper_2506.18902_excerpt_2pages.pdf")
corpus = stack([docs×4, img1, img2, audio, video, pdf])   # shape (9, 768)

# 3) 對每輪 query 編碼並計算相似度矩陣
for name, q in [("R1", ...), ("R2", ...), ("R3", ...)]:
    qv     = raw_model.embed(processor(text=f"Query: {q}", ...))
    fusion = raw_model.embed(processor(images=[beach1.jpg], text=q, ...))
    # fusion = 同一張圖 + 當前 query 文字，融合為單一向量
    vectors = stack([qv, corpus, fusion])   # shape (11, 768)
    sim = cosine_similarity(vectors)         # shape (11, 11)
    print(sim)

# 4) 對比三輪 query 之間的向量方向對齊度
aligned = [cosine(R1[i], R2[i]) for i in range(11)]
```

關鍵設計點：

- 文件編碼使用 `"Document: "` 字首，query 使用 `"Query: "` 字首（retrieval 任務的非對稱路由）
- `fusion` 向量把影象內容和 query 文字合併成一個向量，通常比純文字 query 相似度更高
- `st_model.encode()` 傳入檔案路徑即可，SentenceTransformers >= 5.4 會按副檔名自動選擇音訊/影片/PDF 編解碼路徑

---

## 七、環境準備與執行

### 7.1 依賴安裝

> ⚠️ 處理音訊、影片、PDF 需要的依賴比純文字場景多。`pyproject.toml` 已配置好所有必要包，直接用 `uv sync` 安裝即可。

```bash
cd Extra-chapter/multimodal-embedding-omni-practice/code
uv venv
source .venv/bin/activate
uv sync
```

主要依賴說明：

| 包 | 用途 |
|----|------|
| `sentence-transformers>=5.4.0` | 多模態 encode 支援（5.4 起才有） |
| `av>=17.0.0` | 影片幀提取 |
| `librosa>=0.11.0` + `soundfile` | 音訊解碼 |
| `pypdf` + `pypdfium2` | PDF 解析與渲染 |
| `pillow` + `torchvision` | 影象處理 |

**模型說明**：指令碼優先使用本地路徑 `models/jina-embeddings-v5-omni-nano`（相對倉庫根目錄），若不存在則自動從 HuggingFace 下載 `jinaai/jina-embeddings-v5-omni-nano`。國內網路建議提前手動下載。

### 7.2 執行指令碼

```bash
# 從 code/ 目錄執行（路徑解析基於指令碼位置，不依賴 shell 工作目錄）
python 08_jina_embedding_omni.py
```

指令碼啟動時會列印 `model_source=...` 確認使用的是本地還是遠端模型。載入過程會顯示進度條（`Loading weights`），屬正常現象。

> 💡 **transformers 警告**：啟動時可能出現 `[transformers] torch_dtype is deprecated! Use dtype instead!` 這是 transformers 庫內部版本遷移的提示，不影響結果，可忽略。

---

## 八、結果解讀（真實輸出）

### 8.1 向量佈局

每一輪（R1/R2/R3）輸出一個 shape=(11, 768) 的矩陣，11 個向量的排列固定如下：

| idx | 型別 | 內容 |
|-----|------|------|
| 0 | query | 當前輪次的文字 query（如 "Query: sunset on the beach"） |
| 1 | doc_en | "Document: A beautiful sunset over the beach" |
| 2 | doc_fr | "Document: Un beau coucher de soleil sur la plage" |
| 3 | doc_zh | "Document: 海灘上美麗的日落" |
| 4 | doc_ja | "Document: 浜辺に沈む美しい夕日" |
| 5 | img1 | beach1.jpg |
| 6 | img2 | beach2.jpg |
| 7 | audio | example-audio-clip.wav |
| 8 | video | example-video-clip.mp4 |
| 9 | pdf | paper_2506.18902_excerpt_2pages.pdf |
| 10 | fusion | img1 + 當前 query 文字（圖文融合向量） |

### 8.2 R1 相似度矩陣（真實輸出）

以下是 R1（query: `sunset on the beach`）的真實執行結果中 **query 行**（第 0 行）的相似度值：

```
R1 | shape=(11, 768)

query 行 sim[0][j]：

  idx  型別        相似度    說明
  ---  --------  -------  ------------------------------------------
   10  fusion     0.9159   ★★★ img1 + query 融合向量，遠超純文字 query
    3  doc_zh     0.7372   ★★★ 中文描述（海灘上美麗的日落）
    2  doc_fr     0.7200   ★★  法文描述（同義）
    4  doc_ja     0.5841   ★★  日文描述（同義）
    6  img2       0.5921   ★★  第二張海灘照片（跨模態）
    5  img1       0.5836   ★★  第一張海灘照片（跨模態）
    7  audio      0.1201   ★   音訊，弱正相關
    8  video      0.0632       影片，接近零
    9  pdf       -0.0014       無關論文，接近零（符合預期）
    1  doc_en    -0.0153   ⚠   英文描述（見下方說明）
```

**觀察一：圖文融合 >> 純文字 query**

`fusion`（0.9159）遠高於純文字 query 自身與語料的最高相似度（0.7372）。這印證了"影象 + 文字"聯合輸入可以更精準地表達檢索意圖。如果你的檢索場景能提供示例圖片，把它和 query 文字合併編碼是一個值得嘗試的方向。

**觀察二：跨語言對齊工作良好**

法文（0.72）、中文（0.7372）、日文（0.5841）的描述都與英文 query 有正相關，體現了模型的多語言能力。這些文字在彼此之間的相似度更高（法文 vs 中文：0.84，中文 vs 日文：0.80），說明同語義內容在向量空間中形成了穩定的語義簇。

**觀察三：影象跨模態相似度合理（約 0.58）**

兩張海灘照片與文字 query 的相似度在 0.58 左右，明顯低於同語義文字（0.72-0.74），但遠高於音訊（0.12）和影片（0.06）。圖片之間的相似度很高（img1 vs img2：0.9167），說明模型能識別同一場景的不同拍攝。

**觀察四：音影片相似度分層**

音訊（0.12）、影片（0.06）、PDF（≈0）依次遞減，與"內容相關性"基本一致：音訊可能含有海浪或環境音，影片畫面內容與海灘相關，而論文與海灘完全無關。

### 8.3 多輪對齊分析（真實輸出）

```
R1 -> R2
aligned=[0.7623 1.     1.     1.     1.     1.     1.     1.     1.     1.     0.0309]
best_corpus_idx=3, score=0.7372

R1 -> R3
aligned=[0.6056 1.     1.     1.     1.     1.     1.     1.     1.     1.     0.2969]
best_corpus_idx=3, score=0.7372
```

**aligned** 陣列中：

- 索引 1-9（語料庫向量）的對齊值均為 **1.0**——語料庫在每輪中重新編碼，但輸入不變，所以向量完全一致，屬於預期行為。
- 索引 0（query 向量）：R1 vs R2 = **0.7623**，R1 vs R3 = **0.6056**。這說明語義相近但表述不同的 query，在向量空間中的方向不完全相同，R3 的措辭與 R1 差別更大。
- 索引 10（fusion 向量）：R1 vs R2 = **0.03**，R1 vs R3 = **0.30**。fusion 把影象與 query 文字合併，query 文字變化後 fusion 向量變化幅度更大——這說明 fusion 對文字語義的敏感度比純文字 query 還要高。

**best_corpus_idx=3** 在三輪中保持一致，指向 `doc_zh`（中文海灘描述），score=0.7372。

### 8.4 自檢指引

> 💡 **如何判斷結果是否正常？**
>
> 正常結果應滿足：
> - `fusion` 相似度 > 所有純文字 query 的相似度
> - 同語義多語言文字的相似度 > 影象相似度 > 音影片相似度
> - 無關 PDF 的相似度接近零（-0.1 ~ 0.1 之間）
> - img1 vs img2 相似度 > 0.85（同場景兩張圖）
>
> 如果上述關係顛倒或數值全部接近零，先檢查"常見失敗點"第 2 條（依賴版本）。

---

## 九、常見失敗點

### 9.1 `FileNotFoundError: Missing local asset`

指令碼啟動時找不到 `data/` 目錄下的檔案。確認你是從 `code/` 目錄執行指令碼，且 `data/` 資料夾已包含所有樣本檔案。

### 9.2 `ImportError` 或 `ModuleNotFoundError`（音訊/影片相關）

通常是因為舊版依賴或某個包未安裝。常見原因：

- `sentence-transformers < 5.4.0`：多模態 encode 支援從 5.4 起才有，舊版呼叫 `encode(audio_path)` 會報錯。
- `av` 未安裝：影片幀提取依賴 `PyAV`，需透過 `uv sync` 從更新後的 `pyproject.toml` 安裝。
- `librosa` / `soundfile` 未安裝：音訊解碼依賴。

解決方法：刪除舊環境重建。

```bash
rm -rf .venv && uv venv && uv sync
```

### 9.3 HuggingFace 下載超時

國內網路訪問 `huggingface.co` 可能超時。建議提前設定映象或手動下載模型：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
python 08_jina_embedding_omni.py
```

或透過 `huggingface-cli download jinaai/jina-embeddings-v5-omni-nano --local-dir models/jina-embeddings-v5-omni-nano` 預先下載到本地。

### 9.4 路徑解析錯誤（模型或資料）

指令碼透過 `Path(__file__).resolve().parent` 計算路徑，以指令碼檔案的位置為基準，不依賴 shell 的工作目錄。但如果你移動了 `code/` 或 `data/` 的相對位置，需要同步修改指令碼頂部的路徑常量。

### 9.5 記憶體不足（OOM）

`v5-omni-nano` 在 MPS（Apple Silicon）上記憶體佔用約 2-4 GB，正常可執行。如果遇到 OOM：

- 確認沒有其他大模型佔用記憶體
- 關閉瀏覽器等高記憶體佔用程式
- 如果 MPS 不可用，指令碼會自動退回 CPU（速度慢但可執行）

> ⚠️ **PDF 嵌入的額外記憶體開銷**：對 PDF 做嵌入時，`pypdfium2` 會將每一頁渲染成高解析度影象再送入視覺編碼器。頁數多或解析度高的 PDF 會在渲染階段短暫佔用大量記憶體（以及相應的 MPS 視訊記憶體）。如果指令碼在處理 PDF 時崩潰，優先排查是否記憶體不足，而不是程式碼問題。臨時解決方案：在執行前關閉其他佔記憶體的應用，或換用頁數更少的 PDF 樣本。

### 9.6 輸出全部接近零

如果相似度矩陣裡大多數值都在 ±0.01 之間，通常是模型載入時 `default_task` 或 `modality` 引數沒有生效，導致向量沒有走 retrieval 任務路徑。確認 `transformers` 版本 >= 4.52 且 `sentence-transformers` >= 5.4.0。

---

## 十、侷限與適用邊界

1. **資料規模小**：本 demo 語料庫只有 9 個向量，結論不代表生產環境的檢索效果。
2. **影片和音訊評測不充分**：各只有一個樣本，時序理解能力無法驗證。
3. **跨域遷移**：跨模態相似度受資料域影響明顯，本 demo 的相似度數值（如音訊 0.12、影片 0.06）不適用於其他型別的音影片內容。
4. **nano 模型侷限**：觀察到的英文文件異常（見 8.2）是 nano 模型的特定行為，small 模型或其他 API 介面可能有不同表現。
5. **模型和 API 持續演進**：文件中的引數上限與可用能力以官方最新頁面為準。

---

## 十一、Gemini Embedding 2 參考

作為同類產品的參照，`gemini-embedding-2`（Vertex AI）也提供原生多模態嵌入能力：

- 支援文字、影象、文件、音訊、影片的**交錯輸入**（interleaved）
- 預設輸出維度 **3072**，可透過 `output_dimensionality` 下調
- 多模態共享上下文視窗（總 token 上限 8192）
- 各模態有獨立的輸入上限（圖片數量、PDF 頁數、音影片時長等）

與 v5-omni 的主要差異在於：Gemini Embedding 2 是託管 API，不支援本地部署；v5-omni 是開源模型，可本地執行，且能平滑相容已有的 jina v5-text 文字索引。本專題引用此資訊僅供橫向理解，不做效能優劣判斷。

---

## 十二、參考資料

- Jina v5-omni 官方說明：<https://jina.ai/news/jina-embeddings-v5-omni-multimodal-embeddings-for-text-image-audio-and-video/>
- GELATO 論文（arXiv 2605.08384）：<https://arxiv.org/abs/2605.08384>
- SentenceTransformers 5.4 多模態釋出說明：<https://github.com/UKPLab/sentence-transformers/releases/tag/v5.4.0>
- Gemini Embedding 2（Vertex AI 文件）：<https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings>
- Gemini Embedding 2（DeepMind 模型頁）：<https://deepmind.google/models/gemini/embedding/>
- 主線教程參考：`docs/chapter3/07_multimodal_embedding.md`
