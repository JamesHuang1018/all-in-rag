# PowerRAG SDK 文字問答檢索 Demo

## 一、這篇專題要解決什麼問題？

很多同學做 RAG 時會先把注意力放在“怎麼讓大模型回答得更像人”。但只要檢索沒找對上下文，生成再花哨也只是“把錯講得更順”。

這個專題做一件更樸素、也更值得先掌握的事：

> **只做檢索，不做生成。**

你會把一份 Markdown 文件交給服務端，讓服務端完成解析、切分、向量化，然後用問題去做 Top‑K 檢索，拿回最相關的原文片段（chunks）。

**驗收標準也很直接**：Top‑K chunks 是否與問題語義相關（不要求最終答案）。

本專題目錄結構：

- `readme.md`：本文（教學文件）
- `images/`：配圖
- `code/`：可執行指令碼與配置（`main.py`、`config.py`、`.env.example`、`requirements.txt`）
- `data/`：可復現樣例資料（`sample.md` + `questions.txt`）

---

## 二、技術方案：從 Markdown 到 Top‑K chunks（圖文講清楚）

下面這張圖展示了端到端鏈路，也基本對應 `code/main.py` 的執行順序。

<div align="center">
  <img src="images/10_1_1.webp" alt="端到端流程圖：上傳→解析/切分→向量化→Top-K 檢索" width="100%" />
  <p>圖 10.1: 端到端流程（本 demo 只驗收檢索結果，不要求生成最終回答）</p>
</div>

為了避免“看完圖還是不知道自己要做什麼”，這裡把圖 10.1 的關鍵節點按順序講清楚（你可以邊對照圖邊往下讀）：

**（1）本地輸入：Markdown 文件**

你可以直接用本專題提供的 `data/sample.md`。這份檔案故意寫得短：包含“排隊規則”和“退款規則”，方便你用不同問題去驗證檢索是否命中。

**（2）Upload：上傳到 dataset**

上傳不是“把文字發過去就結束”，它的意義在於：服務端要把這份文件納入某個 **dataset**（容器）裡，後續切分出來的 chunks、embedding、索引都掛在這個容器下面。

**（3）Parse/Chunk：解析 + 切分**

這一步會把 Markdown 解析成可檢索的文字結構，並按服務端策略切成多個 chunk。

> ⚠️ 圖裡標了一個常見失敗點：如果你的 tenant 沒有配置預設 embedding（`embd_id` 為空或未授權），解析任務可能直接 FAIL。

**（4）Embedding：向量化**

每個 chunk 會被對映成向量（embedding）。這一步是向量檢索的前提——沒有向量，後面就談不上“語義相似”。

**（5）寫入向量庫/索引**

chunk + embedding 會寫入向量索引（圖裡叫 Vector Store / Index）。

**（6）Retrieve Top‑K：檢索並返回 chunks**

輸入一個問題（question），服務端從索引裡找出最相關的 K 個 chunk，並把這些原文片段返回給你。本 demo 的驗收就看這裡：**返回的 chunks 是否包含你期望的規則段落**。

---

到這裡，你應該已經能把這條鏈路從頭到尾“順著說一遍”了：

> 文件上傳 → 服務端解析/切分/向量化 → 寫入索引 → 問題檢索 → 返回 Top‑K chunks。

但很多初學者還有一個常見困惑：**這些名詞到底對應什麼物件？我拿到的結果到底是誰？**

所以下面我們換一個視角：不再看“流程”，而是看“物件之間的關係”。

---

再看圖 10.2（物件關係）。這張圖的目的只有一個：把“你上傳的檔案”和“檢索返回的結果”徹底區分開。

很多同學第一次用 RAG 平臺 SDK，會把這些概念混在一起。你只要記住：

- **dataset**：容器（裝很多文件）
- **document**：你上傳的那份檔案
- **chunk**：文件切分出來的文字片段（檢索返回的就是它）

<div align="center">
  <img src="images/10_1_2.webp" alt="物件關係圖：dataset-document-chunk-embedding 與 Top-K 返回" width="100%" />
  <p>圖 10.2: 物件關係與返回結構（檢索返回的核心物件是 chunk）</p>
</div>

圖 10.2 裡最容易忽略、但最關鍵的一點是：**檢索返回的是 chunk，不是 document。**

- document 是“你上傳的整份檔案”
- chunk 是“切分後的片段”，它才是檢索、重排、壓縮、最終拼上下文的基本單位

所以你在終端裡看到的 Top‑K 結果，應該是一段段原文片段，而不是整篇 Markdown。

> 💡 小白自檢：我怎麼判斷“這段 chunk 就是我想要的那段”？
>
> 很簡單：用你自己的語言把問題再複述一遍，然後在返回的 chunk 裡找“能直接支撐答案的原文句子”。
> 例如你問“已發貨未簽收能不能退款”，chunk 裡應當出現“已發貨未簽收：可申請退款，但需要承擔退貨運費”這一類關鍵句。

---
## 三、實現思路：從零寫一版“最小檢索指令碼”（帶程式碼塊）

先給一個“最小骨架”（你可以把它當作虛擬碼，但它基本就是 `code/main.py` 的主幹）：

```python
rag = RAGFlow(api_key=..., base_url=...)

# 1) 建立 dataset（容器）
dataset = rag.create_dataset(name=...)

# 2) 上傳文件（拿到 doc.id）
doc = dataset.upload_documents([{...}])[0]

# 3) 解析/切分/向量化（失敗大多發生在這裡）
parse_results = dataset.parse_documents([doc.id])

# 4) 檢索 Top-K chunks（驗收點）
chunks = rag.retrieve(question=..., dataset_ids=[dataset.id], document_ids=[doc.id], page_size=top_k)
```

下面把每一步展開講清楚（並配上程式碼片段）。

### 3.1 引數與配置：先讓指令碼可復現

先從命令列引數入手，理解指令碼“能調什麼”。`code/main.py` 裡最常用的是這幾個：

```python
parser.add_argument("--file", type=Path, required=True)
parser.add_argument("--question", type=str, required=True)
parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k)
parser.add_argument("--dataset-name", type=str, default=DEFAULT_CONFIG.dataset_name)
parser.add_argument("--base-url", type=str, default=DEFAULT_CONFIG.base_url)
parser.add_argument("--api-key", type=str, default=DEFAULT_CONFIG.api_key)
```

- `--file`：你要上傳哪份 Markdown
- `--question`：你想驗證的提問
- `--top-k`：返回多少個 chunk
- `--dataset-name`：本次建立/使用的資料集名字
- `--base-url/--api-key`：PowerRAG 服務端地址與 SDK token

這幾個引數足夠讓你完成“換文件、換問題、調 Top‑K、連不同服務端”這四類最常見實驗。

> 💡 小白自檢：為什麼這裡既支援命令列引數，又支援 `.env`？
>
> 因為這兩種場景都很常見：
>
> - 你本地除錯時，喜歡用 `.env` 固定住 base_url/api_key
> - 你改引數做實驗時，喜歡命令列直接覆蓋（不用反覆改檔案）

### 3.2 初始化 SDK：先連上再說

```python
from ragflow_sdk import RAGFlow

rag = RAGFlow(api_key=api_key, base_url=base_url)
```

這裡沒有花活：就是把請求的 base_url 和 token 配好。

### 3.3 建立 dataset：把文件放進“一個籃子裡”

```python
dataset_kwargs = {"name": args.dataset_name}
if args.embedding_model:
    dataset_kwargs["embedding_model"] = args.embedding_model
dataset = rag.create_dataset(**dataset_kwargs)
```

為什麼要先有 dataset？因為“上傳/解析/檢索”都需要一個邊界。
你不希望每次檢索都在整個租戶的所有文件裡搜；你希望“只在這次實驗的文件集合裡搜”。

> 💡 小白自檢：能不能不建 dataset，直接上傳然後檢索？
>
> 取決於平臺能力。但在 PowerRAG/RAGFlow 這類系統裡，dataset 是“組織邊界”。
> 沒有邊界，檢索要麼全庫搜（不可控），要麼壓根沒有地方掛索引。

### 3.4 上傳 document：得到 doc.id，後面都靠它

```python
docs = dataset.upload_documents([
    {"display_name": display_name, "blob": blob}
])
doc = docs[0]
```

上傳成功後，SDK 會返回一個 document 物件（至少包含 `doc.id`）。
後續的 parse 和 retrieve 都要用它來限定範圍。

> 💡 小白自檢：為什麼要限定 `document_ids=[doc.id]`？
>
> 因為你這次實驗只關心“這份文件”的檢索效果。
> 如果不限定，dataset 裡有多份文件時，你可能會檢索到別的文件的 chunk，導致結果看起來“跑偏”。

### 3.5 解析 / 切分 / 向量化：最容易踩坑的一步

```python
parse_results = dataset.parse_documents([doc.id])
print("Parse results:")
print(parse_results)
```

指令碼會把 parse 的狀態列印出來，並且做了一個很直接的判斷：

```python
statuses = {r[1] for r in parse_results if isinstance(r, (list, tuple)) and len(r) >= 2}
if statuses and statuses != {"DONE"}:
    raise SystemExit("Document parsing failed (status not DONE)...")
```

你可以把它理解為“驗收關卡”：

- **DONE**：說明服務端已經把文件切成 chunk，並完成（或至少開始完成）向量化與索引寫入
- **FAIL/其他狀態**：先彆著急改程式碼，優先排查 tenant 預設 embedding

> 經驗：`Model(@None) not authorized` 基本就是在提示“預設 embedding 沒配/沒許可權”。

> 💡 小白自檢：為什麼 embedding 配置會影響“解析（parse）”？
>
> 因為這裡的 parse 往往不是“純語法解析 Markdown”，而是一條“解析 → 切分 → 向量化 → 寫索引”的流水線任務。
> embedding 不可用時，流水線中途失敗，平臺就會把整個任務標為 FAIL。

### 3.6 檢索 Top‑K：你真正要驗收的結果

```python
chunks = rag.retrieve(
    question=args.question,
    dataset_ids=[dataset.id],
    document_ids=[doc.id],
    page=1,
    page_size=args.top_k,
    similarity_threshold=args.similarity_threshold,
    vector_similarity_weight=args.vector_similarity_weight,
    top_k=args.candidate_k,
    keyword=args.keyword,
)
```

這裡有兩個點值得你留意（也是很多人調參的入口）：

- `page_size=args.top_k`：你最終想看多少條 chunk
- `similarity_threshold`：太高會過濾掉結果導致空，太低會混進無關段落

最後指令碼會把每條 chunk 的內容預覽列印出來：

```python
for i, c in enumerate(chunks, start=1):
    content = _safe_get(c, "content", "")
    print(f"{i:02d}. {content[:260]}")
```

你要做的“人工驗收”也很簡單：看看這幾段文字是不是回答問題所需的那幾段原文。

> 💡 小白自檢：Top‑K 是不是越大越好？
>
> 不是。Top‑K 太大容易把無關 chunk 混進來；太小又可能漏掉關鍵段落。
> 教學 demo 裡一般用 3~8 都夠用。

---

### 3.7 先跑通一次（最短路徑）

> ⚠️ 注意：解析/向量化依賴 embedding。如果你的 tenant 沒有配置預設 embedding（`embd_id` 為空或未授權），解析階段會 FAIL。不要先懷疑 Python。

```bash
# 1) 安裝依賴
cd Extra-chapter/PowerRAG-SDK-Text-QA/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) 配置 .env（在 code/ 目錄下）
cp .env.example .env

# 3) 回到專題根目錄執行（data/ 路徑更直觀）
cd ..
python code/main.py \
  --file data/sample.md \
  --question "已發貨未簽收的退款規則是什麼？" \
  --top-k 5 \
  --cleanup
```

你會看到兩段關鍵輸出：

1. `Parse results`：解析/分塊狀態（期望 `DONE`）
2. `Retrieved chunks`：Top‑K chunks 的內容預覽

---

---

## 四、經驗總結與坑點（把時間花在對的地方）

很多時候問題不在“你寫的 Python”，而在“服務端是不是已經把 embedding 產出來了”。

<div align="center">
  <img src="images/10_1_3.webp" alt="簡化時序圖：上傳→解析→寫入索引→Top-K 檢索→返回 chunks" width="100%" />
  <p>圖 10.3: code/main.py 與服務端 API 的互動順序（簡化版）</p>
</div>

如果你只記一條順序，就記這句：

> **先上傳 → 再解析（產出 chunk+embedding）→ 最後檢索（返回 chunk）**

很多“為什麼檢索不到”的問題，本質是解析還沒成功，索引里根本沒有向量。

---

### 4.1 Parse results 是 FAIL

優先檢查 tenant 的預設 embedding（`embd_id`）是否已配置且可用。典型錯誤：

- `Model(@None) not authorized`
- `Parse results: ... FAIL ...`

如果已經配置仍失敗，直接看 task executor 日誌最省時間：

```bash
docker exec powerrag-powerrag-1 sh -lc 'tail -n 200 /ragflow/logs/task_executor_* | tail -n 200'
```

### 4.2 401/403：token 型別搞混

PowerRAG 常見會同時出現兩類 token：

- Web 層 `AUTH`（用於 `/v1/*`）
- SDK 的 `ragflow-...` token（用於 `/api/v1/*`，通常寫在 `Authorization: Bearer <ragflow-...>`）

如果你看到 401/403，先確認 token 型別和介面字首是否匹配。

---

---

## 附錄：用 API 配預設 embedding + 生成 ragflow token（重操作區）

> 這部分是“環境/賬號/服務端配置”，放到附錄，避免主線被淹沒。

### A1. 用 API 配好 embedding（通用）

這一步需要一個 Web 層的 `AUTH`（`/v1/*` 使用），它和 SDK 的 `ragflow-...` key 不是一回事。

你可以把 embedding 配置寫進 `.env`（見 `.env.example` 的 `EMB_*`），下面命令會讀取 `EMB_FACTORY/EMB_MODEL/EMB_API_BASE/EMB_API_KEY`。

#### A1.1 獲取 `AUTH`（註冊並從響應頭拿 Authorization）

PowerRAG 的 `/v1/user/register` 要求 password 先用服務端的 RSA public key 加密。最省事的方式是在容器內呼叫它自帶的加密函式：

```bash
BASE_URL="http://127.0.0.1:9380"

ENC_PW="$(docker exec powerrag-powerrag-1 sh -lc 'python - <<"PY"\nfrom api.utils.crypt import crypt\nprint(crypt("powerrag"))\nPY')"

EMAIL="powerrag.demo.$(date +%s)@example.com"
AUTH="$(curl -sS -D - -o /dev/null -X POST "$BASE_URL/v1/user/register" \
  -H 'Content-Type: application/json' \
  -d "{\"nickname\":\"demo\",\"email\":\"$EMAIL\",\"password\":\"$ENC_PW\"}" \
| awk 'BEGIN{IGNORECASE=1} /^authorization:/{print $2}' | tr -d '\r')"
```

#### A1.2 繫結 embedding 的外部 API

> 注意：`max_tokens` 需要顯式傳，否則可能報資料庫欄位錯誤。

```bash
curl -sS -X POST "$BASE_URL/v1/llm/add_llm" \
  -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' \
  -d '{
    "llm_factory": "'"${EMB_FACTORY}"'",
    "model_type": "embedding",
    "llm_name": "'"${EMB_MODEL}"'",
    "api_base": "'"${EMB_API_BASE}"'",
    "api_key": "'"${EMB_API_KEY}"'",
    "max_tokens": 8192
  }'
```

#### A1.3 設定 tenant 預設 `embd_id`

```bash
TENANT_ID="$(curl -sS -H "Authorization: $AUTH" "$BASE_URL/v1/user/tenant_info" | python -c 'import sys,json; print(json.load(sys.stdin)["data"]["tenant_id"])')"

curl -sS -X POST "$BASE_URL/v1/user/set_tenant_info" \
  -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' \
  -d "{\"tenant_id\":\"$TENANT_ID\",\"llm_id\":\"\",\"embd_id\":\"${EMB_MODEL}@${EMB_FACTORY}\",\"asr_id\":\"\",\"img2txt_id\":\"\"}"
```

### A2. 生成 SDK 的 `ragflow-...` api_key

SDK 介面在 `/api/v1/*`，它不認 `AUTH`，需要 `ragflow-...` 這種 token（放在 header：`Authorization: Bearer <ragflow-...>`）。

用 `AUTH` 建立一個 SDK key：

```bash
DIALOG_ID="$(python -c 'import uuid; print(uuid.uuid4().hex)')"
API_KEY="$(curl -sS -X POST "$BASE_URL/v1/api/new_token" \
  -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' \
  -d "{\"dialog_id\":\"$DIALOG_ID\"}" \
| python -c 'import sys,json; print(json.load(sys.stdin)["data"]["token"])')"

echo "$API_KEY"
```
