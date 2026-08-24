# 第三節 四步構建RAG

透過第一節的學習，我們對RAG已經有了基本認識，並且也準備好了虛擬環境和api_key，接下來將嘗試使用[**LangChain**](https://python.langchain.com/docs/introduction/)和[**LlamaIndex**](https://docs.llamaindex.ai/en/stable/)框架完成第一個RAG應用的實現與執行。透過一個示例，演示如何載入本地Markdown文件，利用嵌入模型處理文字，並結合大型語言模型（LLM）來回答與文件內容相關的問題。

## 一、啟動虛擬環境

### 1.1 啟用虛擬環境

假設已經按照前一章節的指導，建立了名為 `all-in-rag` 的 Conda 虛擬環境。在執行指令碼前，先啟用虛擬環境：

> 如果使用是Cloud Studio，需要確認當前是否是使用者環境，如果不是請執行 `su ubuntu` 切換到使用者環境。

```bash
conda activate all-in-rag
```

### 1.2 切換到專案目錄

```bash
# 假設當前在 all-in-rag 專案的根目錄下
cd code/C1
```

每章內容中的程式碼檔案都存放在 `code/Cx` 目錄下，其中 `x` 表示章節編號。

## 二、執行RAG示例程式碼

完成上述所有設定後，就可以執行RAG示例了。

開啟終端，確保虛擬環境已啟用，然後執行以下命令：

```bash
python 01_langchain_example.py
```

> 若出現nltk相關報錯，嘗試執行程式碼路徑下[fix_nltk.py](https://github.com/datawhalechina/all-in-rag/blob/main/code/C1/fix_nltk.py)

程式碼執行後，可以看到類似下面的輸出（格式化後）：

```bash
Downloading Model from https://www.modelscope.cn to directory: Path\to\all-in-rag\models\bge-small-zh-v1.5
2025-06-08 02:36:19,318 - modelscope - INFO - Target directory already exists, skipping creation.
content='
文中舉了以下例子：

1. **自然界中的羚羊**：剛出生的羚羊透過試錯學習站立和奔跑，適應環境。
2. **股票交易**：透過買賣股票並根據市場反饋調整策略，最大化獎勵。
3. **雅達利遊戲（如Breakout和Pong）**：透過不斷試錯學習如何通關或贏得遊戲。
4. **選擇餐館**：利用（去已知喜歡的餐館）與探索（嘗試新餐館）的權衡。
5. **做廣告**：利用（採取已知最優廣告策略）與探索（嘗試新廣告策略）。
6. **挖油**：利用（在已知地點挖油）與探索（在新地點挖油，可能發現大油田）。
7. **玩遊戲（如《街頭霸王》）**：利用（固定策略如蹲角落出腳）與探索（嘗試新招式如“大招”）。

這些例子用於說明強化學習中的核心概念（如探索與利用、延遲獎勵等）及其在實際場景中的應用。
'
additional_kwargs={'refusal': None}
response_metadata={
    'token_usage': {
        'completion_tokens': 209,
        'prompt_tokens': 5576,
        'total_tokens': 5785,
        'completion_tokens_details': None,
        'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 5568},
        'prompt_cache_hit_tokens': 5568,
        'prompt_cache_miss_tokens': 8
    },
    'model_name': 'deepseek-chat',
    'system_fingerprint': 'fp_8802369eaa_prod0425fp8',
    'id': '67a0580d-78b1-44d6-bccf-f654ae0e9bba',
    'service_tier': None,
    'finish_reason': 'stop',
    'logprobs': None
}
id='run--919cedcd-771e-4aed-8dfd-cf436795792e-0'
usage_metadata={
    'input_tokens': 5576,
    'output_tokens': 209,
    'total_tokens': 5785,
    'input_token_details': {'cache_read': 5568},
    'output_token_details': {}
}
```

> 首次執行時，指令碼會下載`BAAI/bge-small-zh-v1.5`嵌入模型。

輸出引數解析：
- **`content`**: 這是最核心的部分，即大型語言模型（LLM）根據你的問題和提供的上下文生成的具體回答。
- **`additional_kwargs`**: 包含一些額外的引數，在這個例子中是 `{'refusal': None}`，表示模型沒有拒絕回答。
- **`response_metadata`**: 包含了關於LLM響應的後設資料。
    - `token_usage`: 顯示了本次呼叫消耗的token數量，包括完成（completion_tokens）、提示（prompt_tokens）和總量（total_tokens）。
    - `model_name`: 使用的LLM模型名稱，當前是 `deepseek-chat`。
    - `system_fingerprint`, `id`, `service_tier`, `finish_reason`, `logprobs`: 這些是更詳細的API響應資訊，例如 `finish_reason: 'stop'` 表示模型正常完成了生成。
- **`id`**: 本次執行的唯一識別符號。
- **`usage_metadata`**: 與 `response_metadata` 中的 `token_usage` 類似，提供了輸入和輸出token的統計。

## 三、基於 LangChain 框架的 RAG 實現

在第一節中，我們提到四步構建最小可行系統分別是資料準備、索引構建、檢索最佳化和生成整合。下面將圍繞這四個方面來實現一個基於 LangChain 框架的 RAG 應用。

> [本節完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C1/01_langchain_example.py)

### 3.1 初始化設定

首先進行基礎配置，包括匯入必要的庫、載入環境變數以及下載嵌入模型。

```python
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatOpenAI

# 載入環境變數
load_dotenv()
```

### 3.2 資料準備

- **載入原始文件**: 先定義Markdown檔案的路徑，然後使用`TextLoader`載入該檔案作為知識源。
    ```python
    markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"
    loader = TextLoader(markdown_path)
    docs = loader.load()
    ```
- **文字分塊 (Chunking)**: 為了便於後續的嵌入和檢索，長文件被分割成較小的、可管理的文字塊（chunks）。這裡採用了遞迴字元分割策略，使用其預設引數進行分塊。當不指定引數初始化 `RecursiveCharacterTextSplitter()` 時，其預設行為旨在最大程度保留文字的語義結構：
    - **預設分隔符與語義保留**: 按順序嘗試使用一系列預設的分隔符 `["\n\n" (段落), "\n" (行), " " (空格), "" (字元)]` 來遞迴分割文字。這種策略的目的是儘可能保持段落、句子和單詞的完整性，因為它們通常是語義上最相關的文字單元，直到文字塊達到目標大小。
    - **保留分隔符**: 預設情況下 (`keep_separator=True`)，分隔符本身會被保留在分割後的文字塊中。
    - **預設塊大小與重疊**: 使用其基類 `TextSplitter` 中定義的預設引數 `chunk_size=4000`（塊大小）和 `chunk_overlap=200`（塊重疊）。這些引數確保文字塊符合預定的大小限制，並透過重疊來減少上下文資訊的丟失。
    ```python
    text_splitter = RecursiveCharacterTextSplitter()
    texts = text_splitter.split_documents(docs)
    ```

### 3.3 索引構建

資料準備完成後，接下來構建向量索引：

- **初始化中文嵌入模型**: 使用`HuggingFaceEmbeddings`載入之前在初始化設定中下載的中文嵌入模型。配置模型在CPU上執行，並啟用嵌入歸一化 (`normalize_embeddings: True`)。
    ```python
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    ```
- **構建向量儲存**: 將分割後的文字塊 (`texts`) 透過初始化好的嵌入模型轉換為向量表示，然後使用`InMemoryVectorStore`將這些向量及其對應的原始文字內容新增進去，從而在記憶體中構建出一個向量索引。
    ```python
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(texts)
    ```
    這個過程完成後，便構建了一個可供查詢的知識索引。

### 3.4 查詢與檢索

索引構建完畢後，便可以針對使用者問題進行查詢與檢索：

- **定義使用者查詢**: 設定一個具體的使用者問題字串。
    ```python
    question = "文中舉了哪些例子？"
    ```
- **在向量儲存中查詢相關文件**: 使用向量儲存的`similarity_search`方法，根據使用者問題在索引中查詢最相關的 `k` (此處示例中 `k=3`) 個文字塊。
    ```python
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    ```
- **準備上下文**: 將檢索到的多個文字塊的頁面內容 (`doc.page_content`) 合併成一個單一的字串，並使用雙換行符 (`"\n\n"`) 分隔各個塊，形成最終的上下文資訊 (`docs_content`) 供大語言模型參考。
    ```python
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    ```
    > 使用 `"\n\n"` (雙換行符) 而不是 `"\n"` (單換行符) 來連線不同的檢索文件塊，主要是為了在傳遞給大型語言模型（LLM）時，能夠更清晰地在語義上區分這些獨立的文字片段。雙換行符通常代表段落的結束和新段落的開始，這種格式有助於LLM將每個塊視為一個獨立的上下文來源，從而更好地理解和利用這些資訊來生成回答。

### 3.5 生成整合

最後一步是將檢索到的上下文與使用者問題結合，利用大語言模型（LLM）生成答案：

- **構建提示詞模板**: 使用`ChatPromptTemplate.from_template`建立一個結構化的提示模板。此模板指導LLM根據提供的上下文 (`context`) 回答使用者的問題 (`question`)，並明確指出在資訊不足時應如何回應。
    ```python
    prompt = ChatPromptTemplate.from_template("""請根據下面提供的上下文資訊來回答問題。
    請確保你的回答完全基於這些上下文。
    如果上下文中沒有足夠的資訊來回答問題，請直接告知：“抱歉，我無法根據提供的上下文找到相關資訊來回答此問題。”
    
    上下文:
    {context}
    
    問題: {question}
    
    回答:"""
                                              )
    ```
- **配置大語言模型**: 初始化 `ChatOpenAI` 客戶端，配置所用模型（`glm-4.7-flash-free`）、生成答案的溫度引數（`temperature=0.7`）、最大Token數 (`max_tokens=2048`) 以及API金鑰（從環境變數載入）和 url。
    ```python
    llm = ChatOpenAI(
        model="glm-4.7-flash-free",
        temperature=0.7,
        max_tokens=2048,
        api_key=os.getenv("DEEPSEEK_API_KEY")
        base_url="https://aihubmix.com/v1"
    )
    ```
- **呼叫LLM生成答案並輸出**: 將使用者問題 (`question`) 和先前準備好的上下文 (`docs_content`) 格式化到提示模板中，然後呼叫ChatDeepSeek的`invoke`方法獲取生成的答案。
    ```python
    answer = llm.invoke(prompt.format(question=question, context=docs_content))
    print(answer)
    ```

> 老溼老溼，Langchain 很強大但還是太吃操作了，有沒有更加簡單又好用的框架推薦呢？

> 有的兄弟，有的！像這樣好用的框架還有LlamaIndex😉

## 四、低程式碼（基於LlamaIndex）

在 RAG 方面，LlamaIndex 提供了更多封裝好的 API 介面，這無疑降低了上手門檻，下面是一個簡單實現：

```python
import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

documents = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中舉了哪些例子?"))
```

## 練習（可利用大模型輔助完成）

- LangChain程式碼最終得到的輸出攜帶了各種引數，查詢相關資料嘗試把這些引數過濾掉得到`content`裡的具體回答。
- 修改Langchain程式碼中`RecursiveCharacterTextSplitter()`的引數`chunk_size`和`chunk_overlap`，觀察輸出結果有什麼變化。
- 給LlamaIndex程式碼新增程式碼註釋。