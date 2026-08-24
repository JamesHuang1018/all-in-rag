# 第五節 檢索進階

在基礎的 RAG 流程中，依賴向量相似度從知識庫中檢索資訊。不過，這種方法存在一些固有的侷限性，例如最相關的文件不總是在檢索結果的頂端，以及語義理解的偏差等。為了構建更強大、更精準的生產級 RAG 應用，需要引入更高階的檢索技術。

![retrieval](images/4_5_1.webp)

## 一、重排序 (Re-ranking)

### 1.1 RRF (Reciprocal Rank Fusion)

我們在 [**混合檢索章節**](./11_hybrid_search.md) 中已經接觸過 RRF。它是一種簡單而有效的**零樣本**重排方法，不依賴於任何模型訓練，而是純粹基於文件在多個不同檢索器（例如，一個稀疏檢索器和一個密集檢索器）結果列表中的**排名**來計算最終分數。

一個文件如果在多個檢索結果中都排名靠前，那麼它很可能更重要。RRF 透過計算排名的倒數來為文件打分，有效融合了不同檢索策略的優勢。但是如果只考慮排名資訊，會忽略原始的相似度分數，可能丟失部分有用資訊。

### 1.2 RankLLM / LLM-based Reranker

![rankllm](images/4_5_2.webp)

RankLLM 代表了一類直接利用大型語言模型本身來進行重排的方法[^1]。其基本邏輯非常直觀：既然 LLM 最終要負責根據上下文來生成答案，那麼為什麼不直接讓它來判斷哪些上下文最相關呢？

這種方法透過一個精心設計的提示詞來實現。該提示詞會包含使用者的查詢和一系列候選文件（通常是文件的摘要或關鍵部分），然後要求 LLM 以特定格式（如 JSON）輸出一個排序後的文件列表，並給出每個文件的相關性分數。

一個提示詞示例如下：

```text
以下是一個文件列表，每個文件都有一個編號和摘要。同時提供一個問題。請根據問題，按相關性順序列出您認為需要查閱的文件編號，並給出相關性分數（1-10分）。請不要包含與問題無關的文件。

示例格式:
文件 1: <文件1的摘要>
文件 2: <文件2的摘要>
...
文件 10: <文件10的摘要>

問題: <使用者的問題>

回答:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3
```

### 1.3 Cross-Encoder 重排

Cross-Encoder（交叉編碼器）能提供出色的重排精度[^2]。它的工作原理是將查詢（Query）和每個候選文件（Document）**拼接**成一個單一的輸入（例如，`[CLS] query [SEP] document [SEP]`），然後將這個整體輸入到一個預訓練的 Transformer 模型（如 BERT）中，模型最終會輸出一個單一的分數（通常在 0 到 1 之間），這個分數直接代表了文件與查詢的**相關性**。

> 注：**[SEP]** 是在 BERT 這類基於 Transformer 架構的模型中，用於分隔不同文字片段（如查詢和文件）的特殊標記。

<div align="center">
<img src="./images/4_5_3.svg" alt="cross-encoder" width="600">
</div>

上圖清晰地展示了 Cross-Encoder 的工作流程：
1.  **初步檢索**：搜尋引擎首先從知識庫中召回一個初始的文件列表（例如，前 50 篇）。
2.  **逐一評分**：對於列表中的**每一篇**文件，系統都將其與原始查詢**配對**，然後傳送給 Cross-Encoder 模型。
3.  **獨立推理**：模型對每個“查詢-文件”對進行一次完整的、獨立的推理計算，得出一個精確的相關性分數。
4.  **返回重排結果**：系統根據這些新的分數對文件列表進行重新排序，並將最終結果返回給使用者。

這個流程凸顯了其高精度的來源（同時分析查詢和文件），也解釋了其高延遲的原因（需要N次獨立的模型推理）。

常見的 Cross-Encoder 模型包括 `ms-marco-MiniLM-L-12-v2`、`ms-marco-TinyBERT-L-2-v2` 等。

### 1.4 ColBERT 重排

ColBERT（Contextualized Late Interaction over BERT）是一種創新的重排模型，它在 Cross-Encoder 的高精度和雙編碼器（Bi-Encoder）的高效率之間取得了平衡[^3]。採用了一種“**後期互動**”機制。

其工作流程如下：

1.  **獨立編碼**：ColBERT 分別為查詢（Query）和文件（Document）中的每個 Token 生成上下文相關的嵌入向量。這一步是獨立完成的，可以預先計算並儲存文件的向量，從而加快查詢速度。
2.  **後期互動**：在查詢時，模型會計算查詢中每個 Token 的向量與文件中每個 Token 向量之間的最大相似度（MaxSim）。
3.  **分數聚合**：最後，將查詢中所有 Token 得到的最大相似度分數相加，得到最終的相關性總分。

透過這種方式，ColBERT 避免了將查詢和文件拼接在一起進行昂貴的聯合編碼，同時又比單純比較單個 `[CLS]` 向量的雙編碼器模型捕捉了更細粒度的詞彙級互動資訊。

### 1.5 重排方法對比

為了更直觀地理解不同重排方法的特點和適用場景，下表對討論過的幾種主流方法進行了總結：

| 特性 | RRF | RankLLM | Cross-Encoder | ColBERT |
| :--- | :--- | :--- | :--- | :--- |
| **核心機制** | 融合多個排名 | LLM 推理，生成排序列表 | 聯合編碼查詢與文件，計算單一相關分 | 獨立編碼，後期互動 |
| **計算成本** | 低（簡單數學計算） | 中 (API 費用與延遲) | 高（N次模型推理） | 中（向量點積計算） |
| **互動粒度** | 無（僅排名） | 概念/語義級 | 句子級（Query-Doc Pair） | Token 級 |
| **適用場景** | 多路召回結果融合 | 高價值語義理解場景 | Top-K 精排 | Top-K 重排 |

## 二、壓縮 (Compression)

“壓縮”技術旨在解決一個常見問題：初步檢索到的文件塊（Chunks）雖然整體上與查詢相關，但可能包含大量無關的“噪音”文字。將這些未經處理的、冗長的上下文直接提供給 LLM，不僅會增加 API 呼叫的成本和延遲，還可能因為資訊過載而降低最終生成答案的質量。

壓縮的目標就是對檢索到的內容進行“壓縮”和“提煉”，只保留與使用者查詢最直接相關的資訊。這可以透過兩種主要方式實現：
1.  **內容提取**：從文件中只抽出與查詢相關的句子或段落。
2.  **文件過濾**：完全丟棄那些雖然被初步召回，但經過更精細判斷後認為不相關的整個文件。

### 2.1 LangChain 的 ContextualCompressionRetriever

LangChain 提供了一個強大的元件 `ContextualCompressionRetriever` 來實現上下文壓縮[^4]。它像一個包裝器，包裹在基礎的檢索器（如 `FAISS.as_retriever()`）之上。當基礎檢索器返回文件後，`ContextualCompressionRetriever` 會使用一個指定的 `DocumentCompressor` 對這些文件進行處理，然後再返回給呼叫者。

LangChain 內建了多種 `DocumentCompressor`：

*   `LLMChainExtractor`: 這是最直接的壓縮方式。它會遍歷每個文件，並利用一個 LLM Chain 來判斷並提取出其中與查詢相關的部分。這是一種“內容提取”。
*   `LLMChainFilter`: 這種壓縮器同樣使用 LLM，但它做的是“文件過濾”。它會判斷整個文件是否與查詢相關，如果相關，則保留整個文件；如果不相關，則直接丟棄。
*   `EmbeddingsFilter`: 這是一種更快速、成本更低的過濾方法。它會計算查詢和每個文件的嵌入向量之間的相似度，只保留那些相似度超過預設閾值的文件。

### 2.2 自定義重排器與壓縮管道

在前面我們就提到根據實際應用，需要自己進行一些功能的實現。這裡以 ColBERT 為例，展示如何整合未被官方支援的功能。

整個探索和實現過程如下：

1.  **從官方文件出發**：首先，透過 LangChain 官方文件，瞭解到可以透過 `DocumentCompressorPipeline` 來組合多個壓縮器和文件轉換器。
2.  **需求缺口**：希望使用 ColBERT 模型進行重排，但發現 LangChain 並沒有內建的 `ColBERT` 重排器。
3.  **分析示例與原始碼**：回頭分析 `ContextualCompressionRetriever` 的用法和原始碼。我們發現，其處理邏輯分為兩步：首先使用 `base_retriever` 獲取原始文件，然後將這些文件交給 `base_compressor` 進行壓縮或重排。這說明，實現自定義後處理（如重排）功能的關鍵在於 `base_compressor`。
4.  **定位核心基類**：透過f12檢視原始碼，確定 `base_compressor` 引數接收的是 `BaseDocumentCompressor` 型別的物件。這就是實現自定義功能的核心切入點。
5.  **參考與實現**：最後，參考 LangChain 中其他重排器的實現方式，透過繼承 `BaseDocumentCompressor` 基類並實現其關鍵方法，建立自己的 `ColBERTReranker` 類。

> PS：如果程式碼基礎薄弱，想借助大模型幫你完成 `ColBERTReranker` ，需要提供給大模型的關鍵資訊：`BaseDocumentCompressor` 的原始碼和 `ContextualCompressionRetriever` 的原始碼及其使用示例、你的明確目標（實現 ColBERT 重排邏輯）、以及 LangChain 中其他重排器的程式碼作為參考。資訊越充分，模型生成的程式碼越準確。

#### 程式碼示例

自定義 `ColBERTReranker` 的程式碼實現：

```python
class ColBERTReranker(BaseDocumentCompressor):
    """ColBERT重排器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_name = "bert-base-uncased"

        # 載入模型和分詞器
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, 'model', AutoModel.from_pretrained(model_name))
        self.model.eval()
        print(f"ColBERT模型載入完成")

    def encode_text(self, texts):
        """ColBERT文字編碼"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def calculate_colbert_similarity(self, query_emb, doc_embs, query_mask, doc_masks):
        """ColBERT相似度計算（MaxSim操作）"""
        scores = []

        for i, doc_emb in enumerate(doc_embs):
            doc_mask = doc_masks[i:i+1]

            # 計算相似度矩陣
            similarity_matrix = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))

            # 應用文件mask
            doc_mask_expanded = doc_mask.unsqueeze(1)
            similarity_matrix = similarity_matrix.masked_fill(~doc_mask_expanded.bool(), -1e9)

            # MaxSim操作
            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]

            # 應用查詢mask
            query_mask_expanded = query_mask.unsqueeze(0)
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(~query_mask_expanded.bool(), 0)

            # 求和得到最終分數
            colbert_score = max_sim_per_query_token.sum(dim=-1).item()
            scores.append(colbert_score)

        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """對文件進行ColBERT重排序"""
        if len(documents) == 0:
            return documents

        # 編碼查詢
        query_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embeddings = F.normalize(query_outputs.last_hidden_state, p=2, dim=-1)

        # 編碼文件
        doc_texts = [doc.page_content for doc in documents]
        doc_inputs = self.tokenizer(
            doc_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            doc_outputs = self.model(**doc_inputs)
            doc_embeddings = F.normalize(doc_outputs.last_hidden_state, p=2, dim=-1)

        # 計算ColBERT相似度
        scores = self.calculate_colbert_similarity(
            query_embeddings,
            doc_embeddings,
            query_inputs['attention_mask'],
            doc_inputs['attention_mask']
        )

        # 排序並返回前5個
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs[:5]]

        return reranked_docs
```

1.  **繼承與實現**：`ColBERTReranker` 類繼承自 `BaseDocumentCompressor`，並實現了其核心的抽象方法 `compress_documents`。這個方法接收基礎檢索器返回的文件列表 `documents` 和原始查詢 `query` 作為輸入。

2.  **實現ColBERT邏輯**：`compress_documents` 方法的內部邏輯遵循了在 “1.4 ColBERT 重排” 中描述的“後期互動”原理。
    *   **獨立編碼**：在 `_colbert_score` 輔助函式中，查詢和文件分別被獨立編碼，透過 `self.model` 得到各自所有 Token 的嵌入向量（`query_embeddings` 和 `doc_embeddings`）。
    *   **後期互動**：程式碼 `similarity_matrix.max(dim=1).values` 實現了最大相似度（MaxSim）計算。為查詢中的每一個 Token 向量，都從文件的所有 Token 向量中尋找一個最相似的，並記錄下這個最大相似度值。
    *   **分數聚合**：最後的 `.sum()` 操作將查詢中所有 Token 算出的最大相似度值相加，得到該文件與查詢的最終相關性總分。

3.  **排序與返回**：`compress_documents` 方法遍歷所有文件、計算出各自的分數後，根據分數從高到低對文件進行重新排序，並返回排序後的文件列表。

接下來，將這個自定義的 `ColBERTReranker` 與 LangChain 的其他元件（如 `LLMChainExtractor`）組合成一個強大的“重排+壓縮”管道，並應用在實際的檢索任務中。

```python
# 初始化配置...(略)

# 1. 載入和處理文件
loader = TextLoader("../../data/C4/txt/ai.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 2. 建立向量儲存和基礎檢索器
vectorstore = FAISS.from_documents(docs, hf_bge_embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 3. 設定ColBERT重排序器
reranker = ColBERTReranker()

# 4. 設定LLM壓縮器
compressor = LLMChainExtractor.from_llm(llm)

# 5. 使用DocumentCompressorPipeline組裝壓縮管道
# 流程: ColBERT重排 -> LLM壓縮
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[reranker, compressor]
)

# 6. 建立最終的壓縮檢索器
final_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever
)

# 7. 執行查詢並展示結果
query = "AI還有哪些缺陷需要克服？"
print(f"\n{'='*20} 開始執行查詢 {'='*20}")
print(f"查詢: {query}\n")

# 7.1 基礎檢索結果
print(f"--- (1) 基礎檢索結果 (Top 20) ---")
base_results = base_retriever.get_relevant_documents(query)
for i, doc in enumerate(base_results):
    print(f"  [{i+1}] {doc.page_content[:100]}...\n")

# 7.2 使用管道壓縮器的最終結果
print(f"\n--- (2) 管道壓縮後結果 (ColBERT重排 + LLM壓縮) ---")
final_results = final_retriever.get_relevant_documents(query)
for i, doc in enumerate(final_results):
    print(f"  [{i+1}] {doc.page_content}\n")
```

這段程式碼展示瞭如何將各個元件串聯起來，形成一個完整的檢索流程：

1.  **建立基礎元件**：首先建立一個標準的 `FAISS` 向量儲存和一個基礎檢索器 `base_retriever`，負責從向量庫中初步召回20個可能相關的文件。
2.  **準備處理單元**：例項化兩個關鍵的處理單元：
    *   `reranker`: 自定義的 `ColBERTReranker` 例項。
    *   `compressor`: LangChain 內建的 `LLMChainExtractor`，用於從文件中提取與查詢相關的句子。
3.  **構建處理管道 (`DocumentCompressorPipeline`)**：這是整個流程的核心。建立一個 `DocumentCompressorPipeline` 例項，並將 `reranker` 和 `compressor` 按順序放入 `transformers` 列表中。根據 `DocumentCompressorPipeline` 的原始碼，它會依次呼叫列表中的每個處理器。因此，文件會先經過 `ColBERTReranker` 重排，重排後的結果再被送入 `LLMChainExtractor` 進行壓縮。
4.  **組裝最終檢索器**：最後，用 `ContextualCompressionRetriever` 將 `base_retriever` 和我們建立的 `pipeline_compressor` 包裝在一起。當呼叫 `final_retriever` 時，它會自動執行“基礎檢索 -> 管道處理（重排 -> 壓縮）”的完整流程。

> [完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C4/07_rerank_and_refine.py)

### 2.3 LlamaIndex 中的檢索壓縮

LlamaIndex 同樣提供了封裝好的壓縮功能，其代表是 `SentenceEmbeddingOptimizer`[^5]。它也是一個後處理器（Node Postprocessor），工作在檢索之後。

它的工作原理是，對於每個檢索到的文件，將其分解成句子。然後計算每個句子與使用者查詢的嵌入相似度，最後只保留那些相似度最高的句子，從而“最佳化”文件，去除無關資訊。

## 三、校正 (Correcting)

傳統的 RAG 流程有一個隱含的假設：檢索到的文件總是與問題相關且包含正確答案。然而在現實世界中，檢索系統可能會失敗，返回不相關、過時或甚至完全錯誤的文件。如果將這些“有毒”的上下文直接餵給 LLM，就可能導致幻覺（Hallucination）或產生錯誤的回答。

**校正檢索（Corrective-RAG, C-RAG）** 正是為解決這一問題而提出的一種策略[^6]。思路是引入一個“自我反思”或“自我修正”的迴圈，在生成答案之前，對檢索到的文件質量進行評估，並根據評估結果採取不同的行動。

C-RAG 的工作流程可以概括為 **“檢索-評估-行動”** 三個階段：

![C-RAG](images/4_5_4.webp)

1.  **檢索 (Retrieve)** ：與標準 RAG 一樣，首先根據使用者查詢從知識庫中檢索一組文件。

2.  **評估 (Assess)** ：這是 C-RAG 的關鍵步驟。如圖所示，一個“檢索評估器 (Retrieval Evaluator)”會判斷每個文件與查詢的相關性，並給出“正確 (Correct)”、“不正確 (Incorrect)”或“模糊 (Ambiguous)”的標籤。

3.  **行動 (Act)** ：根據評估結果，系統會進入不同的知識修正與獲取流程：
    *   **如果評估為“正確”**：系統會進入“知識精煉 (Knowledge Refinement)”環節。如圖，它會將原始文件分解成更小的知識片段 (strips)，過濾掉無關部分，然後重新組合成更精準、更聚焦的上下文，再送給大模型生成答案。
    *   **如果評估為“不正確”**：系統認為內部知識庫無法回答問題，此時會觸發“知識搜尋 (Knowledge Searching)”。它會先對原始查詢進行“查詢重寫 (Query Rewriting)”，生成一個更適合搜尋引擎的查詢，然後進行 Web 搜尋，用外部資訊來回答問題。
    *   **如果評估為“模糊”**：同樣會觸發“知識搜尋”，但通常會直接使用原始查詢進行 Web 搜尋，以獲取額外資訊來輔助生成答案。

透過這種方式，C-RAG 極大地增強了 RAG 系統的魯棒性。不再盲目信任檢索結果，而是增加了一個“事實核查”層，能夠在檢索失敗時主動尋求外部幫助，從而有效減少幻覺，提升答案的準確性和可靠性。

在 LangChain 的 `langgraph` 庫中，可以利用其圖結構來靈活地構建這種帶有條件判斷和迴圈的複雜 RAG 流程[^7]。

## 練習

- 本節“自定義重排器與壓縮管道”部分的程式碼執行後的輸出會出現重複的情況，思考為什麼會出現這個問題並嘗試修改程式碼解決。（[參考程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C4/work_rerank_and_refine.py)）

## 參考文獻

[^1]: [*Using LLM’s for Retrieval and Reranking*](https://www.llamaindex.ai/blog/using-llms-for-retrieval-and-reranking-23cf2d3a14b6).

[^2]: [Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT*](https://arxiv.org/abs/1901.04085).

[^3]: [*Advanced RAG: ColBERT Reranker*](https://www.pondhouse-data.com/blog/advanced-rag-colbert-reranker).

[^4]: [*How to do retrieval with contextual compression*](https://python.langchain.com/docs/how_to/contextual_compression/).

[^5]: [*Sentence Embedding Optimizer*](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/OptimizerDemo/).

[^6]: [Jiang, Z. et al. (2024). *Corrective Retrieval Augmented Generation*](https://arxiv.org/pdf/2401.15884.pdf).

[^7]: [*Corrective-RAG (CRAG)*](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/).


