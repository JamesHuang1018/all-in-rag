# 第二節 評估常用工具

瞭解了評估的基本原理之後，來介紹幾個RAG評估工具，它們各自代表了不同的設計哲學和應用場景。

## 一、LlamaIndex Evaluation

`LlamaIndex Evaluation` 是**深度整合於LlamaIndex框架內的評估模組**，專為使用該框架構建的RAG應用提供無縫的評估能力。作為RAG開發框架的原生元件，其核心定位是**為開發者在開發、除錯和迭代週期中提供快速、靈活的嵌入式評估解決方案**。它強調與開發流程的緊密結合，允許開發者在構建過程中即時驗證和對比不同RAG策略的效能[^1]。

> **適用場景**：對於深度使用 `LlamaIndex` 框架構建RAG應用的開發者而言，其內建評估模組是無縫整合的首選，提供了一站式的開發與評估體驗。

### 1.1 核心理念與工作流

`LlamaIndex` 的評估理念是利用LLM作為“裁判”，以自動化的方式對RAG系統的各個環節進行打分。這種方法在很多場景下無需預先準備“標準答案”，大大降低了評估門檻。其典型工作流如下：

1.  **準備評估資料集**：透過 `DatasetGenerator` 從文件中自動生成問題-答案對（`QueryResponseDataset`），或載入一個已有的資料集。為了效率，通常會將生成的資料集儲存到本地，避免重複生成。
2.  **構建查詢引擎**：搭建一個或多個需要被評估的RAG查詢引擎（`QueryEngine`）。這是進行對比實驗的基礎。
3.  **初始化評估器**：根據評估維度，選擇並初始化一個或多個評估器，如 `FaithfulnessEvaluator`（忠實度）和 `RelevancyEvaluator`（相關性）。
4.  **執行批次評估**：使用 `BatchEvalRunner` 來管理整個評估過程。它能夠高效地（可並行）將查詢引擎應用於資料集中的所有問題，並呼叫所有評估器進行打分。
5.  **分析結果**：從評估執行器返回的結果中，計算各項指標的平均分，從而量化地對比不同RAG策略的優劣。

### 1.2 應用例項：對比不同檢索策略

下面示例基於我們在第三章學習的“句子視窗檢索”技術，透過評估，對比它與“常規分塊檢索”在響應質量上的差異。

**程式碼示例：**


```python
# ... (省略資料載入、文件解析、查詢引擎構建等步驟)

# 1. 初始化評估器
# 定義需要評估的指標：忠實度和相關性
faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)
evaluators = {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator}

# 2. 使用BatchEvalRunner執行批次評估
# 從資料集中獲取查詢列表
queries = response_eval_dataset.queries

# 評估“句子視窗檢索”引擎
print("\n=== 評估句子視窗檢索 ===")
sentence_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)
sentence_response_results = await sentence_runner.aevaluate_queries(
    queries=queries, query_engine=sentence_query_engine
)

# 評估“常規分塊檢索”引擎
print("\n=== 評估常規分塊檢索 ===")
base_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)
base_response_results = await base_runner.aevaluate_queries(
    queries=queries, query_engine=base_query_engine
)

# 3. 分析並列印結果
# ... (省略結果計算與列印的輔助函式)
print(f"句子視窗檢索: 忠實度={sentence_faith:.1%}, 相關性={sentence_rel:.1%}")
print(f"常規分塊檢索: 忠實度={base_faith:.1%}, 相關性={base_rel:.1%}")
```

**輸出如下：**

```bash
============================================================
響應評估結果對比
============================================================

句子視窗檢索:
  忠實度: 53.3%
  相關性: 66.7%

常規分塊檢索:
  忠實度: 0.0%
  相關性: 6.7%
```

透過這個結果可以看出，在本次實驗中“句子視窗檢索”的忠實度和相關性上均顯著優於“常規分塊檢索”。

### 1.3 核心評估維度

LlamaIndex提供了豐富的評估器，覆蓋了從檢索到響應的各個環節。上述示例中主要使用了**響應評估**維度：

- `Faithfulness` (忠實度): 評估生成的答案是否完全基於檢索到的上下文，是檢測“幻覺”現象的關鍵指標。分數越高，說明答案越可靠。
- `Relevancy` (相關性): 評估生成的答案與使用者提出的原始問題是否直接相關，確保答案切題。

此外，它還支援專門的**檢索評估**維度，如：

- `Hit Rate` (命中率): 評估檢索到的上下文中是否包含了正確的答案。
- `MRR` (平均倒數排名): 衡量找到正確答案的效率，排名越靠前得分越高。

## 二、RAGAS

RAGAS（RAG Assessment）是一個**獨立的、專注於RAG的開源評估框架**。提供了一套全面的指標來量化RAG管道的檢索和生成兩大核心環節的效能。其最顯著的特色是支援**無參考評估**，即在許多場景下無需人工標註的“標準答案”即可進行評估，極大地降低了評估成本。現對RAG管道的持續監控和改進。如果你需要一個輕量級、與具體RAG實現解耦、能夠快速對核心指標進行量化評估的工具時，`RAGAS` 是一個理想的選擇。

### 2.1 設計理念

`RAGAS` 的核心思想是透過分析問題（`question`）、生成的答案（`answer`）和檢索到的上下文（`context`）三者之間的關係，來綜合評估RAG系統的效能。它將複雜的評估問題分解為幾個簡單、可量化的維度。

### 2.2 工作流程與核心指標

RAGAS的評估流程非常簡潔，通常遵循以下步驟：

（1）**準備資料集**：根據官方文件，一個標準的評估資料集應包含 `question`（問題）、`answer`（RAG系統生成的答案）、`contexts`（檢索到的上下文）以及 `ground_truth`（標準參考答案）這四列。不過，`ground_truth` 對於計算 `context_recall` 等指標是必需的，但對於 `faithfulness` 等指標則是可選的。

（2）**執行評估**：呼叫 `ragas.evaluate()` 函式，傳入準備好的資料集和需要評估的指標列表。

（3）**分析結果**：獲取一個包含各項指標量化分數的評估報告。

其核心評估指標包括：

- `faithfulness`: 衡量生成的答案中有多少比例的資訊是可以由檢索到的上下文所支援的。
- `context_recall`: 衡量檢索到的上下文與標準答案（`ground_truth`）的對齊程度，即標準答案中的資訊是否被上下文完全“召回”。
- `context_precision`: 衡量檢索到的上下文中，訊雜比如何，即有多少是真正與回答問題相關的。
- `answer_relevancy`: 評估答案與問題的相關程度。此指標不評估事實準確性，只關注答案是否切題。

## 三、Phoenix (Arize Phoenix)

Phoenix (現由Arize維護) 是一個**開源的LLM可觀測性與評估平臺**。在RAG評估生態中，它主要扮演**生產環境中的視覺化分析與故障診斷引擎**的角色。它透過捕獲LLM應用的軌跡（Traces），提供強大的視覺化、切片和聚類分析能力，幫助開發者理解線上真實資料的表現。Phoenix 的核心價值在於**從海量生產資料中發現問題、監控效能漂移並進行深度診斷**，是連線線下評估與線上運維的關鍵橋樑。它不僅提供評估指標，更強調對LLM應用進行追蹤（Tracing）和視覺化分析，從而快速定位問題[^3]。

![phoenix](./images/6_2_1.webp)

### 3.1 核心理念

`Phoenix` 的核心是“AI可觀測性”，它透過追蹤RAG系統內部的每一步呼叫（如檢索、生成等），將整個流程視覺化。這使得開發者可以直觀地看到每個環節的輸入、輸出和耗時，並在此基礎上進行深入的評估和除錯。

### 3.2 工作原理

`Phoenix` 的工作流程是先透過基於開放標準 **OpenTelemetry** 的**程式碼插樁（`Instrumentation`）**，在 RAG 應用中整合追蹤功能，自動捕獲 LLM 呼叫、函式執行等事件；隨後在應用執行過程中持續生成**追蹤資料（`Traces`）**，記錄完整的執行鏈路；接著在本地啟動 `Phoenix` 的 Web 介面，載入並視覺化這些追蹤資料；最後在 UI 中對失敗案例或表現不佳的查詢進行篩選、鑽取，並藉助內建的**評估器（`Evals`）**完成深入的評估與除錯。

特色功能：

- **視覺化追蹤**: 將RAG的執行流程、資料和評估結果進行視覺化展示，極大地方便了問題定位。
- **根本原因分析**: 透過視覺化的介面，可以輕鬆地對錶現不佳的查詢進行切片和鑽取。
- **安全護欄 (`Guardrails`)**: 允許為應用新增保護層，防止惡意或錯誤的輸入輸出，保障生產環境安全。
- **資料探索與標註**: 提供資料探索、清洗和標註工具，幫助開發者利用生產資料反哺模型和系統最佳化。
- **與Arize平臺整合**: `Phoenix` 可以與Arize的商業平臺無縫對接，實現生產環境中對RAG系統的持續監控。

## 四、對比建議

| **工具**     | **核心機制** | **獨特技術**                | **典型應用場景** |
| ---------- | -------- | ----------------------- | --- |
| RAGAS      | LLM驅動評估  | 合成資料生成、無參考評估架構          | 對比不同RAG策略、版本迭代後的效能迴歸測試 |
| LlamaIndex | 嵌入式評估    | 非同步評估引擎、模組化BaseEvaluator | 開發過程中快速驗證單個元件或完整管道的效果 |
| Phoenix    | 追蹤分析型    | 分散式追蹤、向量聚類分析演算法            | 生產環境監控、Bad Case分析、資料漂移檢測 |

> 在實踐中，這些工具並非互斥，可以結合使用，以獲得對RAG系統更全面、多維度的洞察。

## 參考文獻

[^1]: [*LlamaIndex Evaluating*](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/)

[^2]: [*Ragas Docs*](https://docs.ragas.io/en/stable/)

[^3]: [*Arize AI Phoenix*](https://arize.com/docs/phoenix)
