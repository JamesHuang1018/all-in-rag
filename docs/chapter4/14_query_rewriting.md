# 第四節 查詢重構與分發

此前已經學習瞭如何從不同型別的資料來源（如向量資料庫、關係型資料庫）中構建查詢。然而，使用者的原始問題往往不是最優的檢索輸入。它可能過於複雜、包含歧義，或者與文件的實際措辭存在偏差。為了解決這些問題，我們需要在檢索之前對使用者的查詢進行“預處理”，這就是本節要探討的**查詢重構與分發**。

這個階段主要包含兩個關鍵技術：

1.  **查詢翻譯（Query Translation）**：將使用者的原始問題轉換成一個或多個更適合檢索的形式。
2.  **查詢路由（Query Routing）**：根據問題的性質，將其智慧地分發到最合適的資料來源或檢索器。

本節將重點介紹幾種主流的查詢翻譯技術，並簡要討論查詢路由的概念。

## 一、查詢翻譯

查詢翻譯的目標是彌合使用者自然語言提問與文件庫中儲存資訊之間的“語義鴻溝”。透過重寫、分解或擴充套件查詢，我們可以顯著提升檢索的準確率。

### 1.1 提示工程

這是最直接的查詢重構方法。透過精心設計的提示詞（Prompt），可以引導 LLM 將使用者的原始查詢改寫得更清晰、更具體，或者轉換成一種更利於檢索的敘述風格。

在第二節查詢構建的程式碼示例中，我們發現 `SelfQueryRetriever` 無法正確處理“時間最短的影片”這類需要排序或進行比較的查詢。

為了解決這個問題，可以採用一種更高階的提示工程技巧：**讓 LLM 直接構建出查詢指令**。

這種方法的思路是，要求 LLM 直接分析使用者的意圖，並生成一個結構化（例如 JSON 格式）的指令，告訴我們的程式碼應該如何操作。對於“時間最短的影片”這個問題，我們期望 LLM 能直接告訴我們：“請按‘時長’欄位進行升序排序，並返回第一條結果”。

下面，來看看如何修改程式碼來實現這一思路。我們不再使用 `SelfQueryRetriever`，而是直接與 LLM 互動，並根據其返回的指令在程式碼中執行排序邏輯。

關鍵的修改主要有兩部分：

（1）**設計一個新的提示詞（Prompt），要求 LLM 輸出 JSON 格式的排序指令。**

```python
# 使用大模型將自然語言轉換為排序指令
prompt = f"""你是一個智慧助手，請將使用者的問題轉換成一個用於排序影片的JSON指令。

你需要識別使用者想要排序的欄位和排序方向。
- 排序欄位必須是 'view_count' (觀看次數) 或 'length' (時長) 之一。
- 排序方向必須是 'asc' (升序) 或 'desc' (降序) 之一。

例如:
- '時間最短的影片' 或 '哪個影片時間最短' 應轉換為 {{"sort_by": "length", "order": "asc"}}
- '播放量最高的影片' 或 '哪個影片最火' 應轉換為 {{"sort_by": "view_count", "order": "desc"}}

請根據以下問題生成JSON指令:
原始問題: "{query}"

JSON指令:"""
```

（2）**在程式碼中呼叫 LLM，解析其返回的 JSON 指令，並執行相應的排序操作。**

```python
# ... (前略，初始化LLM客戶端)

# 請求LLM生成指令，並指定返回JSON格式
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0,
    response_format={"type": "json_object"}
)

# 解析指令並執行排序
try:
    import json
    instruction_str = response.choices[0].message.content
    instruction = json.loads(instruction_str)
    print(f"--- 生成的排序指令: {instruction} ---")

    sort_by = instruction.get('sort_by')
    order = instruction.get('order')

    if sort_by in ['length', 'view_count'] and order in ['asc', 'desc']:
        # 在程式碼中執行排序
        reverse_order = (order == 'desc')
        sorted_docs = sorted(all_documents, key=lambda doc: doc.metadata.get(sort_by, 0), reverse=reverse_order)
        # 獲取排序後的第一個結果並列印
        if sorted_docs:
            doc = sorted_docs[0]
            # ... (列印結果的程式碼)

except (json.JSONDecodeError, KeyError) as e:
    print(f"解析或執行指令失敗: {e}")
```

透過這種方式，成功地將 LLM 從一個簡單的“文字改寫員”提升為了一個能夠理解複雜意圖並生成可執行計劃的“智慧代理”，從而優雅地解決了“最值”查詢的難題。

> [完整程式碼](https://github.com/datawhalechina/all-in-rag/tree/main/code/C4/04_text_to_metadata_filter_v2.py)

### 1.2 多查詢分解 (Multi-query)

當使用者提出一個複雜的問題時，直接用整個問題去檢索可能效果不佳，因為它可能包含多個子主題或意圖。分解技術的核心思想是將這個複雜問題拆分成多個更簡單、更具體的子問題。然後，系統分別對每個子問題進行檢索，最後將所有檢索到的結果合併、去重，形成一個更全面的上下文，再交給 LLM 生成最終答案。

**示例**：
- **原始問題**：“在《流浪地球》中，劉慈欣對人工智慧和未來社會結構有何看法？”
- **分解後的子問題**：
    - “《流浪地球》中描述的人工智慧技術有哪些？”
    - “《流浪地球》中描繪的未來社會是怎樣的？”
    - “劉慈欣關於人工智慧的觀點是什麼？”

LangChain 提供了 `MultiQueryRetriever` 來完成這一過程[^1]。它在內部利用 LLM 將原始問題從不同角度分解成多個子問題，然後並行為每個子問題檢索相關文件。最後，它將所有檢索到的文件合併並去重，形成一個更全面的上下文，再傳遞給語言模型生成最終答案。透過這種策略，極大地豐富了檢索結果，在有些應用中可以有效提升後續生成環節的質量。

### 1.3 退步提示（Step-Back Prompting）

退步提示是由 Google DeepMind 團隊提出的一種旨在提升大語言模型推理能力的提示工程技巧[^2]。當面對一個細節繁多或過於具體的問題時，模型直接作答（即便是使用思維鏈）也容易出錯。退步提示透過引導模型“退後一步”來解決這個問題。

其核心流程分為兩步：

（1）**抽象化**：首先，引導 LLM 從使用者的原始具體問題中，生成一個更高層次、更概括的“退步問題”（Step-back Question）。這個退步問題旨在探尋原始問題背後的通用原理或核心概念。

（2）**推理**：接著，系統會先獲取“退步問題”的答案（例如，一個物理定律、一段歷史背景等），然後將這個通用原理作為上下文，再結合原始的具體問題，進行推理並生成最終答案。

![“退步提示”與“思維鏈”對比圖](./images/4_4_1.webp)

**示例**：
- **原始問題**：“如果理想氣體的溫度增加2倍，體積增加8倍，其壓力會如何變化？”
- **退步問題**：“這個問題背後的物理原理是什麼？”
- **推理過程**：首先回答退步問題，得到“理想氣體定律 PV=nRT”。然後基於這個定律，代入具體數值進行計算，最終得出壓力變為原來的1/4。

透過先檢索或生成高層知識，再進行具體推理，退步提示能夠幫助模型構建一個更堅實的邏輯基礎，從而提高在複雜問答場景下的準確性。

### 1.4 假設性文件嵌入 (HyDE)

假設性文件嵌入（Hypothetical Document Embeddings, HyDE）是一種無需微調即可顯著提升向量檢索質量的查詢改寫技術，由 Luyu Gao 等人在其論文中首次提出[^3]。其核心是解決一個普遍存在於檢索任務中的難題：使用者的查詢（Query）通常簡短、關鍵詞有限，而資料庫中儲存的文件則內容詳實、上下文豐富，兩者在語義向量空間中可能存在“鴻溝”，導致直接用查詢向量進行搜尋效果不佳。Zilliz 的一篇技術部落格[^4]也對該技術進行了深入淺出的解讀。

![HyDE](./images/4_4_2.webp)

HyDE 透過一種巧妙的方式來“繞過”這個問題：它不直接使用使用者的原始查詢，而是先利用一個生成式大語言模型（LLM）來生成一個“假設性”的、能夠完美回答該查詢的文件。然後，HyDE 將這個內容詳實的假設性文件進行向量化，用其生成的向量去資料庫中尋找與之最相似的真實文件。HyDE 的工作流程可以分為三個步驟：

（1）**生成**：當接收到使用者查詢時，首先呼叫一個生成式 LLM（例如，GPT-3.5）。提示該模型根據查詢生成一個詳細的、可能是理想答案的文件。這個文件不必完全符合事實，但它必須在語義上與一個好的答案高度相關。

（2）**編碼**：將上一步生成的假設性文件輸入到一個對比編碼器（如 Contriever）中，將其轉換為一個高維向量嵌入。這個向量在語義上代表了一個“理想答案”的位置。

（3）**檢索**：使用這個假設性文件的向量，在向量資料庫中執行相似性搜尋，找出與這個“理想答案”最接近的真實文件。這些被檢索出的文件將作為最終的上下文資訊。

透過這種方式，HyDE 將困難的“查詢到文件”的匹配問題，轉化為了一個相對容易的“文件到文件”的匹配問題，從而提升檢索的準確率。

## 二、查詢路由

**查詢路由（Query Routing）** 是用於最佳化複雜 RAG 系統的一項關鍵技術。當系統接入了多個不同的資料來源或具備多種處理能力時，就需要一個“智慧排程中心”來分析使用者的查詢，並動態選擇最合適的處理路徑。其本質是替代硬編碼規則，透過語義理解將查詢分發至最匹配的資料來源、處理元件或提示模板，從而提升系統的效率與答案的準確性。

### 2.1 應用場景

查詢路由的應用場景十分廣泛。

1.  **資料來源路由**：這是最常見的場景。根據查詢意圖，將其路由到不同的知識庫。例如：
    *   查詢“最新的 iPhone 有什麼功能？” -> 路由到**產品文件向量資料庫**。
    *   查詢“我上次訂購了什麼？” -> 路由到**使用者歷史SQL資料庫**（執行Text-to-SQL）。
    *   查詢“A公司和B公司的投資關係是怎樣的？” -> 路由到**企業知識圖譜資料庫**。

2.  **元件路由**：根據問題的複雜性，將其分配給不同的處理元件，以平衡成本和效果。
    *   簡單FAQ → 直接進行向量檢索，速度快、成本低。
    *   複雜操作或需要與外部API互動 → 呼叫 Agent 來執行任務。

3.  **提示模板路由**：為不同型別的任務動態選擇最優的提示詞模板，以最佳化生成效果。
    *   數學問題 → 選用包含分步思考（Step-by-Step）邏輯的提示模板。
    *   程式碼生成 → 選用專門為程式碼最佳化過的提示模板。

### 2.2 實現方法

實現查詢路由主要有兩種主流方法[^5]：

#### 2.2.1 基於LLM的意圖識別

這是最靈活的方法。透過設計一個包含路由選項的提示詞，讓大語言模型（LLM）直接對使用者的查詢進行分類，並輸出一個代表路由選擇的標籤。

![邏輯路由](./images/4_4_3.webp)

*   **實現流程**：
    1.  定義清晰的路由選項（例如，資料來源名稱、功能分類）。
    2.  LLM 分析查詢並輸出決策標籤。
    3.  程式碼根據標籤呼叫相應的檢索器或工具。

該方法的核心在於構建一個“分類-分發”的流水線。這裡以一個菜譜問答為例，系統需要根據使用者提問的菜系（川菜、粵菜或其他）呼叫不同的專家模型。

> 接下來的程式碼示例廣泛使用了 **LCEL**[^6]，它是 LangChain 中用於構建鏈（Chain）的宣告式方法。其核心是 `|` （管道）符號，可以將不同的元件（如提示、模型、解析器）串聯起來，形成一個處理流水線。例如，`prompt | llm | parser` 就清晰地定義了一個“提示->模型->解析器”的呼叫順序。這種方式不僅程式碼可讀性強，而且 LangChain 會在底層自動進行並行、非同步和流式等最佳化。

**第一步：定義分類器**

首先建立一個 `classifier_chain`，它的任務是讀取使用者問題，並利用 LLM 的理解能力給問題打上分類標籤（例如 '川菜', '粵菜', '其他'）。

```python
# 假設 llm 已經定義
classifier_prompt = ChatPromptTemplate.from_template(
    """根據使用者問題中提到的菜品，將其分類為：['川菜', '粵菜', 或 '其他']。
    不要解釋你的理由，只返回一個單詞的分類結果。
    問題: {question}"""
)
classifier_chain = classifier_prompt | llm | StrOutputParser()
```

**第二步：定義路由分支**

接著，使用 `RunnableBranch` 來定義路由規則。它就像一個 `if-elif-else` 語句，根據輸入的 `topic` 欄位來選擇執行哪一個處理鏈（`sichuan_chain`, `cantonese_chain` 或 `general_chain`）。
```python
# 假設 sichuan_chain, cantonese_chain, general_chain 已定義
router_branch = RunnableBranch(
    (lambda x: "川菜" in x["topic"], sichuan_chain),
    (lambda x: "粵菜" in x["topic"], cantonese_chain),
    general_chain  # 預設選項
)
```

**第三步：組合完整路由鏈**

最後，將分類器和路由分支組合起來。這個 `full_router_chain` 首先會並行執行兩個操作：用 `classifier_chain` 為問題生成 `topic`，同時保留原始的 `question`。然後，它將這個包含 `topic` 和 `question` 的字典傳遞給 `router_branch`，由後者根據 `topic` 做出最終的路由決策。

```python
full_router_chain = {"topic": classifier_chain, "question": lambda x: x["question"]} | router_branch

# 呼叫示例
# result = full_router_chain.invoke({"question": "麻婆豆腐怎麼做？"})
```

> [完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C4/05_llm_based_routing.py)

#### 2.2.2 嵌入相似性路由

這種方法不依賴 LLM 進行分類，延遲更低。它透過計算使用者查詢與預設的“路由示例語句”之間的向量嵌入相似度來做出決策。

![語義路由](./images/4_4_4.webp)

**第一步：定義路由描述並向量化**

為每個路由建立一個詳細的文字描述，並使用嵌入模型將其轉換為向量，供後續相似度計算使用。

```python
# 假設 embeddings 模型已經初始化
sichuan_route_prompt = "你是一位處理川菜的專家。使用者的問題是關於麻辣、辛香、重口味的菜餚，例如水煮魚、麻婆豆腐、魚香肉絲、宮保雞丁、花椒、海椒等。"
cantonese_route_prompt = "你是一位處理粵菜的專家。使用者的問題是關於清淡、鮮美、原汁原味的菜餚，例如白切雞、老火靚湯、蝦餃、雲吞麵等。"

route_prompts = [sichuan_route_prompt, cantonese_route_prompt]
route_names = ["川菜", "粵菜"]
route_prompt_embeddings = embeddings.embed_documents(route_prompts)
```

**第二步：定義目標鏈**

建立路由最終要分發到的目標處理鏈，並用一個字典 `route_map` 將路由名稱和鏈對應起來。

```python
# 假設 llm 已經定義
sichuan_chain = (
    PromptTemplate.from_template("你是一位川菜大廚。請用正宗的川菜做法，回答關於「{query}」的問題。")
    | llm
    | StrOutputParser()
)
cantonese_chain = (
    PromptTemplate.from_template("你是一位粵菜大廚。請用經典的粵菜做法，回答關於「{query}」的問題。")
    | llm
    | StrOutputParser()
)

route_map = { "川菜": sichuan_chain, "粵菜": cantonese_chain }
```

**第三步：定義路由函式**

定義一個 `route` 函式，接收使用者問題，計算與各路由描述的相似度，選擇最相似的路由並呼叫相應的處理鏈。

```python
def route(info):
    # 1. 對使用者查詢進行嵌入
    query_embedding = embeddings.embed_query(info["query"])
    
    # 2. 計算與各路由提示的餘弦相似度
    similarity_scores = cosine_similarity([query_embedding], route_prompt_embeddings)[0]
    
    # 3. 找到最相似的路由名稱
    chosen_route_index = np.argmax(similarity_scores)
    chosen_route_name = route_names[chosen_route_index]
    
    # 4. 獲取並呼叫對應的處理鏈，返回結果
    chosen_chain = route_map[chosen_route_name]
    return chosen_chain.invoke(info)
```

**第四步：組合並呼叫**

最後，將 `route` 函式包裝成一個 `RunnableLambda`，形成一個完整的、可執行的路由鏈。

```python
full_chain = RunnableLambda(route)

# 呼叫示例
# result = full_chain.invoke({"question": "如何做一碗清淡的雲吞麵？"})
```

> [完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C4/06_embedding_based_routing.py)

### 2.3 LlamaIndex 拓展

與 LangChain 類似，LlamaIndex 也提供了強大的查詢路由功能[^7]，其思路是將不同的資料來源或查詢策略包裝為“工具（Tool）”，然後透過一個“路由器（Router）”來為使用者查詢動態選擇最合適的工具。實現方式與 LangChain 有異曲同工之處：

*   **基於LLM的意圖識別**：這是 LlamaIndex 的主要實現方式。透過 `RouterQueryEngine` 來管理一組 `QueryEngineTool`。每個 `Tool` 都包含一個查詢引擎和一段描述其功能的文字。路由器會利用一個 `Selector`（如 `LLMSingleSelector` 或更穩定的 `PydanticSingleSelector`）來讓 LLM 根據工具的描述文字和使用者問題進行語義匹配，從而選擇一個或多個最合適的工具來執行。

*   **嵌入相似性路由**：LlamaIndex 沒有提供直接基於向量相似度計算的獨立路由元件。它的“語義路由”是融合在基於 LLM 的意圖識別中的——即讓 LLM 理解每個 `Tool` 描述的 *語義*，並據此做出決策。這種方式更靈活，能夠處理更復雜的路由邏輯，而不僅僅是文字相似度匹配。

## 參考文獻

[^1]: [*How to use the MultiQueryRetriever*](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)

[^2]: [Zheng, H. S. et al. (2023). *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*](https://arxiv.org/abs/2310.06117).

[^3]: [Gao, L. et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels*](https://arxiv.org/abs/2212.10496).

[^4]: [*使用假設性文件嵌入（HyDE）改進資訊檢索和 RAG*](https://zilliz.com.cn/blog/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings).

[^5]: [*How to route between sub-chains*](https://python.langchain.com/docs/how_to/routing/).

[^6]: [*LangChain Expression Language*](https://python.langchain.com/docs/concepts/lcel/).

[^7]: [*LlamaIndex Routing*](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/).