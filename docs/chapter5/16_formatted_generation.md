# 第一節 格式化生成

從大語言模型（LLM）那裡獲得一段非結構化的文字在應用中常常不滿足實際需求。為了實現更復雜的邏輯、與外部工具互動或以使用者友好的方式展示資料，需要模型能夠輸出具有特定結構的資料，例如 JSON 或 XML。

本節將討論實現格式化生成的幾種主流方法，包括 LangChain、LlamaIndex 等框架內建的解決方案，不依賴框架的實現思路，以及一種更強大的技術——Function Calling。

> 在生成階段，提示詞工程也是一個重要的部分。但是因為在前面幾個章節中已經有了比較多的介紹，所以本章就不再贅述了。

## 一、為什麼需要格式化生成？

先來看幾個具體的應用場景：

- **RAG 驅動的電商客服**：當使用者詢問“推薦幾款適合程式設計師的鍵盤”時，我們希望 LLM 返回一個包含產品名稱、價格、特性和購買連結的 JSON 列表，而不是一段描述性文字，以便前端直接渲染成商品卡片。
- **自然語言轉 API 呼叫**：使用者說“幫我查一下明天從上海到北京的航班”，系統需要將這句話解析成一個結構化的 API 請求，如 `{"departure": "上海", "destination": "北京", "date": "2025-07-18"}`。
- **資料自動提取**：從一篇新聞文章中，自動抽取出事件、時間、地點、涉及人物等關鍵資訊，並以結構化形式存入資料庫。

在這些場景中，格式化生成是連線 LLM 的自然語言理解能力和下游應用程式的程式化邏輯之間的關鍵。

## 二、格式化生成的實現方法

> [完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C5/01_pydantic.py)

### 2.1 Output Parsers

LangChain 提供了一個強大的元件——`OutputParsers`（輸出解析器），專門用於處理 LLM 的輸出，其主要思想是在傳送給 LLM 的提示（Prompt）中自動注入一段關於如何格式化輸出的指令，並在得到結果後將 LLM 返回的純文字字串解析成預期的結構化資料（如 Python 物件）。

LangChain 提供了多種開箱即用的解析器，例如：

- **StrOutputParser**：最基礎的輸出解析器，它簡單地將 LLM 的輸出作為字串返回。
- **JsonOutputParser**：可以解析包含巢狀結構和列表的複雜 JSON 字串。
- **PydanticOutputParser**：透過與 Pydantic 模型結合，可以實現對輸出格式最嚴格的定義和驗證。

接下來透過一個具體的程式碼示例，重點分析 `PydanticOutputParser` 的工作原理。它透過將使用者定義的 Pydantic 資料模型轉換為詳細的格式指令，並注入到提示詞中，來引導 LLM 生成嚴格符合該資料結構的 JSON 輸出。最後再將模型返回的 JSON 字串安全地解析為 Pydantic 物件例項。

```python
# (此處省略了匯入和 LLM 初始化程式碼)

# 1. 定義期望的資料結構
class PersonInfo(BaseModel):
    """用於儲存個人資訊的資料結構。"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年齡")
    skills: List[str] = Field(description="技能列表")

# 2. 基於 Pydantic 模型，建立解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. 建立提示模板，注入格式指令
prompt = PromptTemplate(
    template="請根據以下文字提取資訊。\n{format_instructions}\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. 建立處理鏈 (假定 llm 已被初始化)
chain = prompt | llm | parser

# 5. 執行呼叫
text = "張三今年30歲，他擅長Python和Go語言。"
result = chain.invoke({"text": text})

# 6. 列印結果
print(result)
# name='張三' age=30 skills=['Python', 'Go語言']
```

（1）**定義資料模型**：使用 Pydantic 的 `BaseModel` 定義 `PersonInfo` 類，這不僅是一個 Python 物件，更是一個清晰的資料結構規範（Schema）。`Field` 中的 `description` 描述文字將直接作為指令提供給大模型，因此其表述需要清晰準確。

（2）**生成格式指令**：當 `PydanticOutputParser` 例項化後，其 `get_format_instructions()` 方法會執行以下操作：
- 呼叫 Pydantic 模型的 `.model_json_schema()` 方法，提取出該資料結構的 JSON Schema 定義。
- 對該 Schema 進行簡化，並將其嵌入到一個預設的、指導性的提示模板中。這個模板明確要求 LLM 輸出一個符合該 Schema 的 JSON 物件。

（3）**構建並執行呼叫鏈**：透過 LangChain 表示式語言（LCEL），將 `prompt`、`llm` 和 `parser` 連結起來。當呼叫鏈被觸發時：
- `prompt` 會將使用者輸入（`text`）和上一步生成的格式指令（`format_instructions`）組合成最終的提示，傳送給 `llm`。
- `llm` 根據這個包含嚴格格式要求的提示，生成一個 JSON 格式的字串。

（4）**解析與驗證**：`PydanticOutputParser` 接收到 LLM 返回的字串後，會執行一個兩步解析過程：
- 首先，它繼承自 `JsonOutputParser`，會將 LLM 輸出的文字字串解析成一個 Python 字典。
- 然後，最關鍵的一步，它會使用 `PersonInfo.model_validate()` 方法，用定義的資料模型來驗證這個字典。如果字典的鍵和值型別都符合 `PersonInfo` 的定義，解析器就會返回一個 `PersonInfo` 的例項物件；如果驗證失敗，則會丟擲一個 `OutputParserException` 異常。

### 2.2 LlamaIndex 的輸出解析

LlamaIndex 的輸出解析與生成過程緊密結合，主要體現在兩大核心元件中，分別是響應合成（Response Synthesis）和結構化輸出（Structured Output）。

在 RAG 流程中，檢索器召回一系列相關的文字塊（Nodes）後，並不是簡單地將它們拼接起來。響應合成器（Response Synthesizer）負責接收這些文字塊和原始查詢，並以一種更智慧的方式將它們呈現給 LLM 以生成最終答案。例如，它可以逐塊處理資訊並迭代地最佳化答案（`refine` 模式），或者將儘可能多的文字塊壓縮排單次 LLM 呼叫中（`compact` 模式）。這個階段的預設目標是生成一段高質量的**文字**回答。

當需要 LLM 返回結構化資料（如 JSON）而非純文字時，LlamaIndex 主要使用 **Pydantic 程式（Pydantic Programs）**。這與 LangChain 的 `PydanticOutputParser` 思想一致：

- **定義 Schema**：開發者首先定義一個 Pydantic 模型，明確所需輸出的資料結構、欄位和型別。
- **引導生成**：LlamaIndex 會將這個 Pydantic 模型轉換成 LLM 能理解的格式指令。如果底層的 LLM 支援 Function Calling，LlamaIndex 會優先使用該功能以獲得更可靠的結構化輸出。如果不支援，它會回退到將 JSON Schema 注入到提示詞中的方法。
- **解析驗證**：最後，LLM 返回的輸出會被自動解析並用 Pydantic 模型進行驗證，確保其型別和結構完全正確，最終返回一個 Pydantic 物件例項。

### 2.3 不依賴框架的簡單實現思路

如果不想依賴特定的框架，也可以透過提示工程的技巧來實現格式化生成。

主要思路是在提示中給出清晰、明確的指令和示例。以下是一些實用技巧：

- **明確要求 JSON 格式**：在提示中直接、強硬地要求模型“必須返回一個 JSON 物件”、“不要包含任何解釋性文字，只返回 JSON”。
- **提供 JSON Schema**：在提示中給出你想要的 JSON 物件的模式（Schema），描述每個鍵的含義和資料型別。
- **提供 few-shot 示例**：給出 1-2 個“使用者輸入 -> 期望的 JSON 輸出”的完整示例，讓模型學習輸出的格式和風格。
- **使用語法約束**：對於一些本地部署的開源模型（如透過 `llama.cpp` 執行的模型），可以使用 GBNF (GGML BNF) 等語法檔案來強制約束模型的輸出，確保其生成的每一個 token 都嚴格符合預定義的 JSON 語法。這是最嚴格也是最可靠的非 Function Calling 方法。

## 三、Function Calling

Function Calling（或稱 Tool Calling）是近年來 LLM 領域的一個重要進展，提升了模型與外部世界互動和生成結構化資料的能力。

> [本節完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C5/02_function_calling_example.py)

### 3.1 概念與工作流程

Function Calling 的本質是一個多輪對話流程，讓模型、程式碼和外部工具（如 API）協同工作。其核心工作流如下：

（1）**定義工具**：首先，在程式碼中以特定格式（通常是 JSON Schema）定義好可用的工具，包括工具的名稱、功能描述、以及需要的引數。

（2）**使用者提問**：使用者發起一個需要呼叫工具才能回答的請求。

（3）**模型決策**：模型接收到請求後，分析使用者的意圖，並匹配最合適的工具。它不會直接回答，而是返回一個包含 `tool_calls` 的特殊響應。這個響應相當於一個指令：“請呼叫某某工具，並使用這些引數”。

（4）**程式碼執行**：應用接收到這個指令，解析出工具名稱和引數，然後**在程式碼層面實際執行**這個工具（例如，呼叫一個真實的天氣 API）。

（5）**結果反饋**：將工具的執行結果（例如，從 API 獲取的真實天氣資料）包裝成一個 `role` 為 `tool` 的訊息，再次傳送給模型。

（6）**最終生成**：模型接收到工具的執行結果後，結合原始問題和工具返回的資訊，生成最終的、自然的語言回答。

### 3.2 Function Calling 實踐

接下來，直接使用 `openai` 的例子，來展示上述流程。


```python
# 1. 定義工具
tools = [...] 

# 2. 使用者提問
messages = [{"role": "user", "content": "杭州今天天氣怎麼樣？"}]
message = send_messages(messages, tools=tools)

# 3. 程式碼執行：模擬呼叫天氣API，並將結果新增到訊息歷史
if message.tool_calls:
    tool_call = message.tool_calls[0]
    messages.append(message) # 新增模型的回覆
    tool_output = "24℃，晴朗" # 模擬API結果
    messages.append({
        "role": "tool", 
        "tool_call_id": tool_call.id, 
        "content": tool_output
    }) # 新增工具執行結果

    # 4. 第二次呼叫 (`Tool -> Model`)：將工具結果返回給模型，獲取最終回答
    final_message = send_messages(messages, tools=tools)
    print(final_message.content)
```

關鍵步驟：

（1）**定義 `tools`**：用一個列表包含了所有可用的函式定義。每個定義都是一個 JSON 物件，嚴格描述了函式的名稱 (`name`)、功能 (`description`) 和引數 (`parameters`)。這個描述的質量直接決定了模型能否正確選擇和使用工具。

（2）**第一次呼叫 (`User -> Model`)**：將使用者的原始問題（`"role": "user"`）和 `tools` 列表一同傳送給模型。

（3）**處理 `tool_calls`**：檢查模型的響應中是否包含 `tool_calls`。如果包含，就說明模型決定使用工具。解析出函式名和引數，並**模擬執行**（在真實場景中，這裡會是真實的 API 呼叫）。

（4）**第二次呼叫 (`Tool -> Model`)**：將原始的使用者問題、模型的工具呼叫響應，以及模擬執行後得到的工具結果（`"role": "tool"`），一同打包成新的對話歷史，再次傳送給模型。

（5）**獲取最終答案**：模型在看到工具的執行結果後，就能用自然語言回答使用者最初的問題了。

### 3.3 Function Calling 的優勢

相比於單純透過提示工程“請求”模型輸出 JSON，Function Calling 的優勢在於：

- **可靠性更高**：這是模型原生支援的能力，相比於解析可能格式不穩定的純文字輸出，這種方式得到的結構化資料更穩定、精確。
- **意圖識別**：它不僅僅是格式化輸出，更包含了“意圖到函式的對映”。模型能根據使用者問題主動選擇最合適的工具。
- **與外部世界互動**：它是構建能執行實際任務的 AI 代理（Agent）的核心基礎，讓 LLM 可以查詢資料庫、呼叫 API、控制智慧家居等。