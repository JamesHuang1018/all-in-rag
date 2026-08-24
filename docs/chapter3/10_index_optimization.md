# 第五節 索引最佳化

在上一章的文字分塊部分，已經簡單介紹了一些索引最佳化的策略。本節將基於LlamaIndex的高效能生產級RAG構建方案[^1]，對索引最佳化進行更深入的探討。

## 一、上下文擴充套件

在RAG系統中，常常面臨一個權衡問題：使用小塊文字進行檢索可以獲得更高的精確度，但小塊文字缺乏足夠的上下文，可能導致大語言模型（LLM）無法生成高質量的答案；而使用大塊文字雖然上下文豐富，卻容易引入噪音，降低檢索的相關性。為了解決這一矛盾，LlamaIndex 提出了一種實用的索引策略——**句子視窗檢索（Sentence Window Retrieval）**[^2]。該技術巧妙地結合了兩種方法的優點：它在檢索時聚焦於高度精確的單個句子，在送入LLM生成答案前，又智慧地將上下文擴充套件回一個更寬的“視窗”，從而同時保證檢索的準確性和生成的質量。

### 1.1 主要思路

句子視窗檢索的思想可以概括為：**為檢索精確性而索引小塊，為上下文豐富性而檢索大塊**。

其工作流程如下：

（1）**索引階段**：在構建索引時，文件被分割成**單個句子**。每個句子都作為一個獨立的“節點（Node）”存入向量資料庫。同時，每個句子節點都會在後設資料（metadata）中儲存其**上下文視窗**，即該句子原文中的前N個和後N個句子。這個視窗內的文字不會被索引，僅僅是作為後設資料儲存。

（2）**檢索階段**：當使用者發起查詢時，系統會在所有**單一句子節點**上執行相似度搜尋。因為句子是表達完整語義的最小單位，所以這種方式可以非常精確地定位到與使用者問題最相關的核心資訊。

（3）**後處理階段**：在檢索到最相關的句子節點後，系統會使用一個名為 `MetadataReplacementPostProcessor` 的後處理模組。該模組會讀取到檢索到的句子節點的後設資料，並用後設資料中儲存的**完整上下文視窗**來替換節點中原來的單一句子內容。

（4）**生成階段**：最後，這些被替換了內容的、包含豐富上下文的節點被傳遞給LLM，用於生成最終的答案。

### 1.2 程式碼實現

下面透過 LlamaIndex 官網的示例，來演示如何實現句子視窗檢索，並與常規的檢索方法進行對比。該示例將載入一份PDF格式的IPCC氣候報告，並就其中的專業問題進行提問。

核心程式碼如下：

```python
# 假設 Settings.llm 和 Settings.embed_model 已經預先配置好

# 1. 載入文件
documents = SimpleDirectoryReader(
    input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# 2. 建立節點與構建索引
# 2.1 句子視窗索引
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
sentence_nodes = node_parser.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(sentence_nodes)
```

根據 LlamaIndex 的底層原始碼，`SentenceWindowNodeParser` 的核心邏輯位於 `build_window_nodes_from_documents` 方法中。其實現過程可以分解為以下幾個關鍵步驟：

（1）**句子切分 (`sentence_splitter`)** ：解析器首先接收一個文件（`Document`），然後呼叫 `self.sentence_splitter(doc.text)` 方法。這個 `sentence_splitter` 是一個可配置的函式，預設為 `split_by_sentence_tokenizer`，它負責將文件的全部文字精確地切分成一個句子列表（`text_splits`）。

（2）**建立基礎節點 (`build_nodes_from_splits`)** ：切分出的 `text_splits` 列表被傳遞給 `build_nodes_from_splits` 工具函式。這個函式會為列表中的**每一個句子**都建立一個獨立的 `TextNode`。此時，每個 `TextNode` 的 `text` 屬性就是這個句子的內容。

（3）**構建視窗並填充後設資料 (主要迴圈)** ：接下來，解析器會遍歷所有新建立的 `TextNode`。對於位於第 `i` 個位置的節點，它會執行以下操作：

*   **定位視窗**：透過列表切片 `nodes[max(0, i - self.window_size) : min(i + self.window_size + 1, len(nodes))]` 來獲取一個包含中心句子及其前後 `window_size`（預設為3）個鄰近節點的列表（`window_nodes`）。這個切片操作很巧妙地處理了文件開頭和結尾的邊界情況。
*   **組合視窗文字**：將 `window_nodes` 列表中所有節點的 `text`（即所有在視窗內的句子）用空格拼接成一個長字串。
*   **填充後設資料**：將上一步生成的長字串（完整的上下文視窗）存入當前節點（第`i`個節點）的後設資料中，鍵為 `self.window_metadata_key`（預設為 `"window"`）。同時，也會將節點自身的文字（原始句子）存入後設資料，鍵為 `self.original_text_metadata_key`（預設為 `"original_text"`）。

4.  **設定後設資料排除項**：這是一個非常關鍵的細節。在填充完後設資料後，程式碼會執行 `node.excluded_embed_metadata_keys.extend(...)` 和 `node.excluded_llm_metadata_keys.extend(...)`。這行程式碼的作用是告訴後續的嵌入模型和LLM，在處理這個節點時，**應當忽略** `"window"` 和 `"original_text"` 這兩個後設資料欄位。這確保了只有單個句子的純淨文字被用於生成向量嵌入，從而保證了檢索的高精度。而 `"window"` 欄位僅供後續的 `MetadataReplacementPostProcessor` 使用。

透過以上步驟，`SentenceWindowNodeParser` 最終返回一個 `TextNode` 列表。列表中的每個節點都代表一個獨立的句子，其 `text` 屬性用於精確檢索，而其 `metadata` 中則“隱藏”了用於生成答案的豐富上下文視窗。

```python
# 2.2 常規分塊索引 (基準)
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)

# 3. 構建查詢引擎
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# 4. 執行查詢並對比結果
query = "What are the concerns surrounding the AMOC?"
print(f"查詢: {query}\n")

print("--- 句子視窗檢索結果 ---")
window_response = sentence_query_engine.query(query)
print(f"回答: {window_response}\n")

print("--- 常規檢索結果 ---")
base_response = base_query_engine.query(query)
print(f"回答: {base_response}\n")
```

（1）**構建句子視窗索引**：這一步利用了 `SentenceWindowNodeParser`。它將文件解析為以單個句子為單位的 `Node`，同時將包含上下文的“視窗”文字（預設為前後各3個句子）儲存在每個 `Node` 的後設資料中。這一步是實現“為檢索精確性而索引小塊”思想的關鍵。

（2）**構建查詢引擎與後處理**：查詢引擎的構建是實現“為生成質量而擴充套件上下文”的關鍵。

*   在建立 `sentence_query_engine` 時，配置中加入了一個重要的後處理器 `MetadataReplacementPostProcessor`。
*   它的作用是：當檢索器根據使用者查詢找到最相關的節點（也就是單個句子）後，這個後處理器會立即介入。
*   它會從該節點的後設資料中讀取出預先儲存的完整“視窗”文字，並用它**替換**掉節點中原來的單個句子內容。
*   這樣，最終傳遞給大語言模型的就不再是孤立的句子，而是包含豐富上下文的完整文字段落，從而確保了生成答案的質量和連貫性。

我們向兩個引擎提出的問題是：“關於大西洋經向翻轉環流（AMOC），人們主要擔憂什麼？” (What are the concerns surrounding the AMOC?)。

**程式碼輸出如下：**
```bash
查詢: What are the concerns surrounding the AMOC?

--- 句子視窗檢索結果 ---
回答: The Atlantic Meridional Overturning Circulation (AMOC) is projected to decline over the 21st century with high confidence, though there is low confidence in quantitative projections of this decline. Observational records since the mid-2000s are too short to determine the relative contributions of internal variability, natural forcing, and anthropogenic forcing to AMOC changes. Additionally, there is low confidence in reconstructed and modeled AMOC changes for the 20th century due to limited agreement in quantitative trends. While an abrupt collapse before 2100 is not expected, the decline could have significant implications for global climate patterns.

--- 常規檢索結果 ---
回答: The concerns surrounding the Atlantic Meridional Overturning Circulation (AMOC) primarily involve its projected decline over the 21st century across all Shared Socioeconomic Pathway (SSP) scenarios. While an abrupt collapse before 2100 is not expected, there is high confidence in this decline, though quantitative projections remain uncertain. Observational records since the mid-2000s are too short to clearly distinguish the contributions of internal variability, natural forcing, and anthropogenic forcing to these changes. This uncertainty highlights the need for further research to better understand and predict AMOC behavior and its broader climate impacts.
```

從輸出結果中可以觀察到：

*   **兩個答案都抓住了核心**：兩個引擎都正確地識別出，對AMOC的主要擔憂是其在21世紀預計的衰退。
*   **句子視窗檢索的答案更詳盡、更連貫**：句子視窗檢索的回答不僅指出了衰退的趨勢，還補充了關於“定量預測的置信度低”、“觀測記錄時間過短”、“20世紀重建和模擬的變化置信度低”等多個維度的細節。這使得答案的資訊量更大，上下文更完整，更像一個綜述。
*   **常規檢索的答案相對寬泛**：常規檢索的回答雖然正確，但內容相對概括，最後以“需要進一步研究”這樣較為籠同的結論收尾。

這種差異正是句子視窗檢索策略優勢的體現。它透過“精確檢索小文字塊（單個句子），再擴充套件上下文（句子視窗）”的方式，為大語言模型提供了高度相關且資訊豐富的上下文，從而生成了質量更高的答案。

> [完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C3/05_sentence_window_retrieval.py)

## 二、結構化索引

隨著知識庫的規模不斷擴大（例如，包含數百個PDF檔案），傳統的RAG方法（即對所有文字塊進行top-k相似度搜尋）會遇到瓶頸。當一個查詢可能只與其中一兩個文件相關時，在整個文件庫中進行無差別的向量搜尋，不僅效率低下，還容易被不相關的文字塊干擾，導致檢索結果不精確。

為了解決這個問題，一個有效的方法是利用**結構化索引**。其原理是在索引文字塊的同時，為其附加結構化的**後設資料（Metadata）**。這些後設資料可以是任何有助於篩選和定位資訊的標籤，例如：

*   檔名
*   文件建立日期
*   章節標題
*   作者
*   任何自定義的分類標籤

![結構化索引](./images/3_5_1.webp)

實際上，在第二章“文字分塊”中介紹的**基於文件結構的分塊**方法，就是實現結構化索引的一種前置步驟。例如，在使用 `MarkdownHeaderTextSplitter` 時，分塊器會自動將Markdown文件的各級標題（如 `Header 1`, `Header 2` 等）提取並存入每個文字塊的後設資料中。這些標題資訊就是非常有價值的結構化資料，可以直接用於後續的後設資料過濾。

透過這種方式，可以在檢索時實現“後設資料過濾”和“向量搜尋”的結合。例如，當使用者查詢“請總結一下2023年第二季度財報中關於AI的論述”時，系統可以：

（1）**後設資料預過濾**：首先透過後設資料篩選，只在 `document_type == '財報'`、`year == 2023` 且 `quarter == 'Q2'` 的文件子集中進行搜尋。

（2）**向量搜尋**：然後，在經過濾的、範圍更小的文字塊集合中，執行針對查詢“關於AI的論述”的向量相似度搜尋。

這種“先過濾，再搜尋”的策略，能夠極大地縮小檢索範圍，顯著提升大規模知識庫場景下RAG應用的檢索效率和準確性。LlamaIndex 提供了包括“自動檢索”（Auto-Retrieval）在內的多種工具來支援這種結構化的檢索正規化。

### 2.1 程式碼實現：基於多表格的遞迴檢索

在更復雜的場景中，結構化資料可能分佈在多個來源中，例如一個包含多個工作表（Sheet）的 Excel 檔案，每個工作表都代表一個獨立的表格。在這種情況下，需要一種更強大的策略：**遞迴檢索**[^3]。它能實現“路由”功能，先將查詢引導至正確的知識來源（正確的表格），然後再在該來源內部執行精確查詢。

下面使用一個包含多個工作表的電影資料 Excel 檔案（`movie.xlsx`）來演示，其中每個工作表（如 `年份_1994`, `年份_2002` 等）都儲存了對應年份的電影資訊。

```python
# 1. 為每個工作表建立查詢引擎和摘要節點
excel_file = '../../data/C3/excel/movie.xlsx'
xls = pd.ExcelFile(excel_file)

df_query_engines = {}
all_nodes = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    # 為當前工作表建立一個 PandasQueryEngine
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)
    # 為當前工作表建立一個摘要節點（IndexNode）
    year = sheet_name.replace('年份_', '')
    summary = f"這個表格包含了年份為 {year} 的電影資訊，可以用來回答關於這一年電影的具體問題。"
    node = IndexNode(text=summary, index_id=sheet_name)
    all_nodes.append(node)
    # 儲存工作表名稱到其查詢引擎的對映
    df_query_engines[sheet_name] = query_engine

# 2. 建立頂層索引（只包含摘要節點）
vector_index = VectorStoreIndex(all_nodes)

# 3. 建立遞迴檢索器
vector_retriever = vector_index.as_retriever(similarity_top_k=1)
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_query_engines,
    verbose=True,
)

# 4. 建立查詢引擎
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# 5. 執行查詢
query = "1994年評分人數最多的電影是哪一部？"
print(f"查詢: {query}")
response = query_engine.query(query)
print(f"回答: {response}")
```

1.  **建立 PandasQueryEngine** ：遍歷 Excel 中的每個工作表，為每個工作表（即一個獨立的 DataFrame）都例項化一個 `PandasQueryEngine`。其強大之處在於，它能將關於表格的自然語言問題（如“評分人數最多的是哪個”）轉換成實際的 Pandas 程式碼（如 `df.sort_values('評分人數').iloc[-1]`）來執行。
2.  **建立摘要節點 (`IndexNode`)** ：對每個工作表，都建立一個 `IndexNode`，其內容是關於這個表格的一段摘要文字。這個節點將作為頂層檢索的“指標”。
3.  **構建頂層索引** ：使用所有建立的 `IndexNode` 構建一個 `VectorStoreIndex`。這個索引不包含任何表格的詳細資料，只包含指向各個表格的“指標”資訊。
4.  **建立 `RecursiveRetriever`** ：這是實現遞迴檢索的核心。將其配置為：
    *   `retriever_dict`: 指定頂層的檢索器，即在摘要節點中進行檢索的 `vector_retriever`。
    *   `query_engine_dict`: 提供一個從節點 ID（即工作表名稱）到其對應查詢引擎的對映。當頂層檢索器匹配到某個摘要節點後，遞迴檢索器就知道該呼叫哪個 `PandasQueryEngine` 來處理後續查詢。

**執行結果：**

```bash
查詢: 1994年評分人數最少的電影是哪一部？
> Retrieving with query id None: 1994年評分人數最少的電影是哪一部？
> Retrieved node with id, entering: 年份_1994
> Retrieving with query id 年份_1994: 1994年評分人數最少的電影是哪一部？
> Pandas Instructions:
```
df[df['年份'] == 1994].nsmallest(1, '評分人數')['電影名稱'].iloc[0]
```
> Pandas Output: 燃情歲月
回答: 燃情歲月
```

從輸出中可以清晰地看到遞迴檢索的完整流程：

（1）**頂層路由**：`Retrieving with query id None`，系統首先在頂層的摘要索引中檢索，根據問題“1994年...”匹配到了摘要節點 `年份_1994`。

（2）**進入子層**：`Retrieved node with id, entering: 年份_1994`，系統決定進入與“年份_1994”這個工作表關聯的查詢引擎。

（3）**子層查詢**：`Retrieving with query id 年份_1994`，`PandasQueryEngine` 接管查詢，並將問題傳送給 LLM，讓其生成 Pandas 程式碼。

（4）**程式碼生成與執行**：LLM 生成了 `df[df['年份'] == 1994].nsmallest(1, '評分人數')['電影名稱'].iloc[0]`，引擎執行後得到輸出 `燃情歲月`。

> [完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C3/06_recursive_retrieval.py)

> ⚠️ **重要安全警告**：實際上在 LlamaIndex 的官網有提到，`PandasQueryEngine` 是一個實驗性功能，具有潛在的安全風險。它的工作原理是讓 LLM 生成 Python 程式碼，然後使用 `eval()` 函式在本地執行。這意味著，在沒有嚴格沙箱隔離的環境下，理論上可能執行任意程式碼。**因此，強烈不建議在生產環境中使用此工具**。

### 2.2 另一種實現方式

鑑於 `PandasQueryEngine` 的安全風險，還可以採用一種更安全的方式來實現類似的多表格查詢，思路是**將路由和檢索徹底分離**。

這種改進方法的具體步驟如下：

（1）**建立兩個獨立的向量索引**：

*   **摘要索引（用於路由）**：為每個Excel工作表（例如，“1994年電影資料”）建立一個非常簡短的摘要性`Document`，例如：“此文件包含1994年的電影資訊”。然後，用所有這些摘要文件構建一個輕量級的向量索引。這個索引的唯一目的就是充當“路由器”。
*   **內容索引（用於問答）**：將每個工作表的實際資料（例如，整個表格）轉換為一個大的文字`Document`，併為其附加一個關鍵的後設資料標籤，如 `{"sheet_name": "年份_1994"}`。然後，用所有這些包含真實內容的文件構建一個向量索引。

（2）**執行兩步查詢**：

*   **第一步：路由**。當使用者提問（例如，“1994年評分人數最少的電影是哪一部？”）時，首先在“摘要索引”中進行檢索。由於問題中的“1994年”與“此文件包含1994年的電影資訊”這個摘要高度相關，檢索器會快速返回其對應的後設資料，告訴系統目標是 `年份_1994` 這個工作表。
*   **第二步：檢索**。拿到 `年份_1994` 這個目標後，系統會在“內容索引”中進行檢索，但這次會附加一個**後設資料過濾器**（`MetadataFilter`），強制要求只在 `sheet_name == "年份_1994"` 的文件中進行搜尋。這樣，LLM就能在正確的、經過篩選的資料範圍內找到問題的答案。

透過這種“先路由，後用後設資料過濾檢索”的方式，既實現了跨多個資料來源的查詢能力，又避免了執行程式碼的安全隱患。LlamaIndex 官方也提供了類似的結構化分層檢索[^4]可以參考。

> [完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C3/07_recursive_retrieval_v2.py)

## 題外話：關於框架

> **有些人可能疑惑，為什麼本教程不專注於一個框架（如 LlamaIndex 或 LangChain），而是混合使用，甚至造輪子？**

框架是加速開發的強大工具，是幫助我們快速跨越技術鴻溝的“橋樑”。但任何橋樑都有其設計邊界和侷限性。我們的目標不是成為一個熟練的“過橋者”，而是成為一個懂得如何設計和建造橋樑的“工程師”。

因此，本教程選擇的路徑是：

（1）**以原理為主**：我們優先關心的是“它是如何工作的？”而不是“我該呼叫哪個函式？”。新框架在誕生，老框架在迭代（當然不是筆者偷懶沒更新 langchain🤫），但只要理解了底層的思想，我們將能更快地掌握任何現有或未來的框架。

（2）**擁抱靈活性**：真實世界的業務需求往往比框架預設的場景更復雜。當框架無法滿足需求，或者像本節使用的 `PandasQueryEngine` 那樣存在安全隱患時，懂得原理的話，就有能力去修改它，或者像本節的示例一樣，用更底層的模組組合出更安全、合適的解決方案。

（3）**培養解決問題的能力**：只學習使用框架，好比是照著菜譜做菜，雖然能快速復刻出指定的菜餚，但一旦缺少某個食材或遇到意外情況，就可能束手無策。而理解原理，則像是學會了烹飪的精髓。這讓你不僅能輕鬆地做出各種美食，還能創造新菜式。

如果你希望深入某個框架的細節，它的官方文件永遠是最好、最權威的學習資料。而本教程的使命，是幫助你建立起關於 RAG 的堅實知識體系，讓你無論面對何種工具，都能遊刃有餘。


## 參考文獻

[^1]: [*Building Performant RAG Applications for Production*](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)

[^2]: [*LlamaIndex - Sentence Window Retrieval*](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/#metadata-replacement-node-sentence-window)

[^3]: [*Recursive Retriever + Query Engine Demo*](https://docs.llamaindex.ai/en/stable/examples/query_engine/pdf_tables/recursive_retriever)

[^4]: [*Structured Hierarchical Retrieval*](https://docs.llamaindex.ai/en/stable/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval/)