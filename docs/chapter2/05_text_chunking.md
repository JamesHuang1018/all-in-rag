# 第二節 文字分塊

## 一、理解文字分塊

文字分塊（Text Chunking）是構建 RAG 流程的關鍵步驟。它的原理是將載入後的長篇文件，切分成更小、更易於處理的單元。這些被切分出的文字塊，是後續向量檢索和模型處理的**基本單位**。

![文字分塊示意圖](./images/2_2_1.webp)

## 二、文字分塊重要性

### 2.1 滿足模型上下文限制

將文字分塊的首要原因，是為了適應 RAG 系統中兩個核心元件的硬性限制：

-   **嵌入模型 (Embedding Model)**: 負責將文字塊轉換為向量。這類模型有嚴格的輸入長度上限。例如，許多常用的嵌入模型（如 `bge-base-zh-v1.5`）的上下文視窗為512個token。任何超出此限制的文字塊在輸入時都會被截斷，導致資訊丟失，生成的向量也無法完整代表原文的語義。因此，文字塊的大小**必須**小於等於嵌入模型的上下文視窗。

-   **大語言模型 (LLM)**: 負責根據檢索到的上下文生成答案。LLM同樣有上下文視窗限制（儘管通常比嵌入模型大得多，從幾千到上百萬token不等）。檢索到的所有文字塊，連同使用者問題和提示詞，都必須能被放入這個視窗中。如果單個塊過大，可能會導致只能容納少數幾個相關的塊，限制了LLM回答問題時可參考的資訊廣度。

因此，分塊是確保文字能夠被兩個模型完整、有效處理的基礎。

### 2.2 為何“塊”不是越大越好

假設嵌入模型最多能處理 8192 個 token，是否應該把塊切得儘可能大（比如8000個token）呢？答案是否定的。**塊的大小並非越大越好**，過大的塊會嚴重影響RAG系統的效能。

#### 2.2.1 嵌入過程中的資訊損失

大多數嵌入模型都基於 Transformer 編碼器。其工作流程大致如下：

- **分詞 (Tokenization)**: 將輸入的文字塊分解成一個個 token。
- **向量化 (Vectorization)**: Transformer 為**每個 token** 生成一個高維向量表示。
- **池化 (Pooling)**: 透過某種方法（如取 `[CLS]` 位的向量、對所有token向量求平均 `mean pooling` 等），將所有 token 的向量**壓縮**成一個**單一的向量**，這個向量代表了整個文字塊的語義。

> `[CLS]` 是BERT等Transformer模型在輸入文字開頭新增的特殊標記，它透過自注意力機制動態聚合整個序列的上下文資訊，其最終向量被訓練用作代表全域性語義的嵌入。

在這個`壓縮`過程中，資訊損失是不可避免的。一個768維的向量需要概括整個文字塊的所有資訊。**文字塊越長，包含的語義點越多，這個單一向量所承載的資訊就越稀釋**，導致其表示變得籠統，關鍵細節被模糊化，從而降低了檢索的精度。

#### 2.2.2 生成過程的“大海撈針” (Lost in the Middle)

即使將檢索到的多個大塊文字都塞進LLM的長上下文視窗中，也會出現關鍵資訊被“淹沒”在大量無關內容裡的問題。有研究表明 [^1]，當LLM處理非常長的、充滿大量資訊的上下文時，它傾向於更好地記住開頭和結尾的資訊，而忽略中間部分的內容。

如果提供給LLM的上下文塊又大又雜，充滿了與問題無關的噪音，模型就很難從中提取出最關鍵的資訊來形成答案，從而導致回答質量下降或產生幻覺。

#### 2.2.3 主題稀釋導致檢索失敗

一個好的文字塊應該聚焦於一個明確、單一的主題。如果一個塊包含太多不相關的主題，它的語義就會被稀釋，導致在檢索時無法被精確匹配。

**舉個栗子🌰：**

假設有一個關於《王者榮耀》英雄魯班七號的攻略文件。

- **糟糕的分塊策略**：將“技能介紹”、“推薦出裝”和“背景故事”這三個完全不同主題的內容，全部放在一個巨大的文字塊裡。
    - 當玩家查詢“魯班七號怎麼出裝？”時，這個大塊雖然包含了出裝資訊，但由於被技能說明和英雄故事等無關主題嚴重稀釋，其整體的檢索相關性得分可能會很低，導致無法被召回。

- **優秀的分塊策略**：將“技能”、“出裝”和“故事”分別切分為三個獨立的、主題聚焦的塊。
    - 當玩家再次查詢時，“推薦出裝”這個塊會因為與查詢高度相關而獲得極高的分數，從而被精準地檢索出來。

透過合理分塊，可以有效提升檢索的訊雜比，確保了後續生成環節能得到最優質、最相關的上下文。

## 三、基礎分塊策略

LangChain 提供了豐富且易於使用的文字分割器（Text Splitters），下面將介紹幾種最核心的策略。

### 3.1 固定大小分塊

這是最簡單直接的分塊方法。根據LangChain原始碼，這種方法的工作原理分為兩個主要階段：

（1）**按段落分割**：`CharacterTextSplitter` 採用預設分隔符 `"\n\n"`，使用正規表示式將文字按段落進行分割，透過 `_split_text_with_regex` 函式處理。

（2）**智慧合併**：呼叫繼承自父類的 `_merge_splits` 方法，將分割後的段落依次合併。該方法會監控累積長度，當超過 `chunk_size` 時形成新塊，並透過重疊機制（`chunk_overlap`）保持上下文連續性，同時在必要時發出超長塊的警告。

需要注意，`CharacterTextSplitter` 實際實現的並非嚴格的固定大小分塊。根據 `_merge_splits` 原始碼邏輯，這種方法會：

- **優先保持段落完整性**：只有當新增新段落會導致總長度超過 `chunk_size` 時，才會結束當前塊
- **處理超長段落**：如果單個段落超過 `chunk_size`，系統會發出警告但仍將其作為完整塊保留
- **應用重疊機制**：透過 `chunk_overlap` 引數在塊之間保持內容重疊，確保上下文連續性

所以，LangChain 的實現更準確地應該稱為"段落感知的自適應分塊"，塊大小會根據段落邊界動態調整。

下面的程式碼展示瞭如何配置一個固定大小分塊器：

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../data/C2/txt/蜂醫.txt")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,    # 每個塊的目標大小為100個字元
    chunk_overlap=10   # 每個塊之間重疊10個字元，以緩解語義割裂
)

chunks = text_splitter.split_documents(docs)

print(f"文字被切分為 {len(chunks)} 個塊。\n")
print("--- 前5個塊內容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    # chunk 是一個 Document 物件，需要訪問它的 .page_content 屬性來獲取文字
    print(f'塊 {i+1} (長度: {len(chunk.page_content)}): "{chunk.page_content}"')
```

這種方法的主要優勢在於實現簡單、處理速度快且計算開銷小。劣勢在於可能會在語義邊界處切斷文字，影響內容的完整性和連貫性。實際的固定大小分塊實現（如LangChain的 `CharacterTextSplitter`）通常會結合分隔符來減少這種問題，在段落邊界處優先切分，只有在必要時才會強制按大小切斷。因此，這種方法在日誌分析、資料預處理等場景中仍有其應用價值。

### 3.2 遞迴字元分塊

在前面的章節中，已經嘗試了使用 `RecursiveCharacterTextSplitter` 的預設配置來處理文件分塊。現在讓我們深入瞭解 `RecursiveCharacterTextSplitter` 的實現。這種分塊器透過分隔符層級遞迴處理，相對與固定大小分塊，改善了超長文字的處理效果。

**演算法流程**：
（1）**尋找有效分隔符**: 從分隔符列表中從前到後遍歷，找到第一個在當前文字中**存在**的分隔符。如果都不存在，使用最後一個分隔符（通常是空字串 `""`）。

（2）**切分與分類處理**: 使用選定的分隔符切分文字，然後遍歷所有片段：
- **如果片段不超過塊大小**: 暫存到 `_good_splits` 中，準備合併
- **如果片段超過塊大小**:
    - 首先，將暫存的合格片段透過 `_merge_splits` 合併成塊
    - 然後，檢查是否還有剩餘分隔符：
        - **有剩餘分隔符**: 遞迴呼叫 `_split_text` 繼續分割
        - **無剩餘分隔符**: 直接保留為超長塊

（3）**最終處理**: 將剩餘的暫存片段合併成最後的塊

**實現細節**：
- **批處理機制**: 先收集所有合格片段（`_good_splits`），遇到超長片段時才觸發合併操作。
- **遞迴終止條件**: 關鍵在於 `if not new_separators` 判斷。當分隔符用盡時（`new_separators` 為空），停止遞迴，直接保留超長片段。確保演算法不會無限遞迴。

**與固定大小分塊的關鍵差異**：
- 固定大小分塊遇到超長段落時只能發出警告並保留。
- 遞迴分塊會繼續使用更細粒度的分隔符（句子→單詞→字元）直到滿足大小要求。

具體示例如下：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../data/C2/txt/蜂醫.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 分隔符優先順序
    chunk_size=200,
    chunk_overlap=10,
)

chunks = text_splitter.split_text(docs)
```

**分隔符配置**：
- **預設分隔符**：`["\n\n", "\n", " ", ""]`
- **多語言支援**：對於無詞邊界語言（中文、日文、泰文），可新增：
  ```python
  separators=[
      "\n\n", "\n", " ",
      ".", ",", "\u200b",      # 零寬空格(泰文、日文)
      "\uff0c", "\u3001",      # 全形逗號、表意逗號
      "\uff0e", "\u3002",      # 全形句號、表意句號
      ""
  ]
  ```

**程式語言特化支援**：

`RecursiveCharacterTextSplitter` 能夠針對特定的程式語言（如Python, Java等）使用預設的、更符合程式碼結構的分隔符。它們通常包含語言的頂級語法結構（如類、函式定義）和次級結構（如控制流語句），以實現更符合程式碼邏輯的分割。

```python
# 針對程式碼文件的最佳化分隔符
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,  # 支援Python、Java、C++等
    chunk_size=500,
    chunk_overlap=50
)
```

遞迴字元分塊的原理是採用一組有層次結構的分隔符（如段落、句子、單詞）進行遞迴分割，旨在有效平衡語義完整性與塊大小控制。在 `RecursiveCharacterTextSplitter` 的實現中，該分塊器首先嚐試使用最高優先順序的分隔符（如段落標記）來切分文字。如果切分後的塊仍然過大，會繼續對這個大塊應用下一優先順序分隔符（如句號），如此迴圈往復，直到塊滿足大小限制。這種分層處理的機制，能夠在儘可能保持高階語義結構完整性的同時，有效控制塊大小。

### 3.3 語義分塊

語義分塊（Semantic Chunking）是一種更智慧的方法，這種方法不依賴於固定的字元數或預設的分隔符，而是嘗試根據文字的語義內涵來切分。其核心是：**在語義主題發生顯著變化的地方進行切分**。這使得每個分塊都具有高度的內部語義一致性。LangChain 提供了 `langchain_experimental.text_splitter.SemanticChunker` 來實現這一功能。

**實現原理**

`SemanticChunker` 的工作流程可以概括為以下幾個步驟：

（1）**句子分割 (Sentence Splitting)**：首先，使用標準的句子分割規則（例如，基於句號、問號、感嘆號）將輸入文字拆分成一個句子列表。

（2）**上下文感知嵌入 (Context-Aware Embedding)**：這是 `SemanticChunker` 的一個關鍵設計。該分塊器不是對每個句子獨立進行嵌入，而是透過 `buffer_size` 引數（預設為1）來捕捉上下文資訊。對於列表中的每一個句子，這種方法會將其與前後各 `buffer_size` 個句子組合起來，然後對這個臨時的、更長的組合文字進行嵌入。這樣，每個句子最終得到的嵌入向量就融入了其上下文的語義。

（3）**計算語義距離 (Distance Calculation)**：計算每對**相鄰**句子的嵌入向量之間的餘弦距離。這個距離值量化了兩個句子之間的語義差異——距離越大，表示語義關聯越弱，跳躍越明顯。

（4）**識別斷點 (Breakpoint Identification)**：`SemanticChunker` 會分析所有計算出的距離值，並根據一個統計方法（預設為 `percentile`）來確定一個動態閾值。例如，它可能會將所有距離中第95百分位的值作為切分閾值。所有距離大於此閾值的點，都被識別為語義上的“斷點”。

（5）**合併成塊 (Merging into Chunks)**：最後，根據識別出的所有斷點位置，將原始的句子序列進行切分，並將每個切分後的部分內的所有句子合併起來，形成一個最終的、語義連貫的文字塊。

**斷點識別方法 (`breakpoint_threshold_type`)**

如何定義“顯著的語義跳躍”是語義分塊的關鍵。`SemanticChunker` 提供了幾種基於統計的方法來識別斷點：

-   `percentile` (百分位法 - **預設方法**):
    -   **邏輯**: 計算所有相鄰句子的語義差異值，並將這些差異值進行排序。當一個差異值超過某個百分位閾值時，就認為該差異值是一個斷點。
    -   **引數**: `breakpoint_threshold_amount` (預設為 `95`)，表示使用第95個百分位作為閾值。這意味著，只有最顯著的5%的語義差異點會被選為切分點。

-   `standard_deviation` (標準差法):
    -   **邏輯**: 計算所有差異值的平均值和標準差。當一個差異值超過“平均值 + N * 標準差”時，被視為異常高的跳躍，即斷點。
    -   **引數**: `breakpoint_threshold_amount` (預設為 `3`)，表示使用3倍標準差作為閾值。

-   `interquartile` (四分位距法):
    -   **邏輯**: 使用統計學中的四分位距（IQR）來識別異常值。當一個差異值超過 `Q3 + N * IQR` 時，被視為斷點。
    -   **引數**: `breakpoint_threshold_amount` (預設為 `1.5`)，表示使用1.5倍的IQR。

-   `gradient` (梯度法):
    -   **邏輯**: 這是一種更復雜的方法。它首先計算差異值的變化率（梯度），然後對梯度應用百分位法。對於那些句子間語義聯絡緊密、差異值普遍較低的文字（如法律、醫療文件）特別有效，因為這種方法能更好地捕捉到語義變化的“拐點”。
    -   **引數**: `breakpoint_threshold_amount` (預設為 `95`)。

**具體示例如下**

```python
import os
## os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 初始化 SemanticChunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile" # 斷點識別方法
)

loader = TextLoader("../../data/C2/txt/蜂醫.txt")
documents = loader.load()

docs = text_splitter.split_documents(documents)
```

### 3.4 基於文件結構的分塊

對於具有明確結構標記的文件格式（如Markdown、HTML、LaTex），可以利用這些標記來實現更智慧、更符合邏輯的分割。

#### 以 Markdown 結構分塊為例

針對結構清晰的 Markdown 文件，利用其標題層級進行分塊是一種高效且保留了豐富語義的方法。LangChain 提供了 `MarkdownHeaderTextSplitter` 來處理。

- **實現原理**: 該分塊器的主要邏輯是“先按標題分組，再按需細分”。
    1.  **定義分割規則**: 使用者首先需要提供一個標題層級的對映關係，例如 `[ ("#", "Header 1"), ("##", "Header 2") ]`，告訴分塊器 `#` 是一級標題，`##` 是二級標題。
    2.  **內容聚合**: 分塊器會遍歷整個文件，將每個標題下的所有內容（直到下一個同級或更高階別的標題出現前）聚合在一起。每個聚合後的內容塊都會被賦予一個包含其完整標題路徑的後設資料。

- **後設資料注入的優勢**: 這是此方法的主要特點。例如，對於一篇關於機器學習的文章，某個段落可能位於“第三章：模型評估”下的“3.2節：評估指標”中。經過分割後，這個段落形成的文字塊，其後設資料就會是 `{"Header 1": "第三章：模型評估", "Header 2": "3.2節：評估指標"}`。這種後設資料為每個塊提供了精確的“地址”，極大地增強了上下文的準確性，讓大模型能更好地理解資訊片段的來源和背景。

- **侷限性與組合使用**: 單純按標題分割可能會導致一個問題：某個章節下的內容可能非常長，遠超模型能處理的上下文視窗。為了解決這個問題，`MarkdownHeaderTextSplitter` 可以與其它分塊器（如 `RecursiveCharacterTextSplitter`）**組合使用**。具體流程是：
    - 第一步，使用 `MarkdownHeaderTextSplitter` 將文件按標題分割成若干個大的、帶有後設資料的邏輯塊。
    - 第二步，對這些邏輯塊再應用 `RecursiveCharacterTextSplitter`，將其進一步切分為符合 `chunk_size` 要求的小塊。由於這個過程是在第一步之後進行的，所有最終生成的小塊都會**繼承**來自第一步的標題後設資料。

- **RAG應用優勢**: 這種兩階段的分塊方法，既保留了文件的宏觀邏輯結構（透過後設資料），又確保了每個塊的大小適中，是處理結構化文件進行RAG的理想方案。

## 四、其他開源框架中的分塊策略

### 4.1 Unstructured：基於文件元素的智慧分塊

`Unstructured`是一個強大的文件處理工具，同樣提供了實用的[分塊功能](https://docs.unstructured.io/open-source/core-functionality/chunking)。

（1）**分割槽 (Partitioning)**: 這是一個重要功能，負責將原始文件（如PDF、HTML）解析成一系列結構化的“元素”（Elements）。每個元素都帶有語義標籤，如 `Title` (標題)、`NarrativeText` (敘述文字)、`ListItem` (列表項) 等。這個過程本身就完成了對文件的深度理解和結構化。

（2）**分塊 (Chunking)**: 該功能建立在**分割槽**的結果之上。分塊功能不是對純文字進行操作，而是將分割槽產生的“元素”列表作為輸入，進行智慧組合。Unstructured 提供了兩種主要的分塊方法：
-   **`basic`**: 這是預設方法。這種方法會連續地組合文件元素（如段落、列表項），直到達到 `max_characters` 上限，儘可能地填滿每個塊。如果單個元素超過上限，則會對其進行文字分割。
-   **`by_title`**: 該方法在 `basic` 方法的基礎上，增加了對“章節”的感知。該方法將 `Title` 元素視為一個新章節的開始，並強制在此處開始一個新的塊，確保同一個塊內不會包含來自不同章節的內容。這在處理報告、書籍等結構化文件時非常有用，效果類似於 LangChain 的 `MarkdownHeaderTextSplitter`，但適用範圍更廣。

Unstructured 允許將分塊作為分割槽的一個引數在單次呼叫中完成，也支援在分割槽之後作為一個獨立的步驟來執行分塊。這種“先理解、後分割”的策略，使得 Unstructured 能在最大程度上保留文件的原始語義結構，特別是在處理版式複雜的文件時，優勢尤為明顯。

### 4.2 LlamaIndex：面向節點的解析與轉換

[LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/) 將資料處理流程抽象為對“**節點（Node）**”的操作。文件被載入後，首先會被解析成一系列的“節點”，分塊只是節點轉換（Transformation）中的一環。

LlamaIndex 的分塊體系有以下特點：

（1）**豐富的節點解析器 (Node Parser)**: LlamaIndex 提供了大量針對特定資料格式和方法的節點解析器，可以大致分為幾類：
-   **結構感知型**: 如 `MarkdownNodeParser`, `JSONNodeParser`, `CodeSplitter` 等，能理解並根據原始檔的結構（如Markdown標題、程式碼函式）進行切分。
-   **語義感知型**: 
    -   `SemanticSplitterNodeParser`: 與 LangChain 的 `SemanticChunker` 類似，這種解析器使用嵌入模型來檢測句子之間的語義“斷點”，在語義連續性明顯減弱的地方切開，從而讓每個 chunk 內部儘量連貫。
    -   `SentenceWindowNodeParser`: 這是一種巧妙的方法。該方法將文件切分成單個的句子，但在每個句子節點（Node）的後設資料中，會儲存其前後相鄰的N個句子（即“視窗”）。這使得在檢索時，可以先用單個句子的嵌入進行精確匹配，然後將包含上下文“視窗”的完整文字送給LLM，極大地提升了上下文的質量。
-   **常規型**: 如 `TokenTextSplitter`, `SentenceSplitter` 等，提供基於Token數量或句子邊界的常規切分方法。

（2）**靈活的轉換流水線**: 使用者可以構建一個靈活的流水線，例如先用 `MarkdownNodeParser` 按章節切分文件，再對每個章節節點應用 `SentenceSplitter` 進行更細粒度的句子級切分。每個節點都攜帶豐富的後設資料，記錄著其來源和上下文關係。

（3）**良好的互操作性**: LlamaIndex 提供了 `LangchainNodeParser`，可以方便地將任何 LangChain 的 `TextSplitter` 封裝成 LlamaIndex 的節點解析器，無縫整合到其處理流程中。

### 4.3 ChunkViz：簡易的視覺化分塊工具

在本文開頭部分展示的分塊圖就是透過 [**ChunkViz**](https://chunkviz.up.railway.app/) 生成的。可以將你的文件、分塊配置作為輸入，用不同的顏色塊展示每個 chunk 的邊界和重疊部分，方便快速理解分塊邏輯。

## 參考文獻

[^1]: [Nelson F. Liu, et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172).
