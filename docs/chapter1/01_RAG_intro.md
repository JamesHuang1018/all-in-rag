# 第一節 RAG 簡介

## 一、什麼是 RAG？

### 1.1 核心定義

從本質上講，RAG（Retrieval-Augmented Generation）是一種旨在解決大語言模型（LLM）“知其然不知其所以然”問題的技術正規化。它的核心是將模型內部學到的“**引數化知識**”（模型權重中固化的、模糊的“記憶”），與來自外部知識庫的“**非引數化知識**”（精準、可隨時更新的外部資料）相結合。其運作邏輯就是在 LLM 生成文字前，先透過檢索機制從外部知識庫中動態獲取相關資訊，並將這些“參考資料”融入生成過程，從而提升輸出的準確性和時效性 [^1] [^2] [^3]。

> 💡 **一句話總結**：RAG 就是讓 LLM 學會了“開卷考試”，它既能利用自己學到的知識，也能隨時查閱外部資料。

### 1.2 技術原理

那麼，RAG 系統是如何實現“引數化知識”與“非引數化知識”的結合呢？如圖 1-1 所示，其架構主要透過兩個階段來完成這一過程：

（1）**檢索階段：尋找“非引數化知識”**
-   **知識向量化**：**嵌入模型（Embedding Model）** 充當了“聯結器”的角色。它將外部知識庫編碼為向量索引（Index），存入**向量資料庫**。
-   **語義召回**：當使用者發起查詢時，檢索模組利用同樣的嵌入模型將問題向量化，並透過**相似度搜尋（Similarity Search）**，從海量資料中精準鎖定與問題最相關的文件片段。

（2）**生成階段：融合兩種知識**
-   **上下文整合**：**生成模組**接收檢索階段送來的相關文件片段以及使用者的原始問題。
-   **指令引導生成**：該模組會遵循預設的 **Prompt** 指令，將上下文與問題有效整合，並引導 LLM（如 DeepSeek）進行可控的、有理有據的文字生成。

<div align="center">
   <img src="./images/1_1_1.svg" width="60%" alt="RAG 雙階段架構示意圖">
   <p>圖 1-1 RAG 雙階段架構示意圖</p>
</div>

### 1.3 技術演進分類

RAG 的技術架構經歷了從簡單到複雜的演進，如圖 1-2 大致可分為三個階段 [^4]。

<div align="center">
   <img src="./images/1_1_2.png" width="80%" alt="RAG 技術演進分類">
   <p>圖 1-2 RAG 技術演進分類</p>
</div>

這三個階段的具體對比如表 1-1 所示。

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
    <th style="text-align: center;"></th>
    <th style="text-align: center;">初級 RAG（Naive RAG）</th>
    <th style="text-align: center;">高階 RAG（Advanced RAG）</th>
    <th style="text-align: center;">模組化 RAG（Modular RAG）</th>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>流程</strong></td>
    <td style="text-align: center;"><strong>離線:</strong> <code>索引</code><br><strong>線上:</strong> <code>檢索 → 生成</code></td>
    <td style="text-align: center;"><strong>離線:</strong> <code>索引</code><br><strong>線上:</strong> <code>...→ 檢索前 → ... → 檢索後 → ...</code></td>
    <td style="text-align: center;">積木式可編排流程</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>特點</strong></td>
    <td style="text-align: center;">基礎線性流程</td>
    <td style="text-align: center;">增加<strong>檢索前後</strong>的最佳化步驟</td>
    <td style="text-align: center;">模組化、可組合、可動態調整</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>關鍵技術</strong></td>
    <td style="text-align: center;">基礎向量檢索</td>
    <td style="text-align: center;"><strong>查詢重寫（Query Rewrite）</strong><br><strong>結果重排（Rerank）</strong></td>
    <td style="text-align: center;"><strong>動態路由（Routing）</strong><br><strong>查詢轉換（Query Transformation）</strong><br><strong>多路融合（Fusion）</strong></td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>侷限性</strong></td>
    <td style="text-align: center;">效果不穩定，難以最佳化</td>
    <td style="text-align: center;">流程相對固定，最佳化點有限</td>
    <td style="text-align: center;">系統複雜性高</td>
  </tr>
</table>
<p><em>表 1-1 RAG 技術演進分類對比</em></p>
</div>

> “離線”指提前完成的資料預處理工作（如索引構建）；“線上”指使用者發起請求後的實時處理流程。

## 二、為什麼要使用 RAG？

### 2.1 技術選型：RAG vs. 微調

在選擇具體的技術路徑時，一個重要的考量是成本與效益的平衡。通常，我們應優先選擇對模型改動最小、成本最低的方案，所以技術選型路徑往往遵循的順序是**提示詞工程（Prompt Engineering） -> 檢索增強生成 -> 微調（Fine-tuning）**。

我們可以從兩個維度來理解這些技術的區別。如圖 1-3 所示，**橫軸代表“LLM 最佳化”**，即對模型本身進行多大程度的修改。從左到右，最佳化的程度越來越深，其中提示工程和 RAG 完全不改變模型權重，而微調則直接修改模型引數。**縱軸代表“上下文最佳化”**，是對輸入給模型的資訊進行多大程度的增強。從下到上，增強的程度越來越高，其中提示工程只是最佳化提問方式，而 RAG 則透過引入外部知識庫，極大地豐富了上下文資訊。

<div align="center">
  <img src="./images/1_1_3.svg" width="60%" alt="技術選型路徑" />
  <p>圖 1-3 選型路徑圖</p>
</div>

基於此，我們的選擇路徑就清晰了：
- **先嚐試提示工程**：透過精心設計提示詞來引導模型，適用於任務簡單、模型已有相關知識的場景。
- **再選擇 RAG**：如果模型缺乏特定或實時知識而無法回答，則使用 RAG，透過外掛知識庫為其提供上下文資訊。
- **最後考慮微調**：當目標是改變模型“如何做”（行為/風格/格式）而不是“知道什麼”（知識）時，微調是最終且最合適的選擇。例如，讓模型學會嚴格遵循某種獨特的輸出格式、模仿特定人物的對話風格，或者將極其複雜的指令“蒸餾”進模型權重中。

RAG 的出現填補了通用模型與專業領域之間的鴻溝，它在解決如表 1-2 所示 LLM 侷限時尤其有效：

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
    <th style="text-align: center;">問題</th>
    <th style="text-align: center;">RAG的解決方案</th>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>靜態知識侷限</strong></td>
    <td style="text-align: center;">實時檢索外部知識庫，支援動態更新</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>幻覺（Hallucination）</strong></td>
    <td style="text-align: center;">基於檢索內容生成，錯誤率降低</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>領域專業性不足</strong></td>
    <td style="text-align: center;">引入領域特定知識庫（如醫療/法律）</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>資料隱私風險</strong></td>
    <td style="text-align: center;">本地化部署知識庫，避免敏感資料洩露</td>
  </tr>
</table>
<p><em>表 1-2 RAG 對 LLM 侷限的解決方案</em></p>
</div>

### 2.2 關鍵優勢 

（1）**準確性與可信度的雙重提升**

RAG 最核心的價值在於突破了模型預訓練知識的限制。它不僅能**補充專業領域的知識盲區**，還能透過提供具體的參考材料，有效**抑制“一本正經胡說八道”的幻覺現象**。論文研究還表明，RAG 生成的內容在**具體性**和**多樣性**上也顯著優於純 LLM。更重要的是，RAG 具備**可溯源性**——每一條回答都能找到對應的原始文件出處，這種“有據可查”的特性極大提高了內容在法律、醫療等嚴肅場景下的可信度。

（2）**時效性保障**

在知識更新方面，RAG 解決了 LLM 固有的**知識時滯問題**（即模型不知道訓練截止日期之後發生的事）。RAG 允許知識庫獨立於模型進行**動態更新**——新政策或新資料一旦入庫，立刻就能被檢索到。這種能力在論文中被稱為**“索引熱拔插”（Index Hot-swapping）**——就像給機器人換一張儲存卡一樣，瞬間切換其世界知識庫，而無需重新訓練模型，實現了知識的實時線上。

（3）**顯著的綜合成本效益**

從經濟角度看，RAG 是一種高價效比的方案。首先，它**避免了高頻微調**帶來的鉅額算力成本；其次，由於有了外部知識的強力輔助，我們在處理特定領域問題時，往往可以使用**引數量更小的基礎模型**來達到類似的效果，從而直接降低了推理成本。這種架構也減少了試圖將海量知識強行“塞入”模型權重中所需的計算資源消耗。

（4）**靈活的模組化可擴充套件性**

RAG 的架構具備極強的包容性，支援**多源整合**，無論是 PDF、Word 還是網頁資料，都能統一構建進知識庫中。同時，其**模組化設計**實現了檢索與生成的解耦，這意味著我們可以獨立最佳化檢索元件（比如更換更好的 Embedding 模型），而不會影響到生成元件的穩定性，便於系統的長期迭代。

### 2.3 適用場景風險分級 

表 1-3 展示了 RAG 技術在不同風險等級場景中的適用性。

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
    <th style="text-align: center;">風險等級</th>
    <th style="text-align: center;">案例</th>
    <th style="text-align: center;">RAG適用性</th>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>低風險</strong></td>
    <td style="text-align: center;">翻譯/語法檢查</td>
    <td style="text-align: center;">高可靠性</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>中風險</strong></td>
    <td style="text-align: center;">合同起草/法律諮詢</td>
    <td style="text-align: center;">需結合人工稽核</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>高風險</strong></td>
    <td style="text-align: center;">證據分析/簽證決策</td>
    <td style="text-align: center;">需嚴格質量控制機制</td>
  </tr>
</table>
<p><em>表 1-3 RAG 適用場景風險分級</em></p>
</div>

## 三、如何上手 RAG？

### 3.1 基礎工具鏈選擇

構建 RAG 系統通常涉及幾個關鍵環節的選型。在**開發模式**上，我們可以利用 **LangChain** 或 **LlamaIndex** 等成熟框架快速整合，**也可以選擇不依賴框架的原生開發**，以獲得對系統流程更精細的控制力（在 AI 程式設計輔助下這並非難事）。而在**記憶載體**（向量資料庫）方面，既有 **Milvus**、**Pinecone** 等適合大規模資料的方案，也有 **FAISS**、**Chroma** 等輕量級或本地化的選擇，需根據具體業務規模靈活決定。後期為了量化效果，還可以引入 **RAGAS** 或 **TruLens** 等自動化**評估工具**。

### 3.2 四步構建最小可行系統（MVP）

（1）**資料準備與清洗**：這是系統的地基。我們需要將 PDF、Word 等多源異構資料標準化，並採用合理的**分塊策略**（如按語義段落切分而非固定字元數），避免資訊在切割中支離破碎。

（2）**索引構建**：將切分好的文字透過**嵌入模型**轉化為向量，並存入資料庫。可以在此階段關聯**後設資料**（如來源、頁碼），這對後續的精確引用很有幫助。

（3）**檢索策略最佳化**：不要依賴單一的向量搜尋。可以採用**混合檢索**（向量+關鍵詞）等方式來提升召回率，並引入**重排序**模型對檢索結果進行二次精選，確保 LLM 看到的都是精華。

（4）**生成與提示工程**：最後，設計一套清晰的 **Prompt 模板**，引導 LLM 基於檢索到的上下文回答使用者問題，並明確要求模型“不知道就說不知道”，防止幻覺。

### 3.3 新手友好方案

如果希望快速驗證想法而非深耕程式碼，可以嘗試 **FastGPT** 或 **Dify** 這樣的視覺化知識庫平臺，它們封裝了複雜的 RAG 流程，僅需上傳文件即可使用。對於開發者，利用 **LangChain4j Easy RAG** 或 GitHub 上的 **TinyRAG** [^6]等開源模板，也是高效的起手方式。

### 3.4 進階與挑戰

當基礎的 RAG 系統搭建完成後，下一步的進階之路便聚焦於如何評估、診斷並突破其固有的瓶頸。

（1）**評估維度與挑戰**

一套 RAG 系統的好壞，並不能僅憑感覺。業界通常會從幾個維度進行量化評估，首先是**檢索相關性**（找到的內容是否包含答案），其次是**生成質量**，這又可以細分為**語義準確性**（回答的意思是否正確）和**詞彙匹配度**（專業術語是否使用得當）。

這些評估維度也直接對應了 RAG 當前面臨的主要挑戰。比如，**檢索依賴性**問題——如果檢索系統召回了錯誤資訊，再強的 LLM 也會“一本正經地胡說八道”。此外，對於需要跨多個文件進行綜合分析的**多跳推理**問題，常見的 RAG 架構也普遍感到吃力。

（2）**最佳化方向與架構演進**

針對上述挑戰，社群探索出了多種最佳化路徑。在**效能層面**，可以透過**索引分層**（對高頻資料啟用快取）和**多模態擴充套件**（支援影象/表格檢索）來提升效率和能力邊界。而在**架構層面**，簡單的線性流程正在被更復雜的**設計模式**所取代。例如，系統可以透過**分支模式**並行處理多路檢索，或透過**迴圈模式**進行自我修正，這些靈活的架構是通往更智慧 RAG 的必由之路。

## 四、RAG 已死？

隨著大模型長上下文視窗能力的提升，社群中開始出現“RAG 已死”的聲音。這一論調主要來自兩個方面，一是認為長上下文已經能暴力“消化”海量文字，不再需要複雜的檢索系統；二是批評 RAG 這個術語本身就過於寬泛，模糊了太多技術細節，反而阻礙了理解與最佳化。

這些觀點忽略了一個技術概念在演進過程中的普遍規律。正如我們可以輕易地為現代複雜的 RAG 系統起一個更精確、更唬人的名字，比如 **“大模型知識管理專家系統”（Large Language Model Knowledge Management Expert System，LKE）**。因為它早已超出了最初“檢索-增強-生成”的簡單範疇。但這種“換名遊戲”，恰恰說明了“RAG 已死”論的表面化——這無異於在用一個新瓶子去裝 RAG 這個不斷陳化的老酒。

> 筆者在此並非要創造一個新詞，不過為什麼要起 LKE 這個名字？它代表了三個核心要素：
> -   **L（Large Language Model）**：強調系統的驅動力是大語言模型。
> -   **K（Knowledge Management）**：寓意著系統就像一個知識管理員，精準地為我們找到（**檢索**）所需要的知識，輔助我們後續利用大模型進行更高階應用。
> -   **E（Expert）**：說明系統能像專家一樣，透過路由、分析、融合、修正等一系列步驟，最終給出答案（**生成**）、解決問題。

可以類比 **Transformer**。今天無論是以 GPT 為代表的 Decoder-only 還是以 BERT 為代表的 Encoder-only，我們都習慣稱之為“基於 Transformer 架構”，儘管它們與最初論文中的完整形態差異巨大。但是 Transformer 這個標籤抓住了一次技術正規化的核心飛躍，併成為了一個技術時代的象徵。同理，**RAG 的核心在於“將 LLM 的內在引數化知識與外部非引數化知識相結合”**。只要這個思想或需求不變，無論我們為其增加多少模組——查詢轉換、多路召回或者自我修正等等，它本質上依然是在這個框架下的演進。

所以，“RAG 已死”是一個偽命題。相反，**RAG 作為一個概念活得很好**，它正在像 Transformer 一樣，成為一個不斷吸收新技術、不斷進化的基礎架構正規化。它的生命力，正在於它的“面目全非”和“包羅永珍”。而**本教程的目標，就是繪製出這張描繪 RAG 全貌的清晰地圖，當我們可以解構它的每一個模組、理解它的每一種可能性時，RAG 也好，LKE 也罷，這些都無關緊要**。我們要做的就是透過 RAG 這道經典例題來學習和拓展（將 LLM 的內在引數化知識與外部非引數化知識相結合）這類題型的解題思路。

> RAG 技術仍在快速發展中，可以持續關注學術和工業界的最新進展！

## 參考文獻

[^1]: [Genesis, J. (2025). *Retrieval-Augmented Text Generation: Methods, Challenges, and Applications*](https://www.researchgate.net/publication/391141346_Retrieval-Augmented_Generation_Methods_Applications_and_Challenges).

[^2]: [Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*](https://arxiv.org/abs/2312.10997).

[^3]: [Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401). 

[^4]: [Gao et al. (2024). *Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks*](https://arxiv.org/abs/2407.21059).

[^6]: [*TinyRAG: GitHub專案*](https://github.com/KMnO4-zx/TinyRAG). 