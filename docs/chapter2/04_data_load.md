# 第一節 資料載入

雖然本節內容在實際應用中非常重要，但是由於各種文件載入器的迭代更新，以及各類 AI 應用的不同需求，具體選擇需要根據實際情況。本節僅作簡單引入，但請務必**重視資料載入**環節，**“垃圾進，垃圾出 (Garbage In, Garbage Out)”** ——高質量輸入是高質量輸出的前提。

## 一、文件載入器

### 1.1 主要功能

RAG 系統中，**資料載入**是整個流水線的第一步，也是不可或缺的一步。文件載入器負責將各種格式的非結構化文件（如PDF、Word、Markdown、HTML等）轉換為程式可以處理的結構化資料。資料載入的質量會直接影響後續的索引構建、檢索效果和最終的生成質量。

文件載入器在 RAG 的資料管道中一般需要完成三個核心任務，一是解析不同格式的原始文件，將 PDF、Word、Markdown 等內容提取為可處理的純文字，二是在解析過程中同時抽取文件來源、頁碼、作者等關鍵資訊作為後設資料，三是把文字和後設資料整理成統一的資料結構，方便後續進行切分、向量化和入庫，其整體流程與傳統資料工程中的抽取、轉換、載入相似，目標都是把雜亂的原始文件清洗並對齊為適合檢索和建模的標準化語料。

### 1.2 當前主流RAG文件載入器

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
    <th style="text-align: center;">工具名稱</th>
    <th style="text-align: center;">特點</th>
    <th style="text-align: center;">適用場景</th>
    <th style="text-align: center;">效能表現</th>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>PyMuPDF4LLM</strong></td>
    <td style="text-align: center;">PDF→Markdown轉換，OCR+表格識別</td>
    <td style="text-align: center;">科研文獻、技術手冊</td>
    <td style="text-align: center;">開源免費，GPU加速</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>TextLoader</strong></td>
    <td style="text-align: center;">基礎文字檔案載入</td>
    <td style="text-align: center;">純文字處理</td>
    <td style="text-align: center;">輕量高效</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>DirectoryLoader</strong></td>
    <td style="text-align: center;">批次目錄檔案處理</td>
    <td style="text-align: center;">混合格式文件庫</td>
    <td style="text-align: center;">支援多格式擴充套件</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Unstructured</strong></td>
    <td style="text-align: center;">多格式文件解析</td>
    <td style="text-align: center;">PDF、Word、HTML等</td>
    <td style="text-align: center;">統一介面，智慧解析</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>FireCrawlLoader</strong></td>
    <td style="text-align: center;">網頁內容抓取</td>
    <td style="text-align: center;">線上文件、新聞</td>
    <td style="text-align: center;">實時內容獲取</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>LlamaParse</strong></td>
    <td style="text-align: center;">深度PDF結構解析</td>
    <td style="text-align: center;">法律合同、學術論文</td>
    <td style="text-align: center;">解析精度高，商業API</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Docling</strong></td>
    <td style="text-align: center;">模組化企業級解析</td>
    <td style="text-align: center;">企業合同、報告</td>
    <td style="text-align: center;">IBM生態相容</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Marker</strong></td>
    <td style="text-align: center;">PDF→Markdown，GPU加速</td>
    <td style="text-align: center;">科研文獻、書籍</td>
    <td style="text-align: center;">專注PDF轉換</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>MinerU</strong></td>
    <td style="text-align: center;">多模態整合解析</td>
    <td style="text-align: center;">學術文獻、財務報表</td>
    <td style="text-align: center;">整合LayoutLMv3+YOLOv8</td>
  </tr>
</table>
<p><em>表 2-1 當前主流 RAG 文件載入器</em></p>
</div>

## 二、Unstructured文件處理庫

### 2.1 Unstructured 的核心優勢

**Unstructured** [^1]是一個專業的文件處理庫，專門設計用於RAG和AI微調場景的非結構化資料預處理。提供了統一的介面來處理多種文件格式，是目前應用較廣泛的文件載入解決方案之一。Unstructured 在格式支援和內容解析方面具有明顯優勢，它一方面支援 PDF、Word、Excel、HTML、Markdown 等多種文件格式，並透過統一的 API 介面避免為不同格式分別編寫程式碼，另一方面可以自動識別標題、段落、表格、列表等文件結構，同時保留相應的後設資料資訊。

<div align="center">
  <img src="./images/2_1_1.png" width="80%" alt="Unstructured 官網介面">
  <p>圖 2-1 Unstructured 官網介面</p>
</div>

### 2.2 支援的文件元素型別

Unstructured 能夠識別和分類以下文件元素 [^2]：

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
    <th style="text-align: center;">元素型別</th>
    <th style="text-align: center;">描述</th>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Title</code></td>
    <td style="text-align: center;">文件標題</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>NarrativeText</code></td>
    <td style="text-align: center;">由多個完整句子組成的正文文字，不包括標題、頁首、頁尾和說明文字</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>ListItem</code></td>
    <td style="text-align: center;">列表項，屬於列表的正文文字元素</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Table</code></td>
    <td style="text-align: center;">表格</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Image</code></td>
    <td style="text-align: center;">影象後設資料</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Formula</code></td>
    <td style="text-align: center;">公式</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Address</code></td>
    <td style="text-align: center;">實體地址</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>EmailAddress</code></td>
    <td style="text-align: center;">郵箱地址</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>FigureCaption</code></td>
    <td style="text-align: center;">圖片標題/說明文字</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Header</code></td>
    <td style="text-align: center;">文件頁首</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Footer</code></td>
    <td style="text-align: center;">文件頁尾</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>CodeSnippet</code></td>
    <td style="text-align: center;">程式碼片段</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>PageBreak</code></td>
    <td style="text-align: center;">頁面分隔符</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>PageNumber</code></td>
    <td style="text-align: center;">頁碼</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>UncategorizedText</code></td>
    <td style="text-align: center;">未分類的自由文字</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>CompositeElement</code></td>
    <td style="text-align: center;">分塊處理時產生的複合元素*</td>
  </tr>
</table>
<p><em>表 2-2 Unstructured 支援的文件元素型別</em></p>
</div>

> `CompositeElement` 是透過分塊處理產生的特殊元素型別，由一個或多個連續的文字元素組合而成。例如，多個列表項可能會被組合成一個單獨的塊。

## 三、從 LangChain 封裝到原始 Unstructured

在第一章的示例中，我們使用了LangChain的`UnstructuredMarkdownLoader`，它是 LangChain 對 Unstructured 庫的封裝。接下來展示如何直接使用 Unstructured 庫，這樣可以獲得更大的靈活性和控制力。

> [本節完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C2/01_unstructured_example.py)

### 3.1 程式碼示例

建立一個簡單的示例，嘗試使用 Unstructured 庫載入並解析一個PDF檔案。

```python
from unstructured.partition.auto import partition

# PDF檔案路徑
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured載入並解析PDF文件
elements = partition(
    filename=pdf_path,
    content_type="application/pdf"
)

# 列印解析結果
print(f"解析完成: {len(elements)} 個元素, {sum(len(str(e)) for e in elements)} 字元")

# 統計元素型別
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素型別: {dict(types)}")

# 顯示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)
```

> 若程式碼執行出現報錯 `ImportError: libgl.so.1 cannot open shared object file no such file or directory`, 執行 `sudo apt-get install python3-opencv` 安裝依賴庫。

**partition 函式引數解析：**

- `filename`: 文件檔案路徑，支援本地檔案路徑；
- `content_type`: 可選引數，指定MIME型別（如"application/pdf"），可繞過自動檔案型別檢測；
- `file`: 可選引數，檔案物件，與 filename 二選一使用；
- `url`: 可選引數，遠端文件 URL，支援直接處理網路文件；
- `include_page_breaks`: 布林值，是否在輸出中包含頁面分隔符；
- `strategy`: 處理策略，可選 "auto"、"fast"、"hi_res" 等；
- `encoding`: 文字編碼格式，預設自動檢測。

`partition`函式使用自動檔案型別檢測，內部會根據檔案型別路由到對應的專用函式（如PDF檔案會呼叫`partition_pdf`）。如果需要更專業的PDF處理，可以直接使用`from unstructured.partition.pdf import partition_pdf`，它提供更多PDF特有的引數選項，如OCR語言設定、影象提取、表格結構推理等高階功能，同時效能更優。

> 在實際應用中，針對 pdf 的處理，目前更多選用的是 PaddleOCR、MinerU 等模型或工具。

## 練習

- 使用`partition_pdf`替換當前`partition`函式並分別嘗試用`hi_res`和`ocr_only`進行解析，觀察輸出結果有何變化。

## 參考文獻

[^1]: [*Unstructured Open-Source Documentation*](https://docs.unstructured.io/open-source/)

[^2]: [*Unstructured Open-Source: Document Elements*](https://docs.unstructured.io/open-source/concepts/document-elements)
