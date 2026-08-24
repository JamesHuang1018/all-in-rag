# 第二節 多模態嵌入

現代 AI 的一項重要突破，是將簡單的詞向量發展成了能統一理解圖文、音影片的複雜系統。這一發展建立在**注意力機制、Transformer 架構和對比學習**等關鍵技術之上，它們解決了在共享向量空間中對齊不同資料模態的核心挑戰。其發展環環相扣：Word2Vec 為 BERT 的上下文理解鋪路，而 BERT 又為 CLIP 等模型的跨模態能力奠定了基礎。

## 一、為什麼需要多模態嵌入？

前面的章節介紹瞭如何為文字建立向量嵌入。然而，僅有文字的世界是不完整的。現實世界的資訊是多模態的，包含影象、音訊、影片等。傳統的文字嵌入無法理解“那張有紅色汽車的圖片”這樣的查詢，因為文字向量和影象向量處於相互隔離的空間，存在一堵“模態牆”。

**多模態嵌入 (Multimodal Embedding)** 的目標正是為了打破這堵牆。其目的是將不同型別的資料（如影象和文字）對映到**同一個共享的向量空間**。在這個統一的空間裡，一段描述“一隻奔跑的狗”的文字，其向量會非常接近一張真實小狗奔跑的圖片向量。

實現這一目標的關鍵，在於解決 **跨模態對齊 (Cross-modal Alignment)** 的挑戰。以對比學習、視覺 Transformer (ViT) 等技術為代表的突破，讓模型能夠學習到不同模態資料之間的語義關聯，最終催生了像 CLIP 這樣的模型。

## 二、CLIP 模型淺析

在圖文多模態領域，OpenAI 的 **CLIP (Contrastive Language-Image Pre-training)** 是一個很有影響力的模型，它為多模態嵌入定義了一個有效的正規化。

CLIP 的架構清晰簡潔。它採用**雙編碼器架構 (Dual-Encoder Architecture)**，包含一個影象編碼器和一個文字編碼器，分別將影象和文字對映到同一個共享的向量空間中。

![CLIP Architecture](./images/3_2_1.webp)
*圖：CLIP 的工作流程。(1) 透過對比學習訓練雙編碼器，對齊圖文向量空間。(2)和(3) 展示瞭如何利用該空間，透過圖文相似度匹配實現零樣本預測。*

為了讓這兩個編碼器學會“對齊”不同模態的語義，CLIP 在訓練時採用了**對比學習 (Contrastive Learning)** 策略。在處理一批圖文資料時，模型的目標是：最大化正確圖文對的向量相似度，同時最小化所有錯誤配對的相似度。透過這種“拉近正例，推遠負例”的方式，模型從海量資料中學會了將語義相關的影象和文字在向量空間中拉近。

這種大規模的對比學習賦予了 CLIP 有效的**零樣本（Zero-shot）識別能力**。它能將一個傳統的分類任務，轉化為一個“圖文檢索”問題——例如，要判斷一張圖片是不是貓，只需計算圖片向量與“a photo of a cat”文字向量的相似度即可。這使得 CLIP 無需針對特定任務進行微調，就能實現對視覺概念的泛化理解。

## 三、常用多模態嵌入模型(以bge-visualized-m3為例)

雖然 CLIP 為圖文預訓練提供了重要基礎，但多模態領域的研究迅速發展，湧現了許多針對不同目標和場景進行最佳化的模型。例如，BLIP 系列專注於提升細粒度的圖文理解與生成能力，而 ALIGN 則證明了利用海量噪聲資料進行大規模訓練的有效性。

在眾多優秀的模型中，由北京智源人工智慧研究院（BAAI）開發的 **bge-visualized-m3（Visualized-BGE 的 M3 版本）** 是一個很有代表性的現代多模態嵌入模型。它是在 **BGE-M3**（文字嵌入底座）的基礎上引入影象能力而來，體現了當前技術向“更統一、更全面”發展的趨勢。

bge-visualized-m3 的核心特性也可以概括為“M3”（主要繼承自其文字底座 BGE-M3）：
- **多語言性 (Multi-Linguality)**：支援超過 100 種語言的文字表示，可用於跨語言的圖文檢索（文字側）。
- **多功能性 (Multi-Functionality)**：在文字檢索場景下，可按需求使用密集檢索（Dense Retrieval）、多向量檢索（Multi-Vector Retrieval）等不同正規化。
- **多粒度性 (Multi-Granularity)**：文字側可處理從短句到長達 8192 個 token 的長文件，覆蓋更廣泛的應用需求。

在技術架構上，bge-visualized-m3 會先用視覺編碼器提取影象的 **patch token**，再將其對映到與文字同維度的“影象 token”，與文字 token 一起送入 BGE 的 Transformer 編碼器進行聯合建模，最終得到可用於圖文檢索的統一向量表示。

## 四、程式碼示例

### 4.1 環境準備

**步驟1：安裝 visual_bge 模組**

```bash
# 進入 visual_bge 目錄
cd code/C3/visual_bge

# 安裝 visual_bge 模組及其依賴
pip install -e .

# 返回上級目錄
cd ..
```

**步驟2：下載模型權重**

```bash
# 執行模型下載指令碼
python download_model.py
```

模型下載指令碼會自動檢查 `../../models/bge/` 目錄下是否存在模型檔案，如果不存在則從 Hugging Face 映象站下載。

### 4.2 基礎示例

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth")
model.eval()

with torch.no_grad():
    text_emb = model.encode(text="datawhale開源組織的logo")
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale開源組織的logo")
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale開源組織的logo")

# 計算相似度
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度計算結果 ===")
print(f"純影象 vs 純影象: {sim_1}")
print(f"圖文結合1 vs 純影象: {sim_2}")
print(f"圖文結合1 vs 純文字: {sim_3}")
print(f"圖文結合1 vs 圖文結合2: {sim_4}")
```

**程式碼解讀：**

- **模型架構**: `Visualized_BGE` 是透過將影象token嵌入整合到BGE文字嵌入框架中構建的通用多模態嵌入模型，具備處理超越純文字的多模態資料的靈活性。
- **模型引數**:
  - `model_name_bge`: 指定底層BGE文字嵌入模型，繼承其強大的文字表示能力。
  - `model_weight`: Visual BGE的預訓練權重檔案，包含視覺編碼器引數。
- **多模態編碼能力**: Visual BGE提供了編碼多模態資料的多樣性，支援純文字、純影象或圖文組合的格式：
  - **純文字編碼**: 保持原始BGE模型的強大文字嵌入能力。
  - **純影象編碼**: 使用基於EVA-CLIP的視覺編碼器處理影象。
  - **圖文聯合編碼**: 將影象和文字特徵融合到統一的向量空間。
- **應用場景**: 主要用於混合模態檢索任務，包括多模態知識檢索、組合影象檢索、多模態查詢的知識檢索等。
- **相似度計算**: 使用矩陣乘法計算餘弦相似度，所有嵌入向量都被標準化到單位長度，確保相似度值在合理範圍內。

**執行結果：**

```bash
=== 相似度計算結果 ===
純影象 vs 純影象: tensor([[0.8318]])
圖文結合1 vs 純影象: tensor([[0.8291]])
圖文結合1 vs 純文字: tensor([[0.7627]])
圖文結合1 vs 圖文結合2: tensor([[0.9058]])
```

> [完整程式碼](https://github.com/datawhalechina/all-in-rag/blob/main/code/C3/01_bge_visualized.py)

## 練習

嘗試把程式碼中的部分文字替換一下，比如將`datawhale開源組織的logo`替換為`blue whale`看看結果有什麼不同。