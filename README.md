# All-in-RAG | 大模型應用開發實戰一：RAG技術全棧指南

<div align='center'>
  <img src="./docs/logo.svg" alt="All-in-RAG Logo" width="70%">
</div>

<div align="center">
  <h2>🔍 檢索增強生成 (RAG) 技術全棧指南</h2>
  <p><em>從理論到實踐，從基礎到進階，構建你的RAG技術體系</em></p>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/datawhalechina/all-in-rag?style=for-the-badge&logo=github&color=ff6b6b" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/datawhalechina/all-in-rag?style=for-the-badge&logo=github&color=4ecdc4" alt="GitHub forks"/>
  <img src="https://img.shields.io/badge/Python-3.12.7-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <a href="https://zread.ai/datawhalechina/all-in-rag">
    <img src="https://img.shields.io/badge/Ask_Zread-_.svg?style=for-the-badge&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff" alt="zread"/>
  </a>
</div>

<div align="center">
  <a href="https://datawhalechina.github.io/all-in-rag/">
    <img src="https://img.shields.io/badge/📖_線上閱讀-立即開始-success?style=for-the-badge&logoColor=white" alt="線上閱讀"/>
  </a>
  <a href="README_en.md">
    <img src="https://img.shields.io/badge/🌍_English-Version-blue?style=for-the-badge&logoColor=white" alt="English Version"/>
  </a>
  <a href="https://github.com/datawhalechina">
    <img src="https://img.shields.io/badge/💬_討論交流-加入我們-purple?style=for-the-badge&logoColor=white" alt="討論交流"/>
  </a>
</div>

<div align="center">
  <br>
  <table>
    <tr>
      <td align="center">🎯 <strong>系統化學習</strong><br>完整的RAG技術體系</td>
      <td align="center">🛠️ <strong>動手實踐</strong><br>豐富的專案案例</td>
      <td align="center">🚀 <strong>生產就緒</strong><br>工程化最佳實踐</td>
      <td align="center">📊 <strong>多模態支援</strong><br>文字+影象檢索</td>
    </tr>
  </table>
</div>

## 專案簡介（中文 | [English](README_en.md)）

本專案是一個面向大模型應用開發者的RAG（檢索增強生成）技術全棧教程，旨在透過體系化的學習路徑和動手實踐專案，幫助開發者掌握基於大語言模型的RAG應用開發技能，構建生產級的智慧問答和知識檢索系統。

**主要內容包括：**

1. **RAG技術基礎**：深入淺出地介紹RAG的核心概念、技術原理和應用場景
2. **資料處理全流程**：從資料載入、清洗到文字分塊的完整資料準備流程
3. **索引構建與最佳化**：向量嵌入、多模態嵌入、向量資料庫構建及索引最佳化技術
4. **檢索技術進階**：混合檢索、查詢構建、Text2SQL等高階檢索技術
5. **生成整合與評估**：格式化生成、系統評估與最佳化方法
6. **專案實戰**：從基礎到進階的完整RAG應用開發實踐

## 專案意義

隨著大語言模型的快速發展，RAG技術已成為構建智慧問答系統、知識檢索應用的核心技術。然而，現有的RAG教程往往零散且缺乏系統性，初學者難以形成完整的技術體系認知。

本專案從實踐出發，結合最新的RAG技術發展趨勢，構建了一套完整的RAG學習體系，幫助開發者：
- 系統掌握RAG技術的理論基礎和實踐技能
- 理解RAG系統的完整架構和各元件的作用
- 具備獨立開發RAG應用的能力
- 掌握RAG系統的評估和最佳化方法

## 專案受眾

**本專案適合以下人群學習：**
- 具備Python程式設計基礎，對RAG技術感興趣的開發者
- 希望系統學習RAG技術的AI工程師
- 想要構建智慧問答系統的產品開發者
- 對檢索增強生成技術有學習需求的研究人員

**前置要求：**
- 掌握Python基礎語法和常用庫的使用
- 能夠簡單使用docker
- 瞭解基本的LLM概念（推薦但非必需）
- 具備基礎的Linux命令列操作能力

## 專案亮點

1. **體系化學習路徑**：從基礎概念到高階應用，構建完整的RAG技術學習體系
2. **理論與實踐並重**：每個章節都包含理論講解和程式碼實踐，確保學以致用
3. **多模態支援**：不僅涵蓋文字RAG，還包括多模態嵌入和檢索技術
4. **工程化導向**：注重實際應用中的工程化問題，包括效能最佳化、系統評估等
5. **豐富的實戰專案**：提供從基礎到進階的多個實戰專案，幫助鞏固學習成果

## 內容大綱

### 第一部分：RAG基礎入門

**第一章 解鎖RAG** [📖 檢視章節](./docs/chapter1)
- [x] [RAG簡介](./docs/chapter1/01_RAG_intro.md) - RAG技術概述與應用場景
- [x] [準備工作](./docs/chapter1/02_preparation.md) - 環境配置與準備
- [x] [四步構建RAG](./docs/chapter1/03_get_start_rag.md) - 快速上手RAG開發
- [x] [附：環境部署](./docs/chapter1/virtualenv.md) - Python虛擬環境部署方案補充 (貢獻者: [@anarchysaiko](https://github.com/anarchysaiko))

**第二章 資料準備** [📖 檢視章節](./docs/chapter2)
- [x] [資料載入](./docs/chapter2/04_data_load.md) - 多格式文件處理與載入
- [x] [文字分塊](./docs/chapter2/05_text_chunking.md) - 文字切分策略與最佳化

### 第二部分：索引構建與最佳化

**第三章 索引構建** [📖 檢視章節](./docs/chapter3)
- [x] [向量嵌入](./docs/chapter3/06_vector_embedding.md) - 文字向量化技術詳解
- [x] [多模態嵌入](./docs/chapter3/07_multimodal_embedding.md) - 圖文多模態向量化
- [x] [向量資料庫](./docs/chapter3/08_vector_db.md) - 向量儲存與檢索系統
- [x] [Milvus實踐](./docs/chapter3/09_milvus.md) - Milvus多模態檢索實戰
- [x] [索引最佳化](./docs/chapter3/10_index_optimization.md) - 索引效能調優技巧

### 第三部分：檢索技術進階

**第四章 檢索最佳化** [📖 檢視章節](./docs/chapter4)
- [x] [混合檢索](./docs/chapter4/11_hybrid_search.md) - 稠密+稀疏檢索融合
- [x] [查詢構建](./docs/chapter4/12_query_construction.md) - 智慧查詢理解與構建
- [x] [Text2SQL](./docs/chapter4/13_text2sql.md) - 自然語言轉SQL查詢
- [x] [查詢重構與分發](./docs/chapter4/14_query_rewriting.md) - 查詢最佳化策略
- [x] [檢索進階技術](./docs/chapter4/15_advanced_retrieval_techniques.md) - 高階檢索演算法

### 第四部分：生成與評估

**第五章 生成整合** [📖 檢視章節](./docs/chapter5)
- [x] [格式化生成](./docs/chapter5/16_formatted_generation.md) - 結構化輸出與格式控制

**第六章 RAG系統評估** [📖 檢視章節](./docs/chapter6)
- [x] [評估介紹](./docs/chapter6/18_system_evaluation.md) - RAG系統評估方法論
- [x] [評估工具](./docs/chapter6/19_common_tools.md) - 常用評估工具與指標

### 第五部分：高階應用與實戰

**第七章 高階RAG架構（拓展部分）** [📖 檢視章節](./docs/chapter7)

- [x] [基於知識圖譜的RAG](./docs/chapter7/20_kg_rag.md)

**第八章 專案實戰一** [📖 檢視章節](./docs/chapter8)
- [x] [環境配置與專案架構](./docs/chapter8/01_env_architecture.md)
- [x] [資料準備模組實現](./docs/chapter8/02_data_preparation.md)
- [x] [索引構建與檢索最佳化](./docs/chapter8/03_index_retrieval.md)
- [x] [生成整合與系統整合](./docs/chapter8/04_generation_sys.md)

**第九章 專案實戰一最佳化（選修篇）** [📖 檢視章節](./docs/chapter9)

[🍽️ 專案展示](https://github.com/FutureUnreal/What-to-eat-today)
- [x] [圖RAG架構設計](./docs/chapter9/01_graph_rag_architecture.md)
- [x] [圖資料建模與準備](./docs/chapter9/02_graph_data_modeling.md)
- [x] [Milvus索引構建](./docs/chapter9/03_index_construction.md)
- [x] [智慧查詢路由與檢索策略](./docs/chapter9/04_intelligent_query_routing.md)

**第十章 專案實戰二（選修篇）** [📖 檢視章節](./docs/chapter10) *規劃中*

### Extra-chapter

- [Neo4J 簡單應用](./Extra-chapter/Neo4J-Simple-Application/readme.md) （貢獻者: [dalvqw](https://github.com/FutureUnreal)）
- [多模態 Omni Embedding 實踐（Jina v5-omni）](./Extra-chapter/multimodal-embedding-omni-practice/readme.md)（最佳化中）

> 如果你在使用 RAG / 向量資料庫 / Agentic RAG 等相關技術時，也有值得分享的經驗與專題內容，非常歡迎以獨立章節的形式投稿到 [Extra Chapter](./Extra-chapter/) 中。提交前請先閱讀 Extra Chapter 的[貢獻與 PR 指南](./Extra-chapter/README.md)，我們會根據內容的完整度、實踐深度與參考價值綜合評估是否合併，並視情況在主教程中進行引用或擴充套件說明。

## 目錄結構說明

```
all-in-rag/
├── docs/           # 教程文件
├── code/           # 程式碼示例
├── data/           # 示例資料
├── models/         # 預訓練模型
├── Extra-chapter/  # 擴充套件章節與社群實踐內容
└── README.md       # 專案說明
```

## 實戰專案展示

### 第八章 專案一：

![專案一](./project01.png)

### 第九章 專案一（Graph RAG最佳化）：

![專案一（Graph RAG最佳化）](./project01_graph.png)

### 第十章 專案二：

## 致謝

**核心貢獻者**
- [dalvqw-專案負責人](https://github.com/FutureUnreal)（專案發起人與主要貢獻者）

**額外章節貢獻者**
- [孫超-內容創作者](https://github.com/anarchysaiko)（Datawhale成員-上海工程技術大學）

### 特別感謝
- 感謝 [@Sm1les](https://github.com/Sm1les) 對本專案的幫助與支援
- 感謝所有為本專案做出貢獻的開發者們
- 感謝開源社群提供的優秀工具和框架支援
- 特別感謝以下為教程做出貢獻的開發者！

[![Contributors](https://contrib.rocks/image?repo=datawhalechina/all-in-rag)](https://github.com/datawhalechina/all-in-rag/graphs/contributors)

*Made with [contrib.rocks](https://contrib.rocks).*

## 參與貢獻

我們歡迎所有形式的貢獻，包括但不限於：

- 🚨 **Bug報告**：發現問題請提交 [Issue](https://github.com/datawhalechina/all-in-rag/issues)
- 💭 **教程建議**：有好的想法歡迎在 [Discussions](https://github.com/datawhalechina/all-in-rag/discussions) 中討論
- 📚 **文件改進**：幫助完善文件內容和示例程式碼（當前僅支援 Extra-chapter 優質內容pr）

## Star History

[![all-in-rag stats](https://datawhalechina.github.io/members-visualization/badges/all-in-rag.png)](https://datawhalechina.github.io/members-visualization/repo-badge?repo=all-in-rag)

<div align="center">
  <p>如果這個專案對你有幫助，請給我們一個 ⭐️</p>
  <p>讓更多人發現這個專案（護食？發來！）</p>
</div>

![star](./emoji.png)

## 關於 Datawhale

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" alt="Datawhale" width="30%">
    <p>掃描二維碼關注 Datawhale 公眾號，獲取更多優質開源內容</p>
</div>

---

## 許可證

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知識共享許可協議" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

本作品採用 [知識共享署名-非商業性使用-相同方式共享 4.0 國際許可協議](http://creativecommons.org/licenses/by-nc-sa/4.0/) 進行許可。

---
