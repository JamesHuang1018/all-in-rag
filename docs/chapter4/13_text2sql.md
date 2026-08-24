# 第三節 文字到SQL

繼上一節探討了如何為後設資料和圖資料構建查詢後，本節將聚焦於結構化資料領域中一個常見的應用。在資料世界中，除了向量資料庫能夠處理的非結構化資料，關係型資料庫（如 MySQL, PostgreSQL, SQLite）同樣是儲存和管理結構化資料的重點。**文字到SQL（Text-to-SQL）**[^1] 正是為了打破人與結構化資料之間的語言障礙而生。它利用大語言模型（LLM）將使用者的自然語言問題，直接翻譯成可以在資料庫上執行的SQL查詢語句。

![](./images/4_3_1.webp)

## 一、業務挑戰

- **“幻覺”問題**：LLM 可能會“想象”出資料庫中不存在的表或欄位，導致生成的SQL語句無效。
- **對資料庫結構理解不足**：LLM 需要準確理解表的結構、欄位的含義以及表與表之間的關聯關係，才能生成正確的 `JOIN` 和 `WHERE` 子句。
- **處理使用者輸入的模糊性**：使用者的提問可能存在拼寫錯誤或不規範的表達（例如，“上個月的銷售冠軍是誰？”），模型需要具備一定的容錯和推理能力。

## 二、最佳化策略

1.  **提供精確的資料庫模式**：這是最基礎也是最關鍵的一步。我們需要向LLM提供資料庫中相關表的 `CREATE TABLE` 語句。這就像是給了LLM一張地圖，讓它瞭解資料庫的結構，包括表名、列名、資料型別和外來鍵關係。

2.  **提供少量高質量的示例**：在提示（Prompt）中加入一些“問題-SQL”的示例對，可以極大地提升LLM生成查詢的準確性。這相當於給了LLM幾個範例，讓它學習如何根據相似的問題構建查詢。

3.  **利用RAG增強上下文**：這是更進一步的策略。我們可以像RAGFlow一樣，為資料庫構建一個專門的“知識庫”[^2]，其中不僅包含表的DDL（資料定義語言），還可以包含：
    *   **表和欄位的詳細描述**：用自然語言解釋每個表是做什麼的，每個欄位代表什麼業務含義。
    *   **同義詞和業務術語**：例如，將使用者的“花費”對映到資料庫的 `cost` 欄位。
    *   **複雜的查詢示例**：提供一些包含 `JOIN`、`GROUP BY` 或子查詢的複雜問答對。
    當使用者提問時，系統首先從這個知識庫中檢索最相關的資訊（如相關的表結構、欄位描述、相似的Q&A），然後將這些資訊和使用者的問題一起組合成一個內容更豐富的提示，交給LLM生成最終的SQL查詢。這種方式極大地降低了“幻覺”的風險，提高了查詢的準確度。

4.  **錯誤修正與反思 (Error Correction and Reflection)**：在生成SQL後，系統會嘗試執行它。如果資料庫返回錯誤，可以將錯誤資訊反饋給LLM，讓它“反思”並修正SQL語句，然後重試。這個迭代過程可以顯著提高查詢的成功率。

## 三、實現一個簡單的Text2SQL框架

本節基於RAGFlow方案實現了一個簡單的Text2SQL框架。該框架使用Milvus向量資料庫作為知識庫，BGE-M3模型進行語義檢索，DeepSeek作為大語言模型，專門針對SQLite資料庫進行了最佳化。

![Text2SQL框架工作流程](./images/4_3_2.webp)

### 3.1 知識庫模組 (`knowledge_base.py`)

知識庫模組是整個框架的核心，負責儲存和檢索SQL相關的知識資訊。

```python
class SimpleKnowledgeBase:
    """知識庫"""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530"):
        self.milvus_uri = milvus_uri
        self.client = MilvusClient(uri=milvus_uri)
        self.embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.collection_name = "text2sql_kb"
        self._setup_collection()
```

**設計思想：**

1. **統一知識管理**：將DDL定義、Q-SQL示例和表描述三種型別的知識統一儲存在一個Milvus集合中，透過 `type` 欄位區分。

2. **語義檢索能力**：使用BGE-M3模型進行向量化，支援中英文混合的語義相似度搜尋。

```python
def _setup_collection(self):
    """設定集合"""
    # 定義欄位
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),  # ddl, qsql, description
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_function.dim["dense"])
    ]
```

**資料載入策略：**

```python
def load_data(self):
    """載入所有知識庫資料"""
    # 載入DDL資料 - 表結構定義
    # 載入Q->SQL資料 - 問答示例
    # 載入描述資料 - 表和欄位的業務描述
```

框架支援三種型別的知識：
- **DDL知識**[^3]：表的結構定義，包括欄位型別、約束等
- **Q-SQL知識**[^4]：歷史問答對，為新問題提供參考模式
- **描述知識**[^5]：表和欄位的業務含義，幫助理解資料語義

**檢索機制：**

```python
def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """搜尋相關內容"""
    query_embeddings = self.embedding_function([query])
    
    search_results = self.client.search(
        collection_name=self.collection_name,
        data=query_embeddings["dense"],
        anns_field="dense_vector",
        search_params={"metric_type": "IP"},  # 內積相似度
        limit=top_k,
        output_fields=["content", "type"]
    )
```



### 3.2 SQL生成模組 (`sql_generator.py`)

SQL生成模組負責將自然語言問題轉換為SQL查詢語句，並具備錯誤修復能力。

```python
class SimpleSQLGenerator:
    """簡化的SQL生成器"""
    
    def __init__(self, api_key: str = None):
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,  # 確保結果的確定性
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY")
        )
```

**SQL生成策略：**

```python
def generate_sql(self, user_query: str, knowledge_results: List[Dict[str, Any]]) -> str:
    """生成SQL語句"""
    # 構建上下文
    context = self._build_context(knowledge_results)
    
    # 構建提示
    prompt = f"""你是一個SQL專家。請根據以下資訊將使用者問題轉換為SQL查詢語句。

資料庫資訊：
{context}

使用者問題：{user_query}

要求：
1. 只返回SQL語句，不要包含任何解釋
2. 確保SQL語法正確
3. 使用上下文中提供的表名和欄位名
4. 如果需要JOIN，請根據表結構進行合理關聯

SQL語句："""
```

**關鍵設計原則：**

1. **上下文驅動**：透過知識庫檢索結果構建豐富的上下文資訊
2. **結構化提示**：明確的任務要求和格式約束
3. **確定性輸出**：設定temperature=0確保相同輸入產生相同輸出

**錯誤修復機制：**

```python
def fix_sql(self, original_sql: str, error_message: str, knowledge_results: List[Dict[str, Any]]) -> str:
    """修復SQL語句"""
    context = self._build_context(knowledge_results)
    
    prompt = f"""請修復以下SQL語句的錯誤。

資料庫資訊：
{context}

原始SQL：
{original_sql}

錯誤資訊：
{error_message}

請返回修復後的SQL語句（只返回SQL，不要解釋）："""
```

**上下文構建策略：**

```python
def _build_context(self, knowledge_results: List[Dict[str, Any]]) -> str:
    """構建上下文資訊"""
    context = ""

    # 按型別分組
    ddl_info = []        # 表結構資訊
    qsql_examples = []   # 查詢示例
    descriptions = []    # 表描述資訊

    for result in knowledge_results:
        if result["type"] == "ddl":
            ddl_info.append(result["content"])
        elif result["type"] == "qsql":
            qsql_examples.append(result["content"])
        elif result["type"] == "description":
            descriptions.append(result["content"])

    # 分層次組織資訊：結構 → 描述 → 示例
    if ddl_info:
        context += "=== 表結構資訊 ===\n"
        context += "\n".join(ddl_info) + "\n\n"

    if descriptions:
        context += "=== 表和欄位描述 ===\n"
        context += "\n".join(descriptions) + "\n\n"

    if qsql_examples:
        context += "=== 查詢示例 ===\n"
        context += "\n".join(qsql_examples) + "\n\n"

    return context
```



### 3.3 代理模組 (`text2sql_agent.py`)

代理模組是整個框架的控制中心，協調知識庫檢索、SQL生成和執行的完整流程。

```python
class SimpleText2SQLAgent:
    """Text2SQL代理"""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530", api_key: str = None):
        self.knowledge_base = SimpleKnowledgeBase(milvus_uri)
        self.sql_generator = SimpleSQLGenerator(api_key)
        
        # 配置引數
        self.max_retry_count = 3      # 最大重試次數
        self.top_k_retrieval = 5      # 檢索數量
        self.max_result_rows = 100    # 結果行數限制
```

**主要查詢流程：**

```python
def query(self, user_question: str) -> Dict[str, Any]:
    """執行Text2SQL查詢"""
    # 1. 從知識庫檢索相關資訊
    knowledge_results = self.knowledge_base.search(user_question, self.top_k_retrieval)
    
    # 2. 生成SQL語句
    sql = self.sql_generator.generate_sql(user_question, knowledge_results)
    
    # 3. 執行SQL（帶重試機制）
    retry_count = 0
    while retry_count < self.max_retry_count:
        success, result = self._execute_sql(sql)
        
        if success:
            return {"success": True, "sql": sql, "results": result}
        else:
            # 嘗試修復SQL
            sql = self.sql_generator.fix_sql(sql, result, knowledge_results)
            retry_count += 1
```

**安全執行策略：**

```python
def _execute_sql(self, sql: str) -> Tuple[bool, Any]:
    """執行SQL語句"""
    # 新增LIMIT限制，防止大量資料返回
    if sql.strip().upper().startswith('SELECT') and 'LIMIT' not in sql.upper():
        sql = f"{sql.rstrip(';')} LIMIT {self.max_result_rows}"
    
    # 結構化結果返回
    if sql.strip().upper().startswith('SELECT'):
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result_row = {}
            for i, value in enumerate(row):
                result_row[columns[i]] = value
            results.append(result_row)
        
        return True, {"columns": columns, "rows": results, "count": len(results)}
```

### 3.4 完整流程模擬

以查詢"年齡大於30的使用者有哪些"為例，演示框架三個核心模組的完整協作過程：

#### 3.4.1 模擬資料

假設資料庫中的users表包含以下使用者資料：

| ID | 姓名 | 郵箱 | 年齡 | 城市 |
|----|------|------|------|------|
| 1 | 張三 | zhangsan@email.com | 25 | 北京 |
| 2 | 李四 | lisi@email.com | 32 | 上海 |
| 3 | 王五 | wangwu@email.com | 28 | 廣州 |
| 4 | 趙六 | zhaoliu@email.com | 35 | 深圳 |
| 5 | 陳七 | chenqi@email.com | 29 | 杭州 |

#### 3.4.2 Step 1: 知識庫檢索

**使用者輸入**："年齡大於30的使用者有哪些"

**檢索過程**：
1. BGE-M3模型將查詢文字轉換為768維向量
2. Milvus在知識庫中進行語義相似度搜尋
3. 返回最相關的5條知識，按相似度排序

**檢索結果**：

**DDL知識** (相似度: 0.85)
- 表名：users
- 結構：包含id、name、email、age、city欄位
- 約束：id為主鍵，email唯一

**Q-SQL示例** (相似度: 0.82)  
- 問題："查詢年齡超過25歲的使用者"
- SQL：`SELECT * FROM users WHERE age > 25`
> 這是檢索到的相似示例，最終SQL會基於使用者實際問題調整為age > 30

**表描述** (相似度: 0.78)
- age欄位：使用者年齡，整數型別
- name欄位：使用者姓名，文字型別

#### 3.4.3 Step 2: SQL生成

**上下文構建**：
系統將檢索到的知識整理成結構化的上下文資訊：

**表結構資訊**
- 表名：users
- DDL定義：完整的CREATE TABLE語句
- 欄位約束：主鍵、唯一性等

**表和欄位描述**  
- age欄位：使用者年齡，INTEGER型別
- name欄位：使用者姓名，TEXT型別

**查詢示例**
- 相似問題：查詢年齡超過25歲的使用者
- 參考SQL：`SELECT * FROM users WHERE age > 25`

**SQL生成過程**：
1. DeepSeek分析使用者問題的意圖：查詢滿足年齡條件的使用者
2. 識別關鍵資訊：年齡欄位（age）、比較操作（大於）、閾值（**30**）
3. 參考示例模式：從`WHERE age > 25`學習到`WHERE age > 數值`的模式
4. 模式應用：將使用者的實際數值30替換示例中的25
5. 生成目標SQL：`SELECT * FROM users WHERE age > 30`

#### 3.4.4 Step 3: SQL執行與結果處理

**安全處理**：
- 原始SQL：`SELECT * FROM users WHERE age > 30`
- 自動新增限制：`SELECT * FROM users WHERE age > 30 LIMIT 100`

**資料庫執行**：
SQLite引擎逐行檢查users表中的資料：

| 使用者 | 年齡檢查 | 結果 |
|------|----------|------|
| 張三 | 25 > 30? | ❌ 不符合 |
| 李四 | 32 > 30? | ✅ 符合 |
| 王五 | 28 > 30? | ❌ 不符合 |
| 趙六 | 35 > 30? | ✅ 符合 |
| 陳七 | 29 > 30? | ❌ 不符合 |

**結果處理**：
- 篩選出2條符合條件的記錄
- 轉換為結構化JSON格式
- 包含欄位名稱和資料型別資訊

**最終輸出**：
```json
{
    "success": true,
    "error": null,
    "sql": "SELECT * FROM users WHERE age > 30 LIMIT 100",
    "results": {
        "columns": ["id", "name", "email", "age", "city"],
        "rows": [
            {"id": 2, "name": "李四", "email": "lisi@email.com", "age": 32, "city": "上海"},
            {"id": 4, "name": "趙六", "email": "zhaoliu@email.com", "age": 35, "city": "深圳"}
        ],
        "count": 2
    },
    "retry_count": 0
}
```

透過這個**語義理解 → 結構化查詢 → 資料過濾 → 結果輸出**的完整流程，框架成功將使用者的自然語言問題轉換為精確的資料庫查詢結果。

### 3.5 程式碼執行

如果你想測試這個Text2SQL框架，可以透過以下方式進行：

**快速體驗**：執行演示程式
```bash
python code/C4/03_text2sql_demo.py
```
> 完整演示程式碼：[03_text2sql_demo.py](https://github.com/datawhalechina/all-in-rag/blob/main/code/C4/03_text2sql_demo.py)

**核心程式碼獲取**：三個核心模組的完整實現
- `knowledge_base.py` - 知識庫模組
- `sql_generator.py` - SQL生成模組  
- `text2sql_agent.py` - 代理協調模組

> 原始碼地址：[code/C4/text2sql/](https://github.com/datawhalechina/all-in-rag/tree/main/code/C4/text2sql)

**資料資源**：框架使用的JSON知識資料
- `ddl_examples.json` - DDL結構示例
- `qsql_examples.json` - 問題-SQL對示例
- `db_descriptions.json` - 表和欄位描述

> 資料檔案：[code/C4/text2sql/data/](https://github.com/datawhalechina/all-in-rag/tree/main/code/C4/text2sql/data)

### 3.6 為什麼不直接使用封裝好的框架？

> 因為淋過雨，所以想為你撐把傘🤪

市面上確實有很多成熟的Text2SQL框架，但這些高度封裝的工具往往存在**黑盒問題**——當查詢結果不符合預期時，很難定位是檢索環節、SQL生成環節還是執行環節出了問題。正如上一節LangChain示例中遇到的查詢異常，我們很難深入到框架內部進行精確除錯和最佳化。這一點在索引最佳化那節中也提到過。

## 參考文獻

[^1]: [*LangChain Docs: Text to SQL*](https://python.langchain.com/docs/tutorials/sql_qa/)

[^2]: [*RAGFlow Blog: Implementing Text2SQL with RAGFlow*](https://ragflow.io/blog/implementing-text2sql-with-ragflow)

[^3]: DDL（Data Definition Language）是資料定義語言，用於定義資料庫結構，如CREATE TABLE語句。

[^4]: Q-SQL示例是指"問題-SQL"對，即自然語言問題與對應SQL查詢的配對示例，用於少樣本學習。

[^5]: 表描述是對資料庫表和欄位的業務語義說明，幫助模型理解資料的實際含義和用途。
