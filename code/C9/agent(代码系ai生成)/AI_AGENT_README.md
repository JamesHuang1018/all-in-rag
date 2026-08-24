# AI菜譜知識圖譜生成器
## 基於Kimi大模型的智慧菜譜解析系統

### 🌟 新功能特性

- **🤖 AI智慧解析**: 使用Kimi大模型準確提取菜譜資訊
- **📊 結構化輸出**: 自動生成標準化的知識圖譜資料
- **🔄 批次處理**: 支援大規模菜譜目錄的批次轉換
- **💾 多格式匯出**: 支援Neo4j和CSV兩種輸出格式
- **🎯 高精度**: AI模型確保食材分類、步驟解析的準確性
- **📁 智慧目錄識別**: 自動掃描dishes/目錄，根據子目錄名推斷分類
- **⚡ 最佳化處理**: 減少AI分類工作，提高處理效率和準確性

### 🚀 快速開始

#### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

#### 2. 配置API金鑰

方法一：設定環境變數
```bash
export KIMI_API_KEY="your_api_key_here"
```

方法二：編輯config.json
```json
{
  "kimi": {
    "api_key": "your_api_key_here"
  }
}
```

#### 3. 測試AI解析功能

```bash
python run_ai_agent.py test
```

#### 4. 批次處理菜譜

```bash
# 處理HowToCook專案（會自動掃描dishes/目錄）
python run_ai_agent.py /path/to/HowToCook-master

# 處理其他菜譜目錄
python run_ai_agent.py /path/to/your/recipes
```

### 📁 專案結構

```
cook-rag-example/
├── recipe_ai_agent.py         # AI解析核心引擎
├── run_ai_agent.py            # 簡化執行指令碼
├── batch_manager.py           # 批次管理工具
├── amount_normalizer.py       # 用量標準化工具
├── config.json                # 配置檔案
├── requirements.txt           # 依賴列表
└── ai_output/                 # AI輸出目錄
    ├── nodes.csv              # Neo4j節點資料
    ├── relationships.csv      # Neo4j關係資料
    └── neo4j_import.cypher    # Neo4j匯入指令碼
```

### 📁 智慧目錄處理

#### 自動分類識別

系統會智慧識別HowToCook專案的目錄結構：

- **專門掃描**: 只處理 `dishes/` 目錄下的菜譜檔案
- **自動分類**: 根據子目錄名自動推斷菜譜分類
- **排除無關檔案**: 自動跳過 `template/`、`.github/` 等目錄

#### 目錄分類對映

```
vegetable_dish/ → 素菜
meat_dish/     → 葷菜  
aquatic/       → 水產
breakfast/     → 早餐
staple/        → 主食
soup/          → 湯類
dessert/       → 甜品
drink/         → 飲料
condiment/     → 調料
semi-finished/ → 半成品
```

#### 處理最佳化

- **減少AI工作量**: 分類資訊直接從目錄結構獲取，無需AI推理
- **提高準確性**: 避免AI分類錯誤，確保分類一致性
- **加快處理速度**: 減少API呼叫複雜度和處理時間

### 🤖 AI解析能力

#### 智慧資訊提取

AI系統能夠從Markdown菜譜中準確提取：

1. **基本資訊**
   - 菜譜名稱
   - 難度等級（1-5星）
   - 菜譜分類（素菜/葷菜/水產等）
   - 菜系歸屬（川菜/粵菜等）

2. **食材資訊** 
   - 食材名稱和分類
   - 用量和單位
   - 主要食材識別

3. **烹飪步驟**
   - 步驟描述和順序
   - 使用的烹飪方法
   - 需要的工具
   - 時間估計

4. **額外資訊**
   - 準備時間/烹飪時間
   - 供應人數
   - 營養資訊（當可用時）
   - 相關標籤

#### 示例：AI解析結果

輸入菜譜：
```markdown
# 紅燒茄子的做法
預估烹飪難度：★★★★
## 必備原料和工具
- 青茄子
- 大蒜
- 醬油
- 麵粉
```

AI解析輸出：
```json
{
  "name": "紅燒茄子",
  "difficulty": 4,
  "category": "素菜",
  "ingredients": [
    {
      "name": "青茄子",
      "category": "蔬菜",
      "is_main": true
    },
    {
      "name": "大蒜", 
      "category": "蔬菜",
      "is_main": false
    }
  ]
}
```

### 📊 知識圖譜結構

#### Neo4j資料模型

**節點型別**:
- `Recipe`: 菜譜
- `Ingredient`: 食材  
- `CookingStep`: 烹飪步驟

**關係型別**:
- `has_ingredient`: 菜譜包含食材
- `has_step`: 菜譜包含步驟
- `belongs_to_category`: 屬於分類
- `has_difficulty`: 具有難度

#### 示例查詢

```cypher
// 查詢所有包含茄子的菜譜
MATCH (recipe:Concept)-[:has_ingredient]->(ing:Concept)
WHERE ing.name CONTAINS "茄子"
RETURN recipe.name, recipe.difficulty

// 查詢四星難度的素菜
MATCH (recipe:Concept)-[:belongs_to_category]->(cat:Concept)
WHERE cat.name = "素菜" AND recipe.difficulty = 4
RETURN recipe.name

// 推薦基於現有食材的菜譜
MATCH (recipe:Concept)-[:has_ingredient]->(ing:Concept)
WHERE ing.name IN ["茄子", "大蒜", "醬油"]
WITH recipe, count(ing) as matches
WHERE matches >= 2
RETURN recipe.name, matches
ORDER BY matches DESC
```

### ⚙️ 配置選項

#### config.json詳細配置

```json
{
  "deepseek": {
    "api_key": "your_api_key",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "max_retries": 3,
    "timeout": 30
  },
  "processing": {
    "batch_size": 10,
    "delay_between_requests": 1,
    "max_concurrent_requests": 5
  },
  "output": {
    "format": "neo4j",
    "directory": "./ai_output",
    "include_nutrition": true,
    "include_tags": true
  }
}
```

### �️ 輔助工具

#### 批次管理工具 (batch_manager.py)

用於管理分批處理的進度和資料：

```bash
# 檢視處理狀態
python batch_manager.py status

# 繼續中斷的處理
python batch_manager.py continue /path/to/recipes

# 合併批次資料
python batch_manager.py merge

# 清理進度檔案
python batch_manager.py clean-progress

# 清理批次資料
python batch_manager.py clean-batches

# 顯示批次詳情
python batch_manager.py details
```

#### 用量標準化工具 (amount_normalizer.py)

提供食材用量的標準化處理：

```python
from amount_normalizer import AmountNormalizer

normalizer = AmountNormalizer()

# 標準化用量
normalized, estimated = normalizer.normalize_amount("適量", "毫升")
# 返回: ("適量", 10.0)

# 獲取可比較的數值
comparable = normalizer.get_comparable_value("一把", "")
# 返回: 50.0

# 格式化顯示
display = normalizer.format_for_display("2-3個", "")
# 返回: "2-3個"
```

### �🔧 高階用法

#### 1. 自定義分類對映

```python
# 在recipe_ai_agent.py中修改
category_mapping = {
    "素菜": "710000000",
    "葷菜": "720000000", 
    "自定義分類": "999000000"
}
```

#### 2. 擴充套件AI提示詞

```python
# 修改extract_recipe_info方法中的prompt
prompt = f"""
請分析菜譜並按以下格式提取資訊：
- 新增您的自定義要求
- 特殊的分類規則
- 額外的營養資訊要求
"""
```

#### 3. 批次處理最佳化

```python
# 調整處理引數
builder = RecipeKnowledgeGraphBuilder(ai_agent)
builder.batch_size = 20  # 增加批次大小
builder.delay = 0.5      # 減少請求間隔
```

### 📈 效能最佳化

#### API呼叫最佳化

1. **合理的請求頻率**: 預設每秒1次請求，避免API限制
2. **錯誤重試機制**: 自動重試失敗的請求
3. **批次處理**: 分批處理大量菜譜檔案

#### 記憶體最佳化

1. **流式處理**: 逐個處理菜譜檔案，避免記憶體溢位
2. **定期清理**: 處理完成後及時釋放記憶體
3. **進度監控**: 實時顯示處理進度

### 🔍 故障排除

#### 常見問題

1. **API金鑰錯誤**
   ```
   錯誤: API呼叫失敗: 401
   解決: 檢查API金鑰是否正確設定
   ```

2. **網路連線問題**
   ```
   錯誤: API呼叫超時
   解決: 檢查網路連線，或增加timeout設定
   ```

3. **JSON解析錯誤**
   ```
   錯誤: JSON解析錯誤
   解決: AI響應格式異常，會自動使用備用解析方法
   ```

4. **菜譜格式問題**
   ```
   錯誤: 無法提取菜譜資訊
   解決: 檢查Markdown格式是否符合要求
   ```

#### 除錯模式

```bash
# 啟用詳細日誌
export DEBUG=true
python run_ai_agent.py /path/to/recipes

# 測試單個菜譜
python run_ai_agent.py test
```

### 📋 使用場景

#### 1. 菜譜網站構建
- 自動分類和標籤
- 智慧推薦系統
- 營養分析

#### 2. 烹飪應用開發
- 食材識別
- 步驟指導
- 工具推薦

#### 3. 營養研究
- 食材營養分析
- 膳食搭配研究
- 健康飲食推薦

#### 4. 餐飲業務
- 選單最佳化
- 成本分析
- 客戶偏好分析

### 🌐 擴充套件開發

#### 新增新的AI模型

```python
class CustomAIAgent(DeepSeekRecipeAgent):
    def __init__(self, api_key, model_name="custom-model"):
        super().__init__(api_key)
        self.model_name = model_name
    
    def call_custom_api(self, messages):
        # 實現您的自定義AI呼叫邏輯
        pass
```

#### 支援新的輸出格式

```python
def export_to_custom_format(self, output_dir):
    """匯出為自定義格式"""
    # 實現您的匯出邏輯
    pass
```

### 📊 資料質量保證

#### AI解析準確性

- **多輪驗證**: AI提取後進行格式驗證
- **備用解析**: AI失敗時使用規則解析
- **人工稽核**: 提供資料稽核介面

#### 資料一致性

- **標準化分類**: 統一的食材和菜譜分類
- **關係驗證**: 確保圖譜關係的邏輯一致性
- **重複檢測**: 自動識別和處理重複資料

---

**享受AI驅動的菜譜知識圖譜構建體驗！** 🎉 

現在你可以安全地進行批次處理了！使用以下命令處理整個HowToCook專案：

```bash
python run_ai_agent.py HowToCook-master
```

系統會：
1. 🎯 專門掃描 `HowToCook-master/dishes/` 目錄
2. 📁 根據子目錄自動識別分類 (vegetable_dish→素菜, meat_dish→葷菜等)
3. 🚫 自動排除template、.github等非菜譜目錄
4. 🤖 用AI智慧解析每個菜譜的詳細資訊
5. 📊 生成完整的知識圖譜資料