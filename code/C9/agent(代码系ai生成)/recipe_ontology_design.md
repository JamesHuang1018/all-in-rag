# 菜譜知識圖譜設計方案
## 烹飪本體設計

### 概述
本設計方案將菜譜資料結構化為本體圖資料庫，支援語義查詢、推理和知識發現。

### 核心概念體系 (Concept Hierarchy)

#### 1. 根概念 (Root Concept)
```
100000000 | Culinary Concept | (烹飪概念)
```

#### 2. 頂級概念 (Top Level Concepts)
```
200000000 | Recipe | (菜譜)
300000000 | Ingredient | (食材)
400000000 | Tool | (工具)
500000000 | Cooking Method | (烹飪方法)
600000000 | Difficulty Level | (難度等級)
700000000 | Recipe Category | (菜譜分類)
800000000 | Measurement Unit | (計量單位)
900000000 | Cooking Step | (烹飪步驟)
```

### 概念分類體系

#### 菜譜分類 (Recipe Categories)
```
710000000 | Vegetable Dish | (素菜)
720000000 | Meat Dish | (葷菜)
730000000 | Aquatic Product | (水產)
740000000 | Breakfast | (早餐)
750000000 | Staple Food | (主食)
760000000 | Soup | (湯類)
770000000 | Dessert | (甜品)
780000000 | Beverage | (飲料)
790000000 | Condiment | (調料)
```

#### 食材分類 (Ingredient Categories)
```
310000000 | Vegetable | (蔬菜)
320000000 | Seasoning | (調料)
330000000 | Protein | (蛋白質)
340000000 | Starch | (澱粉類)
350000000 | Dairy | (乳製品)
360000000 | Fruit | (水果)
370000000 | Herb | (香草香料)
380000000 | Oil Fat | (油脂類)
```

#### 難度等級 (Difficulty Levels)
```
610000000 | One Star | (一星) ★
620000000 | Two Star | (二星) ★★
630000000 | Three Star | (三星) ★★★
640000000 | Four Star | (四星) ★★★★
650000000 | Five Star | (五星) ★★★★★
```

#### 烹飪方法 (Cooking Methods)
```
501000000 | Stir Fry | (炒)
502000000 | Deep Fry | (炸)
503000000 | Braise | (紅燒)
504000000 | Steam | (蒸)
505000000 | Boil | (煮)
506000000 | Roast | (烤)
507000000 | Stew | (燉)
508000000 | Mix | (拌)
509000000 | Marinate | (醃)
510000000 | Blanch | (焯)
```

### 關係型別定義 (Relationship Types)

#### 核心關係 (Core Relationships)
```
116680003 | is_a | (是一個) - 層次分類關係
```

#### 屬性關係 (Attribute Relationships)
```
801000001 | has_ingredient | (包含食材)
801000002 | requires_tool | (需要工具)
801000003 | has_step | (包含步驟)
801000004 | belongs_to_category | (屬於分類)
801000005 | has_difficulty | (具有難度)
801000006 | uses_method | (使用方法)
801000007 | has_amount | (具有用量)
801000008 | step_follows | (步驟順序)
801000009 | serves_people | (供應人數)
801000010 | cooking_time | (烹飪時間)
801000011 | prep_time | (準備時間)
801000012 | ingredient_substitute | (食材替代)
801000013 | recipe_variant | (菜譜變體)
801000014 | nutritional_info | (營養資訊)
```

### 具體例項設計

#### 紅燒茄子菜譜例項
```
概念ID: 201000001
完全限定名: 201000001 | 紅燒茄子 (Braised Eggplant) |
首選術語: 紅燒茄子
同義詞: 茄子燒製, 紅燒青茄子

屬性關係:
- belongs_to_category = 710000000 | 素菜
- has_difficulty = 640000000 | 四星
- serves_people = 2人份
- cooking_time = 30分鐘
- prep_time = 15分鐘

食材關係:
- has_ingredient = 311000001 | 青茄子 | : has_amount = "0.7個/份"
- has_ingredient = 311000002 | 大蒜 | : has_amount = "3瓣"
- has_ingredient = 321000001 | 醬油 | : has_amount = "茄子數量*7克"
- has_ingredient = 331000001 | 雞蛋 | : has_amount = "1個"
- has_ingredient = 341000001 | 麵粉 | : has_amount = "青茄子數量*150克"

工具關係:
- requires_tool = 401000001 | 炒鍋
- requires_tool = 401000002 | 菜刀
- requires_tool = 401000003 | 筷子

方法關係:
- uses_method = 502000000 | 炸
- uses_method = 503000000 | 紅燒
- uses_method = 501000000 | 炒

步驟關係:
- has_step = S001 | 清洗食材
- has_step = S002 | 切配處理  
- has_step = S003 | 調製麵糊
- has_step = S004 | 油炸茄塊
- has_step = S005 | 炒制調味
```

### 表示式系統

#### 預協調表示式 (Precoordinated)
```
201000001 | 紅燒茄子 |
```

#### 後協調表示式 (Postcoordinated)
```
# 四星難度的素菜
710000000 | 素菜 | : has_difficulty = 640000000 | 四星 |

# 包含茄子的紅燒菜譜
200000000 | 菜譜 | : {
    has_ingredient = 311000001 | 青茄子 |,
    uses_method = 503000000 | 紅燒 |
}

# 30分鐘內完成的四星菜譜
200000000 | 菜譜 | : {
    has_difficulty = 640000000 | 四星 |,
    cooking_time <= 30分鐘
}
```

### 資料檔案結構

#### 概念檔案 (rf2_concept.txt)
```
id	effectiveTime	active	moduleId	definitionStatusId
100000000	20241201	1	900000000	900000000
200000000	20241201	1	900000000	900000000
201000001	20241201	1	900000000	900000000
```

#### 描述檔案 (rf2_description.txt)
```
id	effectiveTime	active	moduleId	conceptId	languageCode	typeId	term	caseSignificanceId
D001	20241201	1	900000000	201000001	zh-CN	900000001	紅燒茄子	900000000
D002	20241201	1	900000000	201000001	zh-CN	900000002	茄子燒製	900000000
D003	20241201	1	900000000	201000001	en	900000001	Braised Eggplant	900000000
```

#### 關係檔案 (rf2_relationship.txt)
```
id	effectiveTime	active	moduleId	sourceId	destinationId	relationshipGroup	typeId	characteristicTypeId	modifierId
R001	20241201	1	900000000	201000001	710000000	0	801000004	900000000	900000000
R002	20241201	1	900000000	201000001	640000000	0	801000005	900000000	900000000
R003	20241201	1	900000000	201000001	311000001	1	801000001	900000000	900000000
```

### 查詢示例

#### 1. 基礎查詢 - 所有素菜
```cypher
MATCH (recipe:Concept)-[:IS_A*]->(category:Concept {conceptId: "710000000"})
RETURN recipe.preferredTerm
```

#### 2. 複雜查詢 - 包含特定食材的四星菜譜
```cypher
MATCH (recipe:Concept)-[:HAS_INGREDIENT]->(ingredient:Concept)
WHERE ingredient.conceptId = "311000001" 
AND (recipe)-[:HAS_DIFFICULTY]->(:Concept {conceptId: "640000000"})
RETURN recipe.preferredTerm, recipe.cookingTime
```

#### 3. 語義查詢 - 所有炒制類菜譜
```cypher
MATCH (recipe:Concept)-[:USES_METHOD]->(method:Concept)
WHERE (method)-[:IS_A*]->(:Concept {conceptId: "501000000"})
RETURN DISTINCT recipe.preferredTerm
```

### 實現技術棧

#### 圖資料庫選擇
- **Neo4j**: 成熟的圖資料庫，適合複雜查詢
- **ArangoDB**: 多模型資料庫，支援圖和文件
- **Amazon Neptune**: 雲原生圖資料庫

#### API設計
```python
class RecipeOntologyAPI:
    def search_recipes_by_ingredient(self, ingredient_id: str) -> List[Recipe]:
        """根據食材搜尋菜譜"""
        pass
    
    def get_recipe_variants(self, recipe_id: str) -> List[Recipe]:
        """獲取菜譜變體"""
        pass
    
    def suggest_substitutes(self, ingredient_id: str) -> List[Ingredient]:
        """建議食材替代"""
        pass
    
    def analyze_nutrition(self, recipe_id: str) -> NutritionInfo:
        """分析營養成分"""
        pass
```

### 應用場景

#### 1. 智慧菜譜推薦
- 基於現有食材推薦菜譜
- 根據難度等級篩選
- 營養搭配建議

#### 2. 食材替代建議
- 過敏原替代
- 地域性食材替換
- 營養等價替代

#### 3. 烹飪知識推理
- 步驟最佳化建議
- 工具使用指導
- 時間管理最佳化

#### 4. 營養分析
- 卡路里計算
- 營養成分分析
- 膳食搭配建議

### 擴充套件性設計

#### 多語言支援
```
zh-CN: 紅燒茄子
en-US: Braised Eggplant
ja-JP: 茄子の煮物
ko-KR: 가지조림
```

#### 地域化擴充套件
```
中式烹飪: 701000000
法式烹飪: 702000000
意式烹飪: 703000000
日式烹飪: 704000000
```

#### 個性化標籤
```
vegetarian: 素食主義
vegan: 純素
halal: 清真
kosher: 猶太教食物
gluten_free: 無麩質
```

### 質量控制

#### 概念一致性檢查
- 迴圈依賴檢測
- 概念完整性驗證
- 關係合理性校驗

#### 資料質量保證
- 重複概念檢測
- 缺失關係補充
- 術語標準化

這個設計方案提供了一個完整的菜譜知識圖譜框架，可以支援複雜的語義查詢、推理和知識發現功能。 