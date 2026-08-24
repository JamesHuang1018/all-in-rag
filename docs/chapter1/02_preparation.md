# 第二節 準備工作

> 本節環境配置方面主推兩種基於瀏覽器的整合開發環境。不管是手機、平板還是電腦，隨時都可以上號執行程式碼。雖然手機平板可能體驗不佳，但勝在能用。

## 一、大模型 API 配置

### 1.1 AIHubmix API 申請

AIHubmix 是一個美國平臺，公司註冊在美國的特拉華州，一站式聚合了全球主流的 AI 模型，最新的模型通常能在釋出當天最晚不超過 1 周就會支援。完全對接相關模型的雲廠商（OpenAI 對接的是 Azure 雲，Gemini 對接的 Google 官方，Claude 對接的是 AWS，其他開源等模型是對接到各大知名雲廠商或者推理公司）。AIHubmix 的伺服器是在美國谷歌雲上採用叢集部署，同時因為完全對接雲廠商，所以穩定性非常好，有多端點路由機制，可以達到比直連官方更穩定的效果。

> AIHubmix 提供的免費模型足夠我們完成專案的學習。

1.  **訪問 AIHubmix 平臺**

    開啟瀏覽器，訪問 [AIHubmix](https://aihubmix.com/?aff=anNj)。

    ![AIHubmix](./images/1.png)

2.  **登入或註冊賬號**

    如果已有賬號，可以直接登入。如果沒有，請點選頁面右上角的註冊按鈕，使用郵箱或手機號完成註冊。

3.  **模型篩選**

    註冊完成後，來到[模型頁面](https://aihubmix.com/models)。標籤選擇`免費`，可以看到官方提供了一定數量的免費模型。而且 AIHubmix 還提供了很多嵌入和重排序的國內外模型選擇，這些在 RAG 領域都很常用。

    ![模型頁面](./images/2.png)

4.  **管理 API 金鑰**

    接著進入[金鑰管理頁面](https://console.aihubmix.com/token)，如下圖所示，預設已經有了一個金鑰可以直接複製使用。當然也可以點選 `建立 Key` 填寫名稱後重新建立一個。

    ![金鑰管理](./images/3.png)

### 1.2 DeepSeek API 申請

要使用 Deepseek 提供的大語言模型服務，你首先需要一個 API Key。下面是申請步驟：

1.  **訪問 Deepseek 開放平臺**

    開啟瀏覽器，訪問 [Deepseek 開放平臺](https://platform.deepseek.com/)。

    ![Deepseek 平臺首頁](./images/1_2_1.webp)

2.  **登入或註冊賬號**

    如果你已有賬號，請直接登入。如果沒有，請點選頁面上的註冊按鈕，使用郵箱或手機號完成註冊。

3.  **建立新的 API 金鑰**

    登入成功後，在頁面左側的導航欄中找到並點選 `API Keys`。在 API 管理頁面，點選 `建立 API key` 按鈕。輸入一個跟其他api key不重複的名稱後點選建立。

    ![建立新金鑰按鈕](./images/1_2_2.webp)

4.  **儲存 API Key**

    系統會為你生成一個新的 API 金鑰。請**立即複製**並將其儲存在一個安全的地方。

    > 注意：出於安全原因，這個金鑰只會完整顯示一次，關閉彈窗後就沒法再看到了。

    ![複製並儲存金鑰](./images/1_2_3.webp)

## 二、GitHub Codespaces 環境配置（推薦）

> 首先確定是否具有可以流暢訪問 GitHub 的網路環境，若無法流暢訪問請使用下面的Cloud Studio

GitHub Codespaces 是 GitHub 提供的一項服務，允許開發者在雲端建立、編輯和執行程式碼。它提供了一個預配置的開發環境，包括程式碼編輯器、終端、除錯工具等，可以直接在瀏覽器中使用。

### 2.1 建立Codespaces

1.  **訪問專案地址**

    開啟瀏覽器，訪問 [all-in-rag](https://github.com/datawhalechina/all-in-rag)

2.  **建立新分支**
    在專案頁面的右上角，點選 `Fork` 按鈕，建立一個新的分支。稍等一會兒即可建立成功。

    ![建立新分支1](./images/1_2_4.webp)

    ![建立新分支2](./images/1_2_5.webp)

3.  **建立Codespaces**
    在專案頁面的右上角，點選 `Code` 按鈕，然後選擇 `Codespaces` 選項卡。點選 `New codespace` 按鈕，等待新的 Codespaces 環境建立成功。

    ![建立Codespaces](./images/1_2_6.webp)

4.  **再次進入Codespaces**
    網頁關閉後，找到剛才新建的儲存庫，點選紅框框選內容即可重新進入 codespace 環境。

    ![再次進入Codespaces](./images/1_2_7.webp)

5.  **額度設定**
    找到 GitHub 的賬戶設定中的 codespace 設定，掛起時間建議根據自己情況調整（時間過長會浪費額度，免費賬號提供了單核120小時的額度）

    ![額度設定](./images/1_2_8.webp)

### 2.2 python環境配置

進入 IDE 後先選擇下方終端

![進入終端](./images/1_2_9.webp)

1.  **更新系統軟體包**

    在終端輸入下面指令：

    ```bash
    sudo apt update
    sudo apt upgrade -y
    ```

2.  **安裝Miniconda**

    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh
    ```

    - 按 Enter 閱讀許可協議
    - 輸入 `yes` 同意協議
    - 安裝路徑提示時直接按 Enter（使用預設路徑 /home/ubuntu/miniconda3）
    - 是否初始化Miniconda：輸入 `yes` 將Miniconda新增到您的PATH環境變數中。

    ```bash
    source ~/.bashrc
    conda --version
    ```

    如果顯示版本號，說明安裝成功。

### 2.3 API配置

1.  使用 `vim` 編輯器開啟你的 shell 配置檔案。

    ```bash
    vim ~/.bashrc
    ```

2.  輸入 `i` 進入編輯模式，在檔案末尾新增以下行，將 `[你的大模型 API 金鑰]` 替換為你自己的金鑰：

    ```bash
    export DEEPSEEK_API_KEY=[你的大模型 API 金鑰]
    ```

    如果選擇的是 `AIHubmix` 平臺，為了增加辨識度也可以使用：

    ```bash
    export AIHUBMIX_API_KEY=[你的大模型 API 金鑰]
    ```

    > 不要帶 `[]`

3.  儲存並退出 在 vim 中，按 Esc 鍵進入命令模式，然後輸入 `:wq` 並按 Enter 鍵儲存檔案並退出。

4.  使配置生效 執行以下命令來立即載入更新後的配置，讓環境變數生效：

    ```bash
    source ~/.bashrc
    ```

### 2.4 建立並啟用虛擬環境

1.  **建立虛擬環境**

    ```bash
    conda create --name all-in-rag python=3.12.7
    ```

    出現選項直接回車即可。

2.  **啟用虛擬環境**

    使用以下命令啟用虛擬環境：

    ```bash
    conda activate all-in-rag
    ```

3.  **依賴安裝**
    如果嚴格安裝上述流程當前應該在專案根目錄，進入code目錄安裝依賴庫

    ```bash
    cd code
    pip install -r requirements.txt
    ```

    > 如果出現關於grpcio的版本錯誤無需在意。

## 三、Cloud Studio 環境配置（國內環境推薦）

Cloud Studio 是騰訊雲推出的一款基於瀏覽器的整合開發環境（IDE）。支援CPU與GPU的訪問。

> 聽說一個月是50個小時的免費額度🤔

### 3.1 應用建立

1.  **訪問 Cloud Studio**
    開啟瀏覽器，訪問 [Cloud Studio](https://cloudstudio.net/)。

2.  **登入或註冊賬號**
    點選頁面右上角的 `註冊登入` 按鈕，使用微信等方式完成登入。

3.  **建立應用**
    在頁面上方的導航欄中找到並點選 `建立應用`。選擇 `從 Git 倉庫匯入` ，在專案位址列輸入 `https://github.com/datawhalechina/all-in-rag.git` 後回車，將會自動為你建立標題和描述。

    ![建立應用](./images/1_2_10.webp)

    > 注意描述中不要包含網址

4.  **再次進入**
    後續在[應用管理頁面](https://cloudstudio.net/my-app)找到之前建立的應用，點選後選擇右上角編寫程式碼即可再次進入。

    ![再次進入應用](./images/1_2_11.webp)

### 3.2 python環境配置

進入 IDE 後先選擇右側終端

![進入終端](./images/1_2_12.webp)

1.  **更新系統軟體包**

    在終端輸入下面指令：

    ```bash
    sudo apt update
    sudo apt upgrade -y
    ```

2.  **切換普通使用者**

    ```bash
    su ubuntu
    ```

3.  **安裝Miniconda**

    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh
    ```

    - 按 Enter 閱讀許可協議
    - 輸入 `yes` 同意協議
    - 安裝路徑提示時直接按 Enter（使用預設路徑 /home/ubuntu/miniconda3）
    - 是否初始化Miniconda：輸入 `yes` 將Miniconda新增到您的PATH環境變數中。

    ```bash
    source ~/.bashrc
    conda --version
    ```

    如果顯示版本號，說明安裝成功。

### 3.3 API配置

1.  使用 `vim` 編輯器開啟你的 shell 配置檔案。

    ```bash
    vim ~/.bashrc
    ```

2.  輸入 `i` 進入編輯模式，在檔案末尾新增以下行，將 `[你的大模型 API 金鑰]` 替換為你自己的金鑰：

    ```bash
    export DEEPSEEK_API_KEY=[你的大模型 API 金鑰]
    ```

    如果選擇的是 `AIHubmix` 平臺，為了增加辨識度也可以使用：

    ```bash
    export AIHUBMIX_API_KEY=[你的大模型 API 金鑰]
    ```

    > 不要帶 `[]`

3.  儲存並退出 在 vim 中，按 Esc 鍵進入命令模式，然後輸入 `:wq` 並按 Enter 鍵儲存檔案並退出。

4.  使配置生效 執行以下命令來立即載入更新後的配置，讓環境變數生效：

    ```bash
    source ~/.bashrc
    ```

### 3.4 建立並啟用虛擬環境

1.  **建立虛擬環境**

    ```bash
    conda create --name all-in-rag python=3.12.7
    ```

    出現選項直接回車即可。

2.  **配置檔案許可權**

    ```bash
    sudo chown -R ubuntu:ubuntu code models
    ```

3.  **啟用虛擬環境**

    使用以下命令啟用虛擬環境：

    ```bash
    conda activate all-in-rag
    ```

4.  **依賴安裝**
    如果嚴格安裝上述流程當前應該在專案根目錄，進入code目錄安裝依賴庫

    ```bash
    cd code
    pip install -r requirements.txt
    ```

    > 如果出現關於grpcio的版本錯誤無需在意。

## 四、windows環境配置（使用Cloud Studio 或 Codespaces 可跳過此步驟）

### 4.1 API配置

1.  右鍵點選 “計算機” 或 “此電腦”，然後點選 “屬性”。

2.  在左側選單中，點選 “高階系統設定”。

3.  在 “系統屬性” 對話方塊中，點選 “高階” 選項卡，然後點選下方的 “環境變數” 按鈕。

    ![高階系統設定](./images/1_2_13.webp)

4.  在 “環境變數” 對話方塊中，點選 “新建”（在 “使用者變數” 部分下），然後輸入以下資訊：
    - 變數名：DEEPSEEK_API_KEY
    - 變數值：[你的 Deepseek API 金鑰]

    ![高階系統設定](./images/1_2_14.webp)

### 4.2 安裝Miniconda

1.  **下載安裝程式**

    優先推薦訪問[清華大學開源軟體映象站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)，以獲得更快的下載速度。根據你的系統選擇最新的 `.exe` 版本下載。

    ![選擇Miniconda版本](images/1_2_15.webp)

    你也可以從 [Miniconda 官方網站](https://docs.conda.io/en/latest/miniconda.html)下載。

2.  **執行安裝嚮導**

    下載完成後，雙擊 `.exe` 檔案啟動安裝。按照嚮導提示操作：

    *   **Welcome**: 點選 `Next`。

        ![Welcome](./images/1_2_16.webp)

    *   **License Agreement**: 點選 `I Agree`。

        ![License Agreement](./images/1_2_17.webp)

    *   **Installation Type**: 選擇 `Just Me`，點選 `Next`。

        ![Installation Type](./images/1_2_18.webp)

    *   **Choose Install Location**: 建議保持預設路徑，或選擇一個不含中文和空格的路徑。點選 `Next`。

        ![Install Location](./images/1_2_19.webp)

    *   **Advanced Installation Options**: **請不要勾選** “Add Miniconda3 to my PATH environment variable”。我們將稍後手動配置環境變數。點選 `Install`。

        ![Advanced Options](./images/1_2_20.webp)

    *   **Installation Complete**: 安裝完成後，點選 `Next`，然後取消勾選 “Learn more” 並點選 `Finish` 完成安裝。

3.  **手動配置環境變數**

    為了能在任意終端視窗使用 `conda` 命令，需要手動配置環境變數。

    *   在Windows搜尋欄中搜尋“編輯系統環境變數”並開啟。

        ![編輯系統環境變數](./images/1_2_21.webp)

    *   在“系統屬性”視窗中，點選“環境變數”。

        ![環境變數按鈕](./images/1_2_22.webp)

    *   在“環境變數”視窗中，找到“系統變數”下的 `Path` 變數，選中並點選“編輯”。

        ![編輯Path變數](./images/1_2_23.webp)

    *   在“編輯環境變數”視窗中，新建三個路徑，將它們指向你 Miniconda 的安裝目錄下的相應資料夾。如果你的安裝路徑是 `D:\Miniconda3`，則需要新增：
        ```
        D:\Miniconda3
        D:\Miniconda3\Scripts
        D:\Miniconda3\Library\bin
        ```
        ![新增路徑](./images/1_2_24.webp)
        
    *   完成後，一路點選“確定”儲存更改。

### 4.3 配置 Conda 映象源

為了加快後續使用 `conda` 安裝包的速度，強烈建議配置國內映象源。開啟一個新的終端或 Anaconda Prompt，執行以下命令：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

配置完成後，可以透過 `conda config --show channels` 命令檢視已新增的源。

## 五、專案程式碼拉取（使用Cloud Studio 或 Codespaces 可跳過此步驟）

### 5.1 安裝 Git

如果你尚未安裝 Git，請按照以下步驟安裝。

* **Windows 系統**：訪問[Git 官方網站](https://git-scm.com/download/win)，下載並執行安裝程式，按照預設設定完成安裝。
* **macOS 系統**：開啟終端，輸入以下命令安裝 Git：

  ```bash
  brew install git
  ```
* **Linux 系統（以 Ubuntu 為例）**：開啟終端，輸入以下命令安裝 Git：

  ```bash
  sudo apt-get update
  sudo apt-get install git
  ```

安裝完成後，驗證 Git 是否安裝成功，輸入以下命令：

```bash
git --version
```

如果成功，會顯示 Git 的版本號。

### 5.2 克隆專案程式碼

1. **選擇存放專案的目錄**
   開啟終端（或 Windows 中的 Git Bash），導航到你想存放專案的目錄：

   ```bash
   cd [你希望存放專案的路徑]
   ```

2. **克隆倉庫**
   使用以下命令拉取 `all-in-rag` 倉庫：

   ```bash
   git clone https://github.com/datawhalechina/all-in-rag.git
   ```

   等待下載完成，專案程式碼將存放在當前目錄下的 `all-in-rag` 資料夾中。

3. **進入專案目錄**
   拉取程式碼後，進入專案目錄：

   ```bash
   cd all-in-rag
   ```

### 5.3 建立並啟用虛擬環境

在專案目錄下，推薦使用前面配置好的 Miniconda 來建立 Python 虛擬環境。

1. **建立虛擬環境**

   ```bash
   conda create --name all-in-rag python=3.12.7
   ```

2. **啟用虛擬環境**

   所有系統統一使用以下命令啟用虛擬環境：

   ```bash
   conda activate all-in-rag
   ```

3.  **依賴安裝**
    如果嚴格安裝上述流程當前應該在專案根目錄，進入code目錄安裝依賴庫

    ```bash
    cd code
    pip install -r requirements.txt
    ```
