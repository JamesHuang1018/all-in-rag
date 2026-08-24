# 附：Python虛擬環境部署方案補充

本專案由於涉及到的包過多，並且**依賴問題**自始至終都是Python的老大難問題，這就造成了 Python 工程化方面生態非常割裂。

對於這種問題，uv提供了統一的虛擬環境管理入口，同時吸收了 Rust 語言先進的包管理經驗，使用它可以減少我們在 Python 工程方面折騰的時間，下面我們使用uv來安裝專案的虛擬環境。

## 1.1 uv 環境管理

### 1.1.1 Windows 系統

**使用powershell 安裝 uv**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**安裝成功後，按照提示輸入以下命令新增環境變數**。這裡注意，**不同的人的安裝路徑不同**，請按照提示自行復制貼上命令。


```bash
$env:Path = "C:\Users\michaelbradley\.local\bin;$env:Path" 
```

![安裝成功的提示](./images/1_4_1.webp)


**輸入 uv 命令，如果出現以下提示，說明安裝成功**

![成功安裝uv](./images/1_4_2.webp)

### 1.1.2 Linux / MacOS 系統

**使用 curl 安裝 uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果無法使用 curl 命令，**使用 wget 命令**安裝 uv

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

**輸入 uv 命令，如果出現以下提示，說明安裝成功**

![成功安裝uv](./images/1_4_3.webp)


## 1.2 建立並啟用虛擬環境

### 1.2.1  **建立虛擬環境**

```bash
uv venv rag --python 3.12.7
```

程式碼建立的虛擬環境名稱為 rag ，使用Python版本為 3.12.7

Windows 系統建立成功後顯示如下資訊：

```bash
PS C:\Users\parallel> uv venv rag --python 3.12.7
Using CPython 3.12.7
Creating virtual environment at: rag
Activate with: rag\Scripts\activate
```

Linux / MacOS 系統建立成功後顯示如下資訊：

```bash
┌──(parallel㉿pacman)-[~/桌面]
└─$ uv venv rag -p 3.12.7
Using CPython 3.12.7
Creating virtual environment at: rag
Activate with: source rag/bin/activate
```

### 1.2.2  **啟用虛擬環境**

Windows 系統啟用虛擬環境命令為：

```bash
rag\Scripts\activate
```

Linux / MacOS 系統啟用虛擬環境命令為：

```bash
source rag/bin/activate
```