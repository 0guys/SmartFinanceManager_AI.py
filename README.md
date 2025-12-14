# 🧠 Smart Finance App — Python AI 雲端財務管家

一套以 **Python + Tkinter** 打造的桌面記帳應用，整合 **SQLite 本地資料庫**、**臺灣銀行即時匯率爬蟲**，並支援 **Google Gemini AI（雲端）** 與 **本地 AI 分析（無 API Key 也可用）**。

---

## ✨ 功能特色

### 📝 記帳管理

* 支援 **收入 / 支出** 記錄
* 消費分類（飲食、交通、娛樂、購物…）
* SQLite 永久儲存
* TreeView 即時列表顯示
* 一鍵刪除、CSV 匯出

### 💱 多幣別 + 即時匯率

* 支援 **TWD / USD / JPY**
* 自動爬取「臺灣銀行」現金賣出匯率
* 外幣輸入自動換算為台幣並備註原幣別

### 📊 視覺化分析

* 支出分類圓餅圖（Canvas 繪製）
* 即時顯示總資產（收入 − 支出）

### 🤖 AI 財務分析（雙模式）

#### ☁️ 雲端 AI（Google Gemini）

* 使用 Gemini Flash 模型
* 依近期消費習慣生成 **繁體中文理財建議**
* 自動處理 API 429（配額不足）重試

#### 🖥️ 本地 AI（免 API Key）

* 線性 / 多項式回歸（Scikit‑learn 可選）
* 預測「隔日支出金額」
* 消費結構風險提示（娛樂、飲食佔比）

---

## 🧱 技術架構

| 類別     | 使用技術                     |
| ------ | ------------------------ |
| GUI    | Tkinter / ttk            |
| 資料庫    | SQLite3                  |
| 爬蟲     | requests + BeautifulSoup |
| AI（雲端） | Google Gemini API        |
| AI（本地） | Python 原生 / Scikit‑learn |
| 視覺化    | Tkinter Canvas           |

---

## 📦 安裝與執行

### 1️⃣ 環境需求

* Python **3.9+**
* 作業系統：Windows / macOS / Linux

### 2️⃣ 安裝套件

```bash
pip install requests beautifulsoup4

# 若要啟用進階本地 AI（非必要）
pip install scikit-learn numpy
```

### 3️⃣ 執行程式

```bash
python main.py
```

（首次執行會自動建立 `finance.db`）

---

## 🔑 Google Gemini API 設定（選用）

1. 前往 Google AI Studio 建立 API Key
2. 啟動程式後點擊：
   **「設定 Google Gemini API Key」**
3. 貼上 Key → 即可使用雲端 AI 分析

⚠️ 若未設定 Key，系統將 **自動切換為本地 AI 模式**

---

## 🚨 已知限制與注意事項

* Gemini **免費方案有 RPM / RPD / TPM 配額限制**
* 若出現 `HTTP 429 RESOURCE_EXHAUSTED`：

  * 等待配額重置（或稍後重試）
  * 減少分析頻率
* 匯率來源為公開網站，若臺銀頁面改版可能需調整解析

---

## 📁 專案結構（簡要）

```
.
├─ main.py              # 主程式
├─ finance.db           # SQLite 資料庫（自動產生）
├─ README.md
```

---

## 🛠️ 可擴充方向

* 月/年報表統計
* 收入 vs 支出趨勢圖（Matplotlib）
* 類別預算上限警示
* 匯率快取與離線模式
* 多語系 UI

---

## 📜 License

本專案為 **學習 / 個人使用** 專案，未附商業授權。

---

## 🙌 作者

由 Python + AI 技術愛好者打造

> 如果你在學習 Python、Tkinter、AI API 整合，這個專案非常適合作為完整實戰範例 🚀
