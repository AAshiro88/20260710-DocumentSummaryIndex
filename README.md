# Document Summary Index — IT 支援助手

OKF 格式的 Markdown 知識文件經過向量化索引後，提供多輪問答的 Gradio 聊天介面。

## 系統架構

```
Markdown 文件 (OKF/it-support)
    │
    ▼
train_dsi.py  →  向量化後存入 Qdrant (本機磁碟)
    │
    ▼
dsi_chat.py   →  Gradio 聊天介面
                    ├── Stage 1：向量檢索 + Rerank 找出相關文件
                    └── Stage 2：LLM 根據文件內容回答問題
```

## 檔案說明

| 檔案 | 用途 |
|------|------|
| `train_dsi.py` | 掃描 OKF/it-support 下的 Markdown 文件，建立向量索引存入 Qdrant |
| `vector_store.py` | Qdrant 客戶端封裝（建立 collection、插入向量、查詢） |
| `dsi_chat.py` | Gradio ChatInterface 主程式，含狀態機、意圖分類、LLM 串接 |
| `requirements.txt` | Python 相依套件 |

## 環境需求

- Python 3.13.9

### 套件版本

gradio==6.20.0
qdrant-client==1.18.0
sentence-transformers==5.6.0
PyYAML==6.0.3
python-dotenv==1.1.0
google-genai==2.10.0
requests==2.32.5

### Ollama 模型

- `nomic-embed-text`（向量檢索與意圖分類用）
- `DedeProGames/open-reasoner:1.7b`（Stage 2 回答用，可於 dsi_chat.py 中更換）

## 設定

手動建立 `.env`（若使用 Gemini 作為 Stage 2 模型時才需設定 `GOOGLE_API_KEY`）：

```
GOOGLE_API_KEY=your_gemini_api_key
```

若使用 Ollama 作為 Stage 2 模型（預設），則不需 API Key。

## 使用方式

### 1. 建立索引

```bash
python train_dsi.py
```

此指令會掃描 `OKF/it-support/` 下的所有 Markdown 檔案，解析 YAML frontmatter，使用 `nomic-embed-text` 產生向量後存入本機 Qdrant。

### 2. 啟動聊天

```bash
python dsi_chat.py
```

開啟 Gradio 介面後，輸入問題即可開始多輪檢索問答流程。

## 知識文件格式

知識文件採用 OKF 格式，每個 Markdown 檔案包含 YAML frontmatter：

```yaml
---
title: 文件標題
description: 文件描述
tags: [標籤1, 標籤2]
type: concept
---
```

## Qdrant 資料

向量資料庫儲存於 `db/qdrant/` 目錄下，collection 名稱為 `it_support_dsi`。刪除或重建索引時可直接刪除此目錄後重新執行 `train_dsi.py`。

## 授權

內部使用。
