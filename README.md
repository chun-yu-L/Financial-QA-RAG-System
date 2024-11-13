# Financial-QA-RAG-System

此專案使用 Retrieval Augmented Generation (RAG) 的架構來進行金融問答，結合 Langchain 框架與 Qdrant 向量資料庫進行資料處理、檢索與生成，並採用了多階段檢索提高檢索準確度。

## 專案流程
### 1. 資料前處理 (Preprocess)
- 取出 pdf 內文字並進行清洗、標註

- 將大段文字切割為小塊 (chunking) 並匯入 Qdrant 向量資料庫中

- 此步驟為獨立執行，未包含於 `main.py` 中

### 2. 檔案檢索 (Model)
- 透過 Qdrant 進行密集向量檢索，取得最能回答 query 的文件與編號

- 對於 insurance 資料進一步透過 cross encoder reranker 進行重排，提高檢索準確度

- 針對 finance 相關查詢，先利用 llama-cpp-python 跑地端小型 LLM 進行解析。接著用提取出的關鍵字進行模糊搜尋快速找出關鍵財報文件。若多個文檔符合關鍵字，搭配 dense search 與 reranker 做語義檢索取得最相關文件

## 環境設置
1. 環境需求
    - Python 3.9 以上
    - 安裝所需 python 套件  
        ```bash
        pip install -r requirements.txt
        ```
        若 llama-cpp-python 安裝失敗，參考[官方](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file)安裝環境設置

    - Qdrant 向量資料庫
        - 安裝參考[官方指南](https://qdrant.tech/documentation/guides/installation/)或 Preprocess 資料夾 README
        - `.env` 設置連線參數

2. 模型下載  
本專案使用 Qwen2.5-3B-Instruct-GGUF 模型進行財報相關 query 前處理。使用前先至 [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) 下載對應模型並放置於頂層資料夾

## 使用說明
### 資料前處理
此步驟需要單獨執行，用於將原始資料清洗、切塊並匯入 Qdrant 資料庫。

1. 於 `.env` 檔設置 Qdrant 連線參數，設置方法參考 env.template

2. faq 資料已為文字 json 直接透過對應檔案匯入 Qdrant
    ```bash
    python load_to_Qdrant_qa.py
    ```

3. 將 pdf 檔利用利用##########

4. 取出文字之財報 (finance) 相關文件，利用 `load_to_Qdrant_finance.py` 切塊並匯入向量資料庫

5. 提取出之 insurance 透過 `split_insurance_section.py` 根據章節進行切塊，並將輸出利用 `load_to_Qdrant_insurance.py` 標註 metadata 後匯入向量資料庫

### 啟動查詢檢索
`main.py` 包含查詢的預處理與檢索流程。執行指令如下：
```bash
python main.py --questions_path {path_to_question} --parsed_finance_path {path_to_finance_json}
```
- 執行完會產生 `retrieval_result.json` 檔案為檢索結果