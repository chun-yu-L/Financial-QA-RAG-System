# Financial-QA-RAG-System

此專案使用 Retrieval Augmented Generation (RAG) 的架構來進行金融問答，結合 Langchain 框架與 Qdrant 向量資料庫進行資料處理、檢索與生成，並採用了多階段檢索提高檢索準確度。

## 專案流程圖
### 資料前處理
<div align="center">
    <img src="assets\aicup_2024-Page-1.jpg", width=70%>
</div>

### RAG 流程
<div align="center">
    <img src="assets\aicup_2024-Page-2.jpg", width=70%>
</div>

## 專案流程
### 1. 資料前處理
- 取出 pdf 內文字並進行清洗、標註

- 將大段文字切割為小塊 (chunking) 並匯入 Qdrant 向量資料庫中

- 此步驟為獨立執行，未包含於 `main.py` 中

### 2. 檔案檢索
- 透過 Qdrant 進行密集向量檢索，取得最能回答 query 的文件與編號

- 對於 insurance 資料進一步透過 cross encoder reranker 進行重排，提高檢索準確度

- 針對 finance 相關查詢，先利用 LLM 進行解析。接著用提取出的關鍵字進行模糊搜尋快速找出關鍵財報文件。若多個文檔符合關鍵字，搭配 dense search 與 reranker 做語義檢索取得最相關文件

### 3. 生成回覆
- 將檢索到的文檔交給 LLM 生成回應

- 若 LLM 判斷文檔不含解答，則回覆"不知道"


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
本專案使用 Breeze-7B-Instruct-v1_0-Q8_0-GGUF 模型。使用前先至 [HuggingFace](https://huggingface.co/Chun-Yu/Breeze-7B-Instruct-v1_0-Q8_0-GGUF/) 下載對應模型並放置於頂層資料夾


## 使用說明
### 資料前處理
此步驟需要單獨執行，用於將原始資料清洗、切塊並匯入 Qdrant 資料庫。

1. 於 `.env` 檔設置 Qdrant 連線參數，設置方法參考 env.template
   - 若架設於本地端可直接設定 qdrant_url='http://localhost:6333/'

2. faq 資料已為文字 json 直接透過對應檔案匯入 Qdrant
    - 先將`競賽資料集`資料夾放入`Preprocess/`路徑下
    ```bash
    # 預設於 Preprocess/ 路徑下執行
    python load_to_Qdrant_qa.py
    ```

3. 利用 `pdf_to_text.py` 萃取 pdf 檔案中的文字資訊，並依照 `{"document_id" : "content"}` 存為 json 格式
    - 註1:執行前須先於 `Preprocess/` 資料夾下建立 `raw_json/` 以及 `chunk_json/` 兩個資料夾作為輸出存放
    - 註2:部分檔案內文缺漏使用人工手打補缺
    ```bash
    # 預設於 Preprocess/ 路徑下執行
    python pdf_to_text.py --source_path ./競賽資料集/reference/ --output_path ./
    ```

4. 取出文字之財報 (finance) 相關文件，利用 `load_to_Qdrant_finance.py` 切塊並匯入向量資料庫

5. 提取出之 insurance 文字透過 `split_insurance_section.py` 根據章節進行切塊，並將輸出利用 `load_to_Qdrant_insurance.py` 標註 metadata 後匯入向量資料庫
    ```bash
    # 預設於 Preprocess/ 路徑下執行
    python split_insurance_section.py
    ```

### 啟動查詢檢索生成
`main.py` 包含查詢的預處理、檢索流程、答案生成。執行指令如下：
```bash
python main.py --questions_path {path_to_question} --parsed_finance_path {path_to_finance_json}
```
- 執行完會產生 `generation_result.json` 檔案為最終檢索與生成結果


## 結果評估
### 檢索準確度
- 運用 Precision@1 進行檢索正確性評估
- `calculateScore.py` 腳本用於計算 retrieval 系統的 Precision@1 分數

- **使用方法:**
    ```bash
    python calculateScore.py --ground_truths_path {path_to_ground_truth} --predictions_path {path_to_prediction} --output_file {output_file_path}
    ```