
## 目錄結構
- `split_insurance_section.py`：用於分割保險相關的資料段落，並產出供 `load_to_Qdrant_insurance_chunk.py` 使用的輸入資料。

- `load_to_Qdrant_finance_chunk.py`：負責將財務相關的資料段落載入至 Qdrant 資料庫中。

- `load_to_Qdrant_insurance_chunk.py`：將保險相關的資料段落載入至 Qdrant 資料庫中，需依賴 `split_insurance_section.py` 的輸出結果。

- `load_to_Qdrant_qa.py`：負責載入 FAQ 資料至 Qdrant 資料庫中。

## 使用說明
### 1. 分割保險資料段落
執行 `split_insurance_section.py` 來處理並分割保險資料段落，結果將會產生輸入資料，供後續的 `load_to_Qdrant_insurance_chunk.py` 使用。
```bash
python split_insurance_section.py
```
### 2. 載入資料至 Qdrant
各個 `load_to_Qdrant` 模組可以單獨執行，將對應的資料載入至 Qdrant 資料庫中。

- 載入財務資料段落：
    ```bash
    python load_to_Qdrant_finance_chunk.py
    ```

- 載入保險資料段落（須先執行 split_insurance_section.py）：
    ```bash
    python load_to_Qdrant_insurance_chunk.py
    ```
    
- 載入 FAQ 資料：
    ```bash
    python load_to_Qdrant_qa.py
    ```

## 注意事項
- `split_insurance_section.py` 的輸出為 `load_to_Qdrant_insurance_chunk.py` 的輸入，需按順序執行。
- 其餘 `load_to_Qdrant` 模組皆可獨立執行，不受順序影響。


## Qdrant deployment
1. Install docker and pull the image from qdrant official
```bash
docker pull qdrant/qdrant:latest
# or the version used in this development (2024-10-18)
# docker pull qdrant/qdrant:v1.12.1 
```
2. Run the docker container for qdrant server
```bash
docker run --restart=always \
    --name qdrant -d \
	-p 6333:6333 \ # 6333 - For the HTTP API
    -p 6334:6334 \ # 6334 - For the gRPC API
	-v $(pwd)/qdrant_storage:/qdrant:z qdrant/qdrant # use volume to save data to disk
```
3. Test the connection
    - through python SDK
    ```python
    # pip install qdrant-client
    from qdrant_client import QdrantClient

    qdrant = QdrantClient("http://localhost:6333") # Connect to existing Qdrant instance
    ```
     - through CLI
    ```bash
    curl http://localhost:6333 # response StatusCode should be 200
    ```