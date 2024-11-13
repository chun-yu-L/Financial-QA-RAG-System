# retrieval 相關模組

此資料夾包含兩個主要模組，用於支援 RAG 專案中的查詢前處理與搜尋功能，結合了密集向量搜尋（dense search）與 BM25 查詢，並輔以文字模糊匹配 (fuzzy search) 和交叉編碼器重新排序 (cross encoder reranker)，以達成混合搜尋的最佳效果。

## 檔案介紹
- `finanace_query_preprocess.py`：負責將財務查詢進行結構化處理，解析查詢中的公司名稱、年份、季度及關鍵詞，為後續檢索系統提供標準化的查詢格式。

- `search_core.py`：核心搜尋模組，提供各種搜尋方式，包括密集搜尋、BM25 查詢、模糊搜尋及混合搜尋，並支援結果融合與重新排序。

## finanace_query_preprocess.py
### 功能
- **查詢結構化處理**：此模組定義了 QueryDict 資料格式，包含「公司名稱」、「年份」、「季度」、「目標查詢資訊」和「關鍵字列表」。

- **使用量化 LLM 模型**：透過 llama.cpp 呼叫本地的小型語言模型，並結合示例提示（few-shot learning）進行查詢解析。

### 主要函式
- `QueryDict`：包含查詢的結構化欄位，用來標準化財務查詢中的資訊，包括公司、年份、季度、場景及相關關鍵詞。

- `create_finance_query_parser`：建立並返回財務查詢解析器，將語言模型與多個範例 prompt 結合，輸出結構化查詢。
 
- `query_preprocessor`：將多筆財務問題解析為結構化格式，適用於批次查詢處理。

## search_core.py
### 功能
- 多種搜尋方法：
  - `qdrant_dense_search`：使用 Qdrant 向量庫進行密集向量搜尋。

  - `bm25_jieba_search`：採用 Jieba 分詞的 BM25 檢索，用於稀疏查詢。

  - `fuzzy_search`：基於詞條相似度的模糊搜尋，可調整參數至完全匹配。

- 搜尋結果融合與重新排序：
  - `SearchFusion`：融合密集搜尋與 BM25 搜尋結果，使用 Reciprocal Rank Fusion (RRF) 方法排序。

  - `cross_encoder_rerank`：使用交叉編碼器模型進行結果重新排序，進一步提高準確性。

- 主要類別與函式
  - `FuzzySearchEngine`：模糊搜尋引擎，根據相似度門檻執行模糊查詢並返回相關文件。

  - `qdrant_dense_search`：透過 Qdrant 庫進行向量相似度搜尋。

  - `bm25_jieba_search`：基於 BM25 和 Jieba 分詞的文本查詢。

  - `hybrid_search_rerank`：整合密集搜尋與 BM25 查詢結果並重新排序的混合搜尋方法。

  - `crosss_encoder_rerank`：利用交叉編碼器模型進行最終的結果排序，提升檢索精度。