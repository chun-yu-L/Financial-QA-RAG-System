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