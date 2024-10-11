# Financial-QA-RAG-System


## ChromaDB Deployment Process

This section outlines the steps required to deploy and authenticate ChromaDB using Docker, as well as how to connect to the ChromaDB server via Python.

### 1. Download the ChromaDB Image
Start by pulling the latest ChromaDB image from Docker Hub:
```bash
docker pull chromadb/chroma:latest
```

### 2. Create Authentication Credentials
Create a password for authentication and save it in the container:
```bash
sudo docker run --rm --entrypoint htpasswd httpd:2 -Bbn ADMIN YOURPASSWORD > server.htpasswd
sudo docker cp server.htpasswd chromaDB:/server.htpasswd
```
- `ADMIN`: The username for authentication.
- `YOURPASSWORD`: The password for the chosen username.

### 3. Run the ChromaDB Server
Start the ChromaDB server with authentication by running the following command:
```bash
docker run -d -v ./server.htpasswd:/chroma/server.htpasswd \
    -e CHROMA_SERVER_AUTHN_CREDENTIALS_FILE="server.htpasswd" \
    -e CHROMA_SERVER_AUTHN_PROVIDER="chromadb.auth.basic_authn.BasicAuthenticationServerProvider" \
    -p 7878:8000 \
    chromadb/chroma:latest
```
This command mounts the `server.htpasswd` file, enabling basic authentication on the ChromaDB server.

### 4. Connect to the ChromaDB Server from a Python Client
To connect to the ChromaDB server using a Python client, you can use the following setup:

```python
from chromadb import HttpClient
from chromadb.config import Settings

chroma_client = HttpClient(
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
        chroma_client_auth_credentials=f"{chromadb_user}:{chromadb_pwd}",
    ),
    host=chromadb_host,
    port=7878,
)
```
- `chroma_client_auth_provider`: Specifies the authentication provider.
- `chroma_client_auth_credentials`: Combines the username and password for authentication.
- `host`: The address of the ChromaDB server.
- `port`: The port the ChromaDB server is running on (`7878`).

Ensure that `chroma_client_auth_credentials` contains the correct credentials in the format `username:password`.
