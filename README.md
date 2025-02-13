# Clinia Python Models Client

## Features

- Native Python implementation for communicating with Clinia's models on NVIDIA Triton Inference Server
- gRPC-based communication for efficient and reliable model inference
- Support for both synchronous and asynchronous requests
- Type-safe interfaces for all models and requests
- Supports batch processing for optimal performance
- Minimal dependencies and lightweight implementation
- Designed for internal Clinia services with VPC access

## Getting Started

The prefered way to install the package is using uv:

```bash
uv add git+https://github.com/clinia/models-client-python
```

It can also be installed via pip:

```bash
pip install git+https://github.com/clinia/models-client-python.git
```

## Playground Examples

### Embedder Model

#### Batch inference

```python
from os import getenv
from uuid import uuid4
from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig
from models_client_python.embedder import Embedder, EmbedRequest

texts = ["Where is Clinia based?", "Clinia is based in Montreal"]

requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))

with Embedder.from_grpc(config=requester_config) as embedder:
    request = EmbedRequest(id=str(uuid4()), texts=texts)
    embeddings = embedder.embed(
        model_name=getenv("CLINIA_MODEL_NAME"),
        model_version=getenv("CLINIA_MODEL_VERSION"),
        req=request
    )
```

#### Async single element inference

```python
import asyncio
from os import getenv
from uuid import uuid4
from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig
from models_client_python.embedder import Embedder, EmbedRequest

async def get_embeddings(texts: List[str]):
    requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))
    requests = [EmbedRequest(id=str(uuid4()), texts=[text]) for text in texts]

    async with Embedder.from_grpc_async(config=requester_config) as embedder:
        tasks = [
            embedder.embed_async(
                model_name=getenv("CLINIA_MODEL_NAME"),
                model_version=getenv("CLINIA_MODEL_VERSION"),
                req=req
            ) for req in requests
        ]
        return await asyncio.gather(*tasks)

texts = ["Where is Clinia based?"]
embeddings = asyncio.run(get_embeddings(texts=texts * 32))  # Example with batch size of 32
```

### Ranker Model

#### Single element inference

```python
from os import getenv
from uuid import uuid4
from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig
from models_client_python.ranker import Ranker, RankRequest

requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))

with Ranker.from_grpc(config=requester_config) as ranker:
    request = RankRequest(
        id=str(uuid4()),
        query="Where is Clinia based?",
        texts=["Clinia is based in Montreal"]
    )
    scores = ranker.rank(
        model_name=getenv("CLINIA_MODEL_NAME"),
        model_version=getenv("CLINIA_MODEL_VERSION"),
        req=request
    )
```

### Chunker Model

#### Batch inference

```python
from os import getenv
from uuid import uuid4
from models_client_python.chunker import Chunker, ChunkRequest
from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig

texts = [
    "This is a short text",
    "This is a longer text that. contains. mutliple sentences. but. should still be a single chunk"
]

requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))

with Chunker.from_grpc(config=requester_config) as chunker:
    request = ChunkRequest(id=str(uuid4()), texts=texts)
    chunks = chunker.chunk(
        model_name=getenv("CLINIA_MODEL_NAME"),
        model_version=getenv("CLINIA_MODEL_VERSION"),
        req=request
    )
```

## Note

This repository is automatically generated from a private repository within Clinia that contains additional resources including tests, mock servers, and development tools. The version numbers of this package correspond to the same versions in the respective Go and TypeScript public repositories, ensuring consistency across all implementations.

## License

The Clinia Python Models Client is an open-sourced software licensed under the [MIT license](LICENSE).
