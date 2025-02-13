from os import getenv
from uuid import uuid4

from models_client_python.chunker import Chunker, ChunkRequest
from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig

if __name__ == "__main__":
    MODEL_NAME = getenv("CLINIA_MODEL_NAME")
    MODEL_VERSION = getenv("CLINIA_MODEL_VERSION")
    TEXTS = [
        "This is a short text",
        "This is a longer text that. contains. mutliple sentences. but. should still be a single chunk",
    ]

    ## Define gRPC requester config
    requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))

    ## Initialize Chunker
    chunker = Chunker.from_grpc(config=requester_config)

    ## Send request
    request = ChunkRequest(id=str(uuid4()), texts=TEXTS)
    response = chunker.chunk(model_name=MODEL_NAME, model_version=MODEL_VERSION, req=request)

    print(response)
