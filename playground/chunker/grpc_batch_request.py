from uuid import uuid4

from models_client_python.chunker import Chunker, ChunkRequest
from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig

if __name__ == "__main__":
    MODEL_NAME = "chunker"
    MODEL_VERSION = "120240308103333"
    TEXTS = ["hello, how are you?", "Clinia is based in Montreal"]

    ## Define gRPC requester config
    requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))

    ## Initialize Chunker
    chunker = Chunker.from_grpc(config=requester_config)

    ## Send request
    request = ChunkRequest(id=str(uuid4()), texts=TEXTS)
    response = chunker.chunk(model_name=MODEL_NAME, model_version=MODEL_VERSION, req=request)

    print(response)
