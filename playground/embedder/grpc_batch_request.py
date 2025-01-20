from uuid import uuid4

from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig
from models_client_python.embedder import Embedder, EmbedRequest

if __name__ == "__main__":
    MODEL_NAME = "embedder_medical_journals_qa"
    MODEL_VERSION = "120240905185426"
    TEXTS = ["hello, how are you?", "Clinia is based in Montreal"]

    ## Define gRPC requester config
    requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))

    ## Initialize Embedder
    embedder = Embedder.from_grpc(config=requester_config)

    ## Send request
    request = EmbedRequest(id=str(uuid4()), texts=TEXTS)
    response = embedder.embed(model_name=MODEL_NAME, model_version=MODEL_VERSION, req=request)

    print(response)
