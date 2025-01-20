from uuid import uuid4

from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig
from models_client_python.ranker import Ranker, RankRequest

if __name__ == "__main__":
    MODEL_NAME = "ranker_medical_journals_qa"
    MODEL_VERSION = "120240905185925"
    QUERY = "hello, how are you?"
    TEXTS = ["Clinia is based in Montreal"]

    ## Define gRPC requester config
    requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))

    ## Initialize Ranker
    ranker = Ranker.from_grpc(config=requester_config)

    ## Send request
    request = RankRequest(id=str(uuid4()), query=QUERY, texts=TEXTS)
    response = ranker.rank(model_name=MODEL_NAME, model_version=MODEL_VERSION, req=request)

    print(response)
