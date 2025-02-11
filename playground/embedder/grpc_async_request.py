import asyncio
from typing import List
from uuid import uuid4

from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig
from models_client_python.embedder import Embedder, EmbedRequest, EmbedResponse


async def main(requester_config: RequesterConfig, queries: List[str]) -> List[EmbedResponse]:
    ## Prepare requests
    requests = [EmbedRequest(id=str(uuid4()), texts=[query]) for query in queries]

    ## Initialize Embedder
    async with Embedder.from_grpc_async(config=requester_config) as embedder:
        ## Create asyncio tasks
        tasks = [embedder.embed_async(model_name=MODEL_NAME, model_version=MODEL_VERSION, req=req) for req in requests]

        ## Execute them concurrently
        responses = await asyncio.gather(*tasks)

        return responses


if __name__ == "__main__":
    MODEL_NAME = "embedder_medical_journals_qa"
    MODEL_VERSION = "120240905185426"
    TEXTS = ["Clinia is based in Montreal"]

    ## Define gRPC requester config
    requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))

    # Example for a batch size of 32
    embeddings = asyncio.run(main(requester_config=requester_config, queries=TEXTS * 32))

    print(embeddings)
