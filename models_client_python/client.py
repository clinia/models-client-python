from models_client_python.common.client import BaseClient
from models_client_python.common.requester import Requester, RequesterConfig
from models_client_python.requester_grpc.requester import RequesterGrpc
from models_client_python.requester_grpc.requester_async import RequesterAsyncGrpc


class Client(BaseClient):
    def __init__(self, requester: Requester):
        self.requester = requester

    def __enter__(self):
        return self

    async def __aenter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.requester.close()

    async def __aexit__(self, type, value, traceback):
        await self.requester.close()

    @classmethod
    def from_grpc(cls, config: RequesterConfig):
        return cls(requester=RequesterGrpc(config=config))

    @classmethod
    def from_grpc_async(cls, config: RequesterConfig):
        return cls(requester=RequesterAsyncGrpc(config=config))
