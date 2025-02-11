from abc import ABC, abstractmethod

from models_client_python.common.requester import RequesterConfig


class BaseClient(ABC):
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    async def __aenter__(self):
        pass

    @abstractmethod
    def __exit__(self, type, value, traceback):
        pass

    @abstractmethod
    async def __aexit__(self, type, value, traceback):
        pass

    @classmethod
    @abstractmethod
    def from_grpc(cls, config: RequesterConfig):
        pass

    @classmethod
    @abstractmethod
    def from_grpc_async(cls, config: RequesterConfig):
        pass
