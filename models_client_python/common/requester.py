from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from models_client_python.common.host import Host
from models_client_python.common.input import Input
from models_client_python.common.output import Output


class Requester(ABC):
    @abstractmethod
    def infer(self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> List[Output]:
        pass

    @abstractmethod
    def stream(self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> str:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, type, value, traceback):
        pass

    @abstractmethod
    def __del__(self):
        pass


class RequesterAsync(ABC):
    @abstractmethod
    async def infer(
        self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]
    ) -> List[Output]:
        pass

    @abstractmethod
    async def stream(self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> str:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def __aenter__(self):
        pass

    @abstractmethod
    async def __aexit__(self, type, value, traceback):
        pass


class RequesterConfig(BaseModel):
    host: Host
