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


class RequesterConfig(BaseModel):
    host: Host
