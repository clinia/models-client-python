from abc import ABC, abstractmethod
from typing import List

from common.input import Input
from common.output import Output
from pydantic import BaseModel


class Requester(ABC):
    @abstractmethod
    def infer(self, model_name: str, model_version: str, inputs: List[Input], outputs: List[str]) -> List[Output]:
        pass

    @abstractmethod
    def stream(self, model_name: str, model_version: str, inputs: List[Input], outputs: List[str]) -> str:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class RequesterConfig(BaseModel):
    host: str
