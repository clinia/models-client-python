"""
Defines the base interfaces for synchronous and asynchronous model requesters.

This module provides abstract base classes that define the contract for implementing
model inference clients, both synchronous and asynchronous variants.
"""

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from models_client_python.common.host import Host
from models_client_python.common.input import Input
from models_client_python.common.output import Output


class Requester(ABC):
    """
    Abstract base class for synchronous model inference requests.

    Implementations of this interface handle the communication with model servers,
    managing connections, and performing inference requests in a blocking manner.
    """

    @abstractmethod
    def infer(
        self, id: str, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]
    ) -> List[Output]:
        pass

    @abstractmethod
    def stream(self, id: str, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> str:
        pass

    @abstractmethod
    def ready(self, model_name: str, model_version: str) -> None:
        pass

    @abstractmethod
    def health(self) -> None:
        """Only the requester implements this method, the clients only implement the ready method."""
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
    """
    Abstract base class for asynchronous model inference requests.

    Implementations of this interface handle the communication with model servers,
    managing connections, and performing inference requests in a non-blocking manner
    using async/await patterns.
    """

    @abstractmethod
    async def infer(
        self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]
    ) -> List[Output]:
        pass

    @abstractmethod
    async def stream(self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> str:
        pass

    @abstractmethod
    async def ready(self, model_name: str, model_version: str) -> None:
        pass

    @abstractmethod
    async def health(self) -> None:
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
    """
    Configuration model for Requester instances.

    Attributes:
        host (Host): Host configuration including URL, port, and scheme.
    """

    host: Host
