"""
Defines the base client interface for model interactions.

This module provides the abstract base class that defines the contract for
implementing model clients with both synchronous and asynchronous capabilities.
"""

from abc import ABC, abstractmethod

from models_client_python.common.requester import RequesterConfig


class BaseClient(ABC):
    """
    Abstract base class for model clients.

    This class defines the basic structure for model clients, including context
    manager support and factory methods for creating gRPC-based clients.
    Implementations should support both synchronous and asynchronous operations.
    """

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
