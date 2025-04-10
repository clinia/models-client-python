"""
Implements asynchronous gRPC communication with machine learning model servers.

This module provides the gRPC implementation of the RequesterAsync interface for non-blocking
model inference requests. It handles async channel setup, message size configuration,
and proper cleanup of gRPC resources.
"""

from typing import List

import grpc.aio

from models_client_python.common.host import HostScheme
from models_client_python.common.input import Input
from models_client_python.common.output import Output
from models_client_python.common.requester import RequesterAsync, RequesterConfig
from models_client_python.requester_grpc.gen import grpc_service_pb2, grpc_service_pb2_grpc
from models_client_python.requester_grpc.postprocess import process_response, validate_response
from models_client_python.requester_grpc.preprocess import build_request, format_model_name_and_version
from models_client_python.requester_grpc.requester import CHANNEL_OPT


class RequesterAsyncGrpc(RequesterAsync):
    """
    Asynchronous gRPC client for making model inference requests.

    This class implements the RequesterAsync interface using gRPC for non-blocking communication
    with model servers. It provides async context manager support and resource cleanup.

    Args:
        config (RequesterConfig): Configuration containing host details for the gRPC connection.

    Raises:
        NotImplementedError: If HTTPS/TLS connections are attempted (not yet implemented).
    """

    def __init__(self, config: RequesterConfig):
        # Always define _channel, even if we don’t create a real channel
        self._channel = None

        if config.host.scheme == HostScheme.http:
            self._channel = grpc.aio.insecure_channel(target=config.host.target(), options=CHANNEL_OPT)

        elif config.host.scheme == HostScheme.https:
            raise NotImplementedError("gRPC secure channel hasn't been implemented yet")

        self._client_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel=self._channel)

    async def infer(
        self, id: str, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]
    ) -> List[Output]:
        # Format model name and version
        model_name_formatted, model_version_formatted = format_model_name_and_version(model_name, model_version)

        request = build_request(
            id=id,
            model_name=model_name_formatted,
            model_version=model_version_formatted,
            inputs=inputs,
            output_keys=output_keys,
        )

        response: grpc_service_pb2.ModelInferResponse = await self._client_stub.ModelInfer(request=request)
        validate_response(response=response, request=request, output_keys=output_keys)

        return process_response(response=response)

    async def stream(
        self, id: str, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]
    ) -> str:
        raise NotImplementedError

    async def health(self):
        request = grpc_service_pb2.ServerReadyRequest()
        response: grpc_service_pb2.ServerReadyResponse = await self._client_stub.ServerReady(request=request)

        if not response.ready:
            raise RuntimeError("Server is not ready")

    async def ready(self, model_name: str, model_version: str) -> None:
        # Format model name and version
        model_name_formatted, model_version_formatted = format_model_name_and_version(model_name, model_version)

        request = grpc_service_pb2.ModelReadyRequest(name=model_name_formatted, version=model_version_formatted)
        response: grpc_service_pb2.ModelReadyResponse = await self._client_stub.ModelReady(request=request)

        if not response.ready:
            raise RuntimeError(f"Model {model_name} with version {model_version} is not ready")

    async def close(self):
        """
        Safely close the channel if it exists.
        """
        if self._channel is not None:
            await self._channel.close()
            self._channel = None  # optional, helps avoid double-close

    async def __aenter__(self):
        return self

    async def __aexit__(self, type, value, traceback):
        await self.close()
