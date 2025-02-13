"""
Implements synchronous gRPC communication with machine learning model servers.

This module provides the gRPC implementation of the Requester interface for synchronous
model inference requests. It handles channel setup, message size configuration,
and proper cleanup of gRPC resources.
"""

import struct
from typing import List

import grpc

from models_client_python.common.host import HostScheme
from models_client_python.common.input import Input
from models_client_python.common.output import Output
from models_client_python.common.requester import Requester, RequesterConfig
from models_client_python.requester_grpc.gen import grpc_service_pb2, grpc_service_pb2_grpc
from models_client_python.requester_grpc.postprocess import process_response, validate_response
from models_client_python.requester_grpc.preprocess import build_request

""""
Default values from Nvidia's official cient.
MAX_GRPC_MESSAGE_SIZE Should be kept consistent with the value specified in src/core/constants.h,
which specifies MAX_GRPC_MESSAGE_SIZE as INT32_MAX.

Read the documentation https://github.com/grpc/grpc/blob/master/doc/keepalive.md for more details.
"""
INT32_MAX = 2 ** (struct.Struct("i").size * 8 - 1) - 1
MAX_GRPC_MESSAGE_SIZE = INT32_MAX
CHANNEL_OPT = [
    ("grpc.max_send_message_length", MAX_GRPC_MESSAGE_SIZE),
    ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_SIZE),
    ("grpc.keepalive_time_ms", INT32_MAX),
    ("grpc.keepalive_timeout_ms", 2000),
    ("grpc.keepalive_permit_without_calls", False),
    ("grpc.http2.max_pings_without_data", 2),
]


class RequesterGrpc(Requester):
    """
    Synchronous gRPC client for making model inference requests.

    This class implements the Requester interface using gRPC for synchronous communication
    with model servers. It handles connection setup, message size limits, and cleanup.

    Args:
        config (RequesterConfig): Configuration containing host details for the gRPC connection.

    Raises:
        NotImplementedError: If HTTPS/TLS connections are attempted (not yet implemented).
    """

    def __init__(self, config: RequesterConfig):
        # Always define _channel, even if we don’t create a real channel
        self._channel = None

        if config.host.scheme == HostScheme.http:
            self._channel = grpc.insecure_channel(target=config.host.target(), options=CHANNEL_OPT)

        elif config.host.scheme == HostScheme.https:
            raise NotImplementedError("gRPC secure channel hasn't been implemented yet")

        self._client_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel=self._channel)

    def infer(self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> List[Output]:
        request = build_request(
            model_name=model_name,
            model_version=model_version,
            inputs=inputs,
            output_keys=output_keys,
        )
        response: grpc_service_pb2.ModelInferResponse = self._client_stub.ModelInfer(request=request)
        validate_response(response=response, request=request, output_keys=output_keys)

        return process_response(response=response)

    def stream(self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> str:
        raise NotImplementedError

    def close(self):
        """
        Safely close the channel if it exists.
        """
        if self._channel is not None:
            self._channel.close()
            self._channel = None  # optional, helps avoid double-close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        # Python calls __del__ even if __init__ failed, so let's be safe.
        self.close()
