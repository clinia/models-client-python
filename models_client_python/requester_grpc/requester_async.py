from typing import List

import grpc.aio

from models_client_python.common.host import HostScheme
from models_client_python.common.input import Input
from models_client_python.common.output import Output
from models_client_python.common.requester import RequesterAsync, RequesterConfig
from models_client_python.requester_grpc.gen import grpc_service_pb2, grpc_service_pb2_grpc
from models_client_python.requester_grpc.postprocess import process_response, validate_response
from models_client_python.requester_grpc.preprocess import build_request
from models_client_python.requester_grpc.requester import CHANNEL_OPT


class RequesterAsyncGrpc(RequesterAsync):
    def __init__(self, config: RequesterConfig):
        # Always define _channel, even if we don’t create a real channel
        self._channel = None

        if config.host.scheme == HostScheme.http:
            self._channel = grpc.aio.insecure_channel(target=config.host.target(), options=CHANNEL_OPT)

        elif config.host.scheme == HostScheme.https:
            raise NotImplementedError("gRPC secure channel hasn't been implemented yet")

        self._client_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel=self._channel)

    async def infer(
        self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]
    ) -> List[Output]:
        request = build_request(
            model_name=model_name,
            model_version=model_version,
            inputs=inputs,
            output_keys=output_keys,
        )

        response: grpc_service_pb2.ModelInferResponse = await self._client_stub.ModelInfer(request=request)
        validate_response(response=response, request=request, output_keys=output_keys)

        return process_response(response=response)

    async def stream(self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> str:
        raise NotImplementedError

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
