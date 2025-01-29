from typing import List

import grpc
import numpy as np

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.host import HostScheme
from models_client_python.common.input import Input
from models_client_python.common.output import Output
from models_client_python.common.requester import Requester, RequesterConfig
from models_client_python.requester_grpc.gen import grpc_service_pb2, grpc_service_pb2_grpc
from models_client_python.requester_grpc.postprocess import decode_bytes, decode_fp32
from models_client_python.requester_grpc.preprocess import encode_bytes


class RequesterGrpc(Requester):
    def __init__(self, config: RequesterConfig):
        # Always define _channel, even if we don’t create a real channel
        self._channel = None

        if config.host.scheme == HostScheme.http:
            self._channel = grpc.insecure_channel(config.host.host())

        elif config.host.scheme == HostScheme.https:
            raise NotImplementedError("gRPC secure channel hasn't been implemented yet")

        self._client_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel=self._channel)

    def infer(self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> List[Output]:
        ## Prepare Inputs
        raw_inputs = []
        grpc_inputs = []
        for input in inputs:
            # TODO: Support other datatypes.
            if input.datatype == Datatype.bytes:
                raw_inputs.append(encode_bytes(input=input))

            else:
                raise NotImplementedError(f"Unsupported datatype: {input.datatype}")

            grpc_inputs.append(
                grpc_service_pb2.ModelInferRequest.InferInputTensor(
                    name=input.name,
                    datatype=input.datatype,
                    shape=input.shape,
                )
            )

        ## Prepare Outputs
        grpc_outputs = [
            grpc_service_pb2.ModelInferRequest.InferRequestedOutputTensor(name=output_key) for output_key in output_keys
        ]

        ## Format Request
        # NOTE: There will be multiple IDs inside a request when batching (one per input), we just take the first one to assert later on that the model answered to the right batch.
        request = grpc_service_pb2.ModelInferRequest(
            id=inputs[0].id,
            model_name=model_name,
            model_version=model_version,
            inputs=grpc_inputs,
            outputs=grpc_outputs,
            raw_input_contents=raw_inputs,
        )

        ## Send Request
        response: grpc_service_pb2.ModelInferResponse = self._client_stub.ModelInfer(request=request)

        ## Check that the response ID correspond to the request
        if not response.id == request.id:
            raise ValueError("Response ID {response.id} does not match Request ID {request.id}")

        ## Check that the number of outputs matches the number of output keys
        if not len(response.raw_output_contents) == len(output_keys):
            raise ValueError(f"Expected {len(output_keys)} output keys, got {len(response.raw_output_contents)}")

        ## Prepare Outputs
        outputs = []
        for i, raw_output in enumerate(response.raw_output_contents):
            output = response.outputs[i]

            ## TODO: this was in the original code, assert for these conditions in tests
            # if index < len(response.raw_output_contents):
            #     np_array = decode_bytes | decode_fp32
            # elif len(output.contents.bytes_contents) != 0:
            #     np_array = np.array(output.contents.bytes_contents, copy=False)
            # else:
            #     np_array = np.empty(0)

            # TODO: Support other datatypes.
            if output.datatype == Datatype.fp32:
                content = decode_fp32(raw_output=raw_output)
                outputs.append(
                    Output(
                        name=output.name,
                        shape=output.shape,
                        datatype=output.datatype,
                        content=Content(fp32_contents=content),
                    )
                )

            elif output.datatype == Datatype.bytes:
                content = decode_bytes(raw_output=raw_output)
                outputs.append(
                    Output(
                        name=output.name,
                        shape=output.shape,
                        datatype=output.datatype,
                        content=Content(string_contents=content),
                    )
                )

            else:
                raise NotImplementedError(f"Unsupported datatype: {output.datatype}")

        return outputs

    def stream(self, model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]) -> str:
        raise NotImplementedError

    def close(self):
        """
        Safely close the channel if it exists.
        """
        if self._channel is not None:
            self._channel.close()
            self._channel = None  # optional, helps avoid double-close

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        # Python calls __del__ even if __init__ failed, so let's be safe.
        self.close()
