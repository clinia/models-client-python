import struct
from typing import List

import numpy as np

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.output import Output
from models_client_python.requester_grpc.gen import grpc_service_pb2


def validate_response(
    response: grpc_service_pb2.ModelInferResponse, request: grpc_service_pb2.ModelInferRequest, output_keys: List[str]
) -> None:
    ## Check that the response ID correspond to the request
    if not response.id == request.id:
        raise ValueError("Response ID {response.id} does not match Request ID {request.id}")

    ## Check that the number of outputs matches the number of output keys
    if not len(response.raw_output_contents) == len(output_keys):
        raise ValueError(f"Expected {len(output_keys)} output keys, got {len(response.raw_output_contents)}")


def process_response(response: grpc_service_pb2.ModelInferResponse) -> List[Output]:
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


def decode_fp32(raw_output: bytes) -> List[float]:
    np_array = np.frombuffer(raw_output, dtype=np.float32)

    return np_array.tolist()


def decode_bytes(raw_output: bytes) -> List[str]:
    """
    String results contain a 4-byte string length followed by the actual string characters.
    Hence, need to decode the raw bytes to convert into array elements.
    """
    np_array = _deserialize_bytes_tensor(raw_output)

    return np_array.tolist()


def _deserialize_bytes_tensor(encoded_tensor) -> np.array:
    """
    Deserializes an encoded bytes tensor into a
    numpy array of dtype of python objects

    Parameters
    ----------
    encoded_tensor : bytes
        The encoded bytes tensor where each element
        has its length in first 4 bytes followed by
        the content
    Returns
    -------
    string_tensor : np.array
        The 1-D numpy array of type object containing the
        deserialized bytes in row-major form.

    """
    strs = []
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        length = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4

        sb = struct.unpack_from("<{}s".format(length), val_buf, offset)[0]
        offset += length
        strs.append(sb)
    return np.array(strs, dtype=np.object_)
