"""
Handles postprocessing of model inference responses from gRPC communication.

This module provides functions to validate and decode gRPC responses into the client's
output format. It handles various data types and ensures proper deserialization of
output tensors from the response.

Some functions in this module are adapted from the official Triton client library.
"""

# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import struct
from typing import List

import numpy as np

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.output import Output
from models_client_python.requester_grpc.gen import grpc_service_pb2


def validate_response(
    response: grpc_service_pb2.ModelInferResponse,
    request: grpc_service_pb2.ModelInferRequest,
    output_keys: List[str],
) -> None:
    """
    Validates that the gRPC response matches the request parameters.

    Args:
        response (ModelInferResponse): The response received from the server.
        request (ModelInferRequest): The original request sent to the server.
        output_keys (List[str]): Names of the output tensors that were requested.

    Raises:
        ValueError: If response ID doesn't match request ID or if there is anoutput count mismatch.
    """
    ## Check that the response ID correspond to the request
    if not response.id == request.id:
        raise ValueError("Response ID {response.id} does not match Request ID {request.id}")

    ## Check that the number of outputs matches the number of output keys
    if not len(response.raw_output_contents) == len(output_keys):
        raise ValueError(f"Expected {len(output_keys)} output keys, got {len(response.raw_output_contents)}")


def process_response(response: grpc_service_pb2.ModelInferResponse) -> List[Output]:
    """
    Processes a gRPC response into a list of Output objects.

    Args:
        response (ModelInferResponse): The response received from the server.

    Returns:
        List[Output]: List of processed output tensors.

    Raises:
        NotImplementedError: If an unsupported datatype is encountered.
    """
    ## Prepare Outputs
    outputs = []
    for i, raw_output in enumerate(response.raw_output_contents):
        output = response.outputs[i]

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
    """
    Decodes raw bytes into a list of 32-bit floating point numbers.

    Args:
        raw_output (bytes): Raw bytes from the gRPC response.

    Returns:
        List[float]: List of decoded floating point values.
    """
    np_array = np.frombuffer(raw_output, dtype=np.float32)

    return np_array.tolist()


def decode_bytes(raw_output: bytes) -> List[str]:
    """
    Decodes raw bytes into a list of strings.

    Each string in the raw output is prefixed with a 4-byte length.
    Hence, need to decode the raw bytes to convert into array elements.

    Args:
        raw_output (bytes): Raw bytes from the gRPC response.

    Returns:
        List[str]: List of decoded strings.
    """
    np_array = _deserialize_bytes_tensor(raw_output)

    return np_array.tolist()


def _deserialize_bytes_tensor(encoded_tensor) -> np.array:
    """
    Deserializes an encoded bytes tensor into a numpy array.

    Args:
        encoded_tensor (bytes): The encoded bytes tensor where each element
            has its length in first 4 bytes followed by the content.

    Returns:
        np.array: 1-D numpy array of type object containing the
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
