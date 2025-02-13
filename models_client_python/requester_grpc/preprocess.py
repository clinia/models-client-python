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

from models_client_python.common.datatype import Datatype
from models_client_python.common.input import Input
from models_client_python.requester_grpc.gen import grpc_service_pb2


def build_request(
    model_name: str, model_version: str, inputs: List[Input], output_keys: List[str]
) -> grpc_service_pb2.ModelInferRequest:
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

    return request


def encode_bytes(input: Input) -> bytes:
    string_contents = input.get_string_contents()
    encoded_list = [text.encode("utf8") for text in string_contents]
    encoded_array = np.array(encoded_list, dtype=np.bytes_).reshape(len(encoded_list), 1)
    serialized_array = _serialize_bytes_tensor(encoded_array)
    if serialized_array.size > 0:
        return serialized_array.item()
    else:
        return b""


def _serialize_bytes_tensor(input_tensor) -> np.array:
    """
    Serializes a bytes tensor into a flat numpy array of length prepended
    bytes. The numpy array should use dtype of np.object. For np.bytes,
    numpy will remove trailing zeros at the end of byte sequence and because
    of this it should be avoided.

    Parameters
    ----------
    input_tensor : np.array
        The bytes tensor to serialize.

    Returns
    -------
    serialized_bytes_tensor : np.array
        The 1-D numpy array of type uint8 containing the serialized bytes in row-major form.

    Raises
    ------
    InferenceServerException
        If unable to serialize the given tensor.
    """

    if input_tensor.size == 0:
        return np.empty([0], dtype=np.object_)

    # If the input is a tensor of string/bytes objects, then must flatten those into
    # a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in row-major
    # order.

    if (input_tensor.dtype != np.object_) and (input_tensor.dtype.type != np.bytes_):
        raise Exception("cannot serialize bytes tensor: invalid datatype")

    flattened_ls = []
    # 'C' order is row-major.
    for obj in np.nditer(input_tensor, flags=["refs_ok"], order="C"):
        # If directly passing bytes to BYTES type,
        # don't convert it to str as Python will encode the
        # bytes which may distort the meaning
        if input_tensor.dtype == np.object_:
            if isinstance(obj.item(), bytes):
                s = obj.item()
            else:
                s = str(obj.item()).encode("utf-8")
        else:
            s = obj.item()
        flattened_ls.append(struct.pack("<I", len(s)))
        flattened_ls.append(s)
    flattened = b"".join(flattened_ls)
    flattened_array = np.asarray(flattened, dtype=np.object_)
    if not flattened_array.flags["C_CONTIGUOUS"]:
        flattened_array = np.ascontiguousarray(flattened_array, dtype=np.object_)
    return flattened_array
