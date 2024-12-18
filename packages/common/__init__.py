from __future__ import annotations

import struct
from typing import List
from uuid import uuid4

import grpc
import numpy as np

from clients.py.generated import grpc_service_pb2, grpc_service_pb2_grpc, model_config_pb2


class InferInput:
    def __init__(
        self,
        data: np.array,
        name: str,
        datatype: str,
    ):
        self.tensor = grpc_service_pb2.ModelInferRequest().InferInputTensor()
        self.tensor.name = name
        self.tensor.datatype = datatype
        self.tensor.ClearField("shape")
        self.tensor.shape.extend(data.shape)

        serialized_output = self._serialize_byte_tensor(data)
        if serialized_output.size > 0:
            self.raw_content = serialized_output.item()
        else:
            self.raw_content = b""

    @classmethod
    def from_texts(cls, texts: List[str] | str, name: str):
        if isinstance(texts, str):
            texts = [texts]

        encoded_list = [text.encode("utf8") for text in texts]
        encoded_array = np.array(encoded_list, dtype=np.bytes_).reshape(len(encoded_list), 1)

        return cls(name=name, datatype="BYTES", data=encoded_array)

    @staticmethod
    def _serialize_byte_tensor(input_tensor):
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


class InferOutput:
    def __init__(
        self,
        response: grpc_service_pb2.ModelInferResponse,
    ):
        self.response = response

    @staticmethod
    def _triton_to_np_dtype(dtype):
        if dtype == "BOOL":
            return bool
        elif dtype == "INT8":
            return np.int8
        elif dtype == "INT16":
            return np.int16
        elif dtype == "INT32":
            return np.int32
        elif dtype == "INT64":
            return np.int64
        elif dtype == "UINT8":
            return np.uint8
        elif dtype == "UINT16":
            return np.uint16
        elif dtype == "UINT32":
            return np.uint32
        elif dtype == "UINT64":
            return np.uint64
        elif dtype == "FP16":
            return np.float16
        elif dtype == "FP32" or dtype == "BF16":
            return np.float32
        elif dtype == "FP64":
            return np.float64
        elif dtype == "BYTES":
            return np.object_
        return None

    @staticmethod
    def _deserialize_bytes_tensor(encoded_tensor):
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

    def as_numpy(self, name: str) -> np.array | None:
        """Get the tensor data for output associated with this object
        in numpy format

        Parameters
        ----------
        name : str
            The name of the output tensor whose result is to be retrieved.

        Returns
        -------
        numpy array
            The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
        """
        index = 0
        for output in self.response.outputs:
            if output.name == name:
                shape = []
                for value in output.shape:
                    shape.append(value)

                datatype = output.datatype
                if index < len(self.response.raw_output_contents):
                    if datatype == "BYTES":
                        # String results contain a 4-byte string length
                        # followed by the actual string characters. Hence,
                        # need to decode the raw bytes to convert into
                        # array elements.
                        np_array = self._deserialize_bytes_tensor(self.response.raw_output_contents[index])
                    # elif datatype == "BF16":
                    #     np_array = deserialize_bf16_tensor(response.raw_output_contents[index])
                    else:
                        np_array = np.frombuffer(
                            self.response.raw_output_contents[index],
                            dtype=self._triton_to_np_dtype(datatype),
                        )
                elif len(output.contents.bytes_contents) != 0:
                    np_array = np.array(output.contents.bytes_contents, copy=False)
                else:
                    np_array = np.empty(0)
                np_array = np_array.reshape(shape)
                return np_array
            else:
                index += 1
        return None


class TritonClient:
    def __init__(self, url: str) -> None:
        self._channel = grpc.insecure_channel(url)
        self._client_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(self._channel)

    def _format_request(
        self,
        model_name: str,
        model_version: str,
        request_id: str,
        inputs: List[InferInput],
        outputs: List[InferOutput],
    ) -> grpc_service_pb2.ModelInferRequest:
        request = grpc_service_pb2.ModelInferRequest()

        request.model_name = model_name
        request.model_version = model_version
        request.id = request_id

        for infer_input in inputs:
            request.inputs.extend([infer_input.tensor])
            if infer_input.raw_content is not None:
                request.raw_input_contents.extend([infer_input.raw_content])

        for infer_output in outputs:
            request.outputs.extend([infer_output])

        return request

    def infer(
        self,
        model_name: str,
        model_version: str,
        inputs: List[InferInput],
        outputs: List[InferOutput],
        request_id: str | None = None,
    ) -> grpc_service_pb2.ModelInferResponse:
        if not request_id:
            request_id = str(uuid4())

        request = self._format_request(
            model_name=model_name,
            model_version=model_version,
            request_id=request_id,
            inputs=inputs,
            outputs=outputs,
        )

        response = self._client_stub.ModelInfer(request=request)

        return InferOutput(response)

    def close(self):
        self._channel.close()

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()
