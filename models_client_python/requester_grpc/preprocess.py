import struct

import numpy as np

from models_client_python.common.input import Input


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
