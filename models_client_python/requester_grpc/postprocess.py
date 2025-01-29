import struct
from typing import List

import numpy as np


# TODO: Unit test in [RET-2078,RET-2079]
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
