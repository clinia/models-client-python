"""
Defines the supported data types for model inputs and outputs.

This module provides an enumeration of supported data types that can be used
in model inference requests and responses, mapping to Python types.
"""

from enum import StrEnum


class Datatype(StrEnum):
    """
    Enumeration of supported data types for model inputs and outputs.

    Each enum value corresponds to a specific data type that can be used in
    model inference, with mappings to Python types. numpy types are also suggested.

    NOTE: Bytes can be used for any datatype but it is mostly used for strings
    """

    bool = "BOOL"  # bool
    int32 = "INT32"  # np.int32
    fp32 = "FP32"  # np.float32
    bytes = "BYTES"  # np.object_
