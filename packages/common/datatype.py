from enum import StrEnum


class Datatype(StrEnum):
    bool = "BOOL"
    int32 = "INT32"
    fp32 = "FP32"
    bytes = "BYTES"  # Bytes can be used for any datatype but it is mostly used for string
