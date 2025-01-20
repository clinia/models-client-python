from enum import StrEnum


class Datatype(StrEnum):
    bool = "BOOL"  # bool
    int32 = "INT32"  # np.int32
    fp32 = "FP32"  # np.float32
    # Bytes can be used for any datatype but it is mostly used for strings
    bytes = "BYTES"  # np.object_
