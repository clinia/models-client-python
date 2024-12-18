from typing import List

from pydantic import BaseModel


# NOTE: Pydantic only supports int and float by default. For extra types, we need to use numpy.
class Content(BaseModel):
    bool_contents = List[bool]
    int32_contents = List[int]
    fp32_contents = List[float]
    string_contents = List[str]
