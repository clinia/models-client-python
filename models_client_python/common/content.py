from typing import List

from pydantic import BaseModel


# NOTE: Pydantic only supports int and float by default. For extra types, we need to use numpy.
class Content(BaseModel):
    bool_contents: List[bool] | None = None
    int32_contents: List[int] | List[List[int]] | None = None
    fp32_contents: List[float] | List[List[float]] | None = None
    string_contents: List[str] | None = None
