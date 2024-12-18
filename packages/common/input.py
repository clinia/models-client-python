from typing import List, Tuple

from common.content import Content
from common.datatype import Datatype
from pydantic import BaseModel


class Input(BaseModel):
    name: str
    shape: Tuple[int, ...]
    datatype: Datatype
    contents: List[Content]
