from typing import List, Tuple

from common.content import Content
from common.datatype import Datatype
from pydantic import BaseModel


class Output(BaseModel):
    name: str
    shape: Tuple[int, ...]
    Datatype: Datatype
    Contents: List[Content]
