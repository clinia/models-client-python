from typing import List, Tuple

from pydantic import BaseModel

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype


class Input(BaseModel, use_enum_values=True):
    id: str
    name: str
    shape: Tuple[int, ...]
    datatype: Datatype
    content: Content

    def get_string_contents(self) -> List[str] | None:
        if self.datatype != Datatype.bytes:
            return None

        return self.content.string_contents
