from typing import List, Tuple

from pydantic import BaseModel

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype


class Output(BaseModel, use_enum_values=True):
    name: str
    shape: Tuple[int, ...]
    datatype: Datatype
    contents: Content

    def get_fp32_contents(self) -> List[List[float]] | None:
        if self.datatype != Datatype.fp32:
            return None

        return self.contents.fp32_contents

    def get_string_contents(self) -> List[str] | None:
        if self.datatype != Datatype.bytes:
            return None

        return self.contents.string_contents
