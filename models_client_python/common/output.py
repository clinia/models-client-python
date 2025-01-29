from typing import List, Tuple

import numpy as np
from pydantic import BaseModel

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype


class Output(BaseModel, use_enum_values=True):
    name: str
    shape: Tuple[int, ...]
    datatype: Datatype
    content: Content

    def get_fp32_matrix_contents(self) -> List[List[float]] | None:
        if self.datatype != Datatype.fp32:
            return None

        return np.array(self.content.fp32_contents).reshape(self.shape).tolist()

    def get_string_matrix_contents(self) -> List[List[str]] | None:
        if self.datatype != Datatype.bytes:
            return None

        return np.array(self.content.string_contents).reshape(self.shape).tolist()
