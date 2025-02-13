"""
Defines the output model structure for model inference results.

This module provides the Output class which represents the response data structure
from model inference requests, supporting various data types and matrix operations.
"""

from typing import List, Tuple

import numpy as np
from pydantic import BaseModel

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype


class Output(BaseModel, use_enum_values=True):
    """
    Represents a model's output tensor with shape and content information.

    This class handles different output data types and provides methods to
    extract and reshape the contents into usable matrix formats.

    Attributes:
        name (str): Name of the output tensor.
        shape (Tuple[int, ...]): Dimensions of the output tensor.
        datatype (Datatype): Data type of the output tensor elements.
        content (Content): The actual output data content.
    """

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
