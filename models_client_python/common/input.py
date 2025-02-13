"""
Defines the input model structure for model inference requests.

This module provides the Input class which represents the request data structure
for model inference, supporting various data types and content formats.
"""

from typing import List, Tuple

from pydantic import BaseModel

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype


class Input(BaseModel, use_enum_values=True):
    """
    Represents a model's input tensor with shape and content information.

    This class handles different input data types and provides methods to
    access the contents in appropriate formats.

    Attributes:
        id (str): Unique identifier for the input.
        name (str): Name of the input tensor.
        shape (Tuple[int, ...]): Dimensions of the input tensor.
        datatype (Datatype): Data type of the input tensor elements.
        content (Content): The actual input data content.
    """

    id: str
    name: str
    shape: Tuple[int, ...]
    datatype: Datatype
    content: Content

    def get_string_contents(self) -> List[str] | None:
        if self.datatype != Datatype.bytes:
            return None

        return self.content.string_contents
