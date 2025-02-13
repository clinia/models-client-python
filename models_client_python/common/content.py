"""
Defines the content container for model inputs and outputs.

This module provides the Content class which acts as a container for different
types of data that can be passed to or received from models.
"""

from typing import List

from pydantic import BaseModel


class Content(BaseModel):
    """
    Container for different types of model input/output data.

    This class provides a flexible structure to hold various types of content
    that can be used in model inference requests and responses.

    NOTE: Pydantic only supports int and float by default. For extra types, we need to use numpy.

    Attributes:
        bool_contents (List[bool] | None): Boolean type content.
        int32_contents (List[int] | None): 32-bit integer type content.
        fp32_contents (List[float] | None): 32-bit float type content.
        string_contents (List[str] | None): String type content.
    """

    bool_contents: List[bool] | None = None
    int32_contents: List[int] | None = None
    fp32_contents: List[float] | None = None
    string_contents: List[str] | None = None
