from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import numpy as np
from common.content import Content
from common.datatype import Datatype
from common.input import Input
from pydantic import BaseModel


class EmbedderRequest(BaseModel):
    id: str
    texts: List[str]


class EmbedderResponse(BaseModel):
    id: str
    embeddings: List[List[float]]


# class Embedder(ABC):
#     @abstractmethod
#     def embed(self, model_name: str, model_version: str, req: EmbedderRequest, out: List[str]) -> EmbedderResponse:
#         pass

#     @abstractmethod
#     @classmethod
#     def from_grpc(cls):
#         pass


class EmbedderInfo(Enum):
    embedderInputKey: str = "text"
    embedderOutputKey: str = "embedding"
    embedderInputDatatype: Datatype = Datatype.Bytes


class Embedder:
    def __init__(self, requester: Any):
        self.requester = self.from_grpc()

    def embed(self, model_name: str, model_version: str, req: EmbedderRequest) -> EmbedderResponse:
        inputs: List[Input] = [
            Input(
                name=EmbedderInfo.embedderInputKey,
                shape=len(req.texts),
                datatype=EmbedderInfo.embedderInputDatatype,
                contents=[Content(string_contents=req.texts)],
            )
        ]

        try:
            response = self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=[EmbedderInfo.embedderOutputKey],
            )
        except ValueError as e:
            return EmbedderResponse(id=req.id)

        return response.as_numpy(name=EmbedderInfo.embedderOutputKey)

    def close(self):
        return self.requester.close()
