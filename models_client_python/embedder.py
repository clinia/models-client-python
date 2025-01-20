from typing import List
from uuid import uuid4

from pydantic import BaseModel

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.input import Input
from models_client_python.common.requester import Requester, RequesterConfig
from models_client_python.requester_grpc.requester import RequesterGrpc

_EMBEDDER_INPUT_KEY = "text"
_EMBEDDER_OUTPUT_KEY = "embedding"
_EMBEDDER_INPUT_DATATYPE = Datatype.bytes


class EmbedRequest(BaseModel):
    texts: List[str]
    id: str = str(uuid4())


class EmbedResponse(BaseModel):
    id: str
    embeddings: List[List[float]] | None = None


class Embedder:
    def __init__(self, requester: Requester):
        self.requester = requester

    @classmethod
    def from_grpc(cls, config: RequesterConfig):
        return cls(requester=RequesterGrpc(config=config))

    def embed(self, model_name: str, model_version: str, req: EmbedRequest) -> EmbedResponse:
        inputs: List[Input] = [
            Input(
                id=req.id,
                name=_EMBEDDER_INPUT_KEY,
                shape=(len(req.texts), 1),
                datatype=_EMBEDDER_INPUT_DATATYPE,
                contents=Content(string_contents=req.texts),
            )
        ]

        try:
            outputs = self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_EMBEDDER_OUTPUT_KEY],
            )

            # Since we have only one output, we can directly access the first output.
            embeddings = outputs[0].get_fp32_contents()
            return EmbedResponse(id=req.id, embeddings=embeddings)

        except ValueError as e:
            return EmbedResponse(id=req.id)
