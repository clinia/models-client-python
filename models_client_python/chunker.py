from typing import List
from uuid import uuid4

from pydantic import BaseModel

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.input import Input
from models_client_python.common.requester import Requester, RequesterConfig
from models_client_python.requester_grpc.requester import RequesterGrpc

_CHUNKER_INPUT_KEY = "text"
_CHUNKER_OUTPUT_KEY = "chunk"
_CHUNKER_INPUT_DATATYPE = Datatype.bytes


class Chunk(BaseModel):
    id: str
    text: str
    startIndex: int
    endIndex: int
    tokenCount: int


class ChunkRequest(BaseModel):
    texts: List[str]
    id: str = str(uuid4())


class ChunkResponse(BaseModel):
    id: str
    chunks: List[List[Chunk]] | None = None


class Chunker:
    def __init__(self, requester: Requester):
        self.requester = requester

    @classmethod
    def from_grpc(cls, config: RequesterConfig):
        return cls(requester=RequesterGrpc(config=config))

    def chunk(self, model_name: str, model_version: str, req: ChunkRequest) -> ChunkResponse:
        inputs: List[Input] = [
            Input(
                id=req.id,
                name=_CHUNKER_INPUT_KEY,
                shape=(len(req.texts), 1),
                datatype=_CHUNKER_INPUT_DATATYPE,
                content=Content(string_contents=req.texts),
            )
        ]

        try:
            outputs = self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_CHUNKER_OUTPUT_KEY],
            )

            if outputs:
                # Since we have only one output, we can directly access the first output.
                texts_chunks = outputs[0].get_string_matrix_contents()

                # Format chunks as class
                formatted_texts_chunks = []
                for text_chunks in texts_chunks:
                    formatted_text_chunks = [
                        Chunk.model_validate_json(chunk) for chunk in text_chunks if chunk != "pad"
                    ]
                    formatted_texts_chunks.append(formatted_text_chunks)

                return ChunkResponse(id=req.id, chunks=formatted_texts_chunks)
            else:
                return ChunkResponse(id=req.id)

        except ValueError as e:
            return ChunkResponse(id=req.id)
