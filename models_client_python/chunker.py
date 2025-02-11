import logging
from typing import List
from uuid import uuid4

from pydantic import BaseModel

from models_client_python.client import Client
from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.input import Input
from models_client_python.common.output import Output

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


class Chunker(Client):
    @staticmethod
    def _build_inputs(req: ChunkRequest) -> List[Input]:
        return [
            Input(
                id=req.id,
                name=_CHUNKER_INPUT_KEY,
                shape=(len(req.texts), 1),
                datatype=_CHUNKER_INPUT_DATATYPE,
                content=Content(string_contents=req.texts),
            )
        ]

    @staticmethod
    def _process_outputs(outputs: List[Output], req: ChunkRequest) -> ChunkResponse:
        if not outputs:
            raise ValueError("No outputs received")

        # Since we have only one output, we can directly access the first output.
        texts_chunks = outputs[0].get_string_matrix_contents()
        formatted_texts_chunks = []
        for text_chunks in texts_chunks:
            formatted_text_chunks = [Chunk.model_validate_json(chunk) for chunk in text_chunks if chunk != "pad"]
            formatted_texts_chunks.append(formatted_text_chunks)

        return ChunkResponse(id=req.id, chunks=formatted_texts_chunks)

    def chunk(self, model_name: str, model_version: str, req: ChunkRequest) -> ChunkResponse:
        inputs = self._build_inputs(req)

        try:
            outputs = self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_CHUNKER_OUTPUT_KEY],
            )
            return self._process_outputs(outputs, req)

        except ValueError as e:
            logging.error(f"Error while chunking: {e}")
            return ChunkResponse(id=req.id)

    async def chunk_async(self, model_name: str, model_version: str, req: ChunkRequest) -> ChunkResponse:
        inputs = self._build_inputs(req)

        try:
            outputs = await self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_CHUNKER_OUTPUT_KEY],
            )
            return self._process_outputs(outputs, req)

        except ValueError as e:
            logging.error(f"Error while chunking: {e}")
            return ChunkResponse(id=req.id)
