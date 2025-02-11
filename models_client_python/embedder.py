import logging
from typing import List
from uuid import uuid4

from pydantic import BaseModel

from models_client_python.client import Client
from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.input import Input
from models_client_python.common.output import Output

_EMBEDDER_INPUT_KEY = "text"
_EMBEDDER_OUTPUT_KEY = "embedding"
_EMBEDDER_INPUT_DATATYPE = Datatype.bytes


class EmbedRequest(BaseModel):
    texts: List[str]
    id: str = str(uuid4())


class EmbedResponse(BaseModel):
    id: str
    embeddings: List[List[float]] | None = None


class Embedder(Client):
    @staticmethod
    def _build_inputs(req: EmbedRequest) -> List[Input]:
        return [
            Input(
                id=req.id,
                name=_EMBEDDER_INPUT_KEY,
                shape=(len(req.texts), 1),
                datatype=_EMBEDDER_INPUT_DATATYPE,
                content=Content(string_contents=req.texts),
            )
        ]

    @staticmethod
    def _process_outputs(outputs: List[Output], req: EmbedRequest) -> EmbedResponse:
        if not outputs:
            raise ValueError("No outputs received")

        # Since we have only one output, we can directly access the first output.
        embeddings = outputs[0].get_fp32_matrix_contents()
        return EmbedResponse(id=req.id, embeddings=embeddings)

    def embed(self, model_name: str, model_version: str, req: EmbedRequest) -> EmbedResponse:
        inputs = self._build_inputs(req)

        try:
            outputs = self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_EMBEDDER_OUTPUT_KEY],
            )
            return self._process_outputs(outputs, req)

        except ValueError as e:
            logging.error(f"Error while chunking: {e}")
            return EmbedResponse(id=req.id)

    async def embed_async(self, model_name: str, model_version: str, req: EmbedRequest) -> EmbedResponse:
        inputs = self._build_inputs(req)

        try:
            outputs = await self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_EMBEDDER_OUTPUT_KEY],
            )
            return self._process_outputs(outputs, req)

        except ValueError as e:
            logging.error(f"Error while chunking: {e}")
            return EmbedResponse(id=req.id)
