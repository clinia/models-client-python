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
    """
    A class used to represent a Chunk of text.

    Attributes:
        id (str): The unique identifier for the chunk.
        text (str): The text content of the chunk.
        startIndex (int): The starting index of the chunk in the original text.
        endIndex (int): The ending index of the chunk in the original text.
        tokenCount (int): The number of tokens in the chunk.
    """

    id: str
    text: str
    startIndex: int
    endIndex: int
    tokenCount: int


class ChunkRequest(BaseModel):
    """
    Request model for chunking text.

    Attributes:
        texts (List[str]): The list of texts to be chunked.
        id (str): The unique identifier for the request.
    """

    texts: List[str]
    id: str = str(uuid4())


class ChunkResponse(BaseModel):
    """
    Response model for chunking text.

    Attributes:
        id (str): The unique identifier for the response, corresponding to that of the request.
        chunks (List[List[Chunk]] | None): The list of chunks in which each text is split. The outer list corresponds to the length of the input texts.
    """

    id: str
    chunks: List[List[Chunk]] | None = None


class Chunker(Client):
    """
    Chunker class for splitting text into smaller chunks.

    Methods:
        chunk: Synchronously chunk text using a specified model.
        chunk_async: Asynchronously chunk text using a specified model. Prefer this method if possible since it is non-blocking.
    """

    @staticmethod
    def _build_inputs(req: ChunkRequest) -> List[Input]:
        """
        Build the inputs required for the chunking model.

        Args:
            req (ChunkRequest): The request containing texts to be chunked.

        Returns:
            List[Input]: The list of inputs for the model.
        """
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
        """
        Process the outputs from the chunking model.

        Args:
            outputs (List[Output]): The outputs from the model.
            req (ChunkRequest): The original request.

        Returns:
            ChunkResponse: The response containing the chunks.
        """
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
        """
        Synchronously chunk text using a specified model.

        Args:
            model_name (str): The name of the model to use.
            model_version (str): The version of the model to use.
            req (ChunkRequest): The request containing texts to be chunked.

        Returns:
            ChunkResponse: The response containing the chunks.
        """
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
        """
        Asynchronously chunk text using a specified model.

        Args:
            model_name (str): The name of the model to use.
            model_version (str): The version of the model to use.
            req (ChunkRequest): The request containing texts to be chunked.

        Returns:
            ChunkResponse: The response containing the chunks.
        """
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
