import json
import logging
from typing import Dict, List
from uuid import uuid4

from pydantic import BaseModel

from models_client_python.client import Client
from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.host import Host, HostScheme
from models_client_python.common.input import Input
from models_client_python.common.output import Output
from models_client_python.common.requester import RequesterConfig

_EMBEDDER_INPUT_KEY = "text"
_EMBEDDER_OUTPUT_KEY = "embedding"
_EMBEDDER_INPUT_DATATYPE = Datatype.bytes


class SparseEmbedRequest(BaseModel):
    """
    Request model for generating embeddings from text.

    Attributes:
        texts (List[str]): The list of texts to be embedded.
        id (str): The unique identifier for the request.
    """

    texts: List[str]
    id: str = str(uuid4())


class SparseEmbedResponse(BaseModel):
    """
    Response model for generating embeddings from text.

    Attributes:
        id (str): The unique identifier for the response, corresponding to that of the request.
        embeddings (List[Dict[str, float]] | None): The list of embeddings for each text. Each embedding is a dictionary with tokens as keys and weights as values. The outer list length matches the number of input texts.
    """

    id: str
    embeddings: List[Dict[str, float]] | None = None


class SparseEmbedder(Client):
    """
    Embedder class for generating embeddings from text.

    Methods:
        embed: Synchronously generate embeddings using a specified model.
        embed_async: Asynchronously generate embeddings using a specified model. Prefer this method if possible since it is non-blocking.
    """

    @staticmethod
    def _build_inputs(req: SparseEmbedRequest) -> List[Input]:
        """
        Build the inputs required for the embedding model.

        Args:
            req (EmbedRequest): The request containing texts to be embedded.

        Returns:
            List[Input]: The list of inputs for the model.
        """
        return [
            Input(
                name=_EMBEDDER_INPUT_KEY,
                shape=(len(req.texts), 1),
                datatype=_EMBEDDER_INPUT_DATATYPE,
                content=Content(string_contents=req.texts),
            )
        ]

    @staticmethod
    def _process_outputs(outputs: List[Output], req: SparseEmbedRequest) -> SparseEmbedResponse:
        """
        Process the outputs from the embedding model.

        Args:
            outputs (List[Output]): The outputs from the model.
            req (EmbedRequest): The original request.

        Returns:
            EmbedResponse: The response containing the embeddings.
        """
        if not outputs:
            raise ValueError("No outputs received")

        # Since we have only one output, we can directly access the first output.
        string_embeddings = outputs[0].get_string_matrix_contents()
        embeddings = [json.loads(embedding) for embedding in string_embeddings]
        return SparseEmbedResponse(id=req.id, embeddings=embeddings)

    def embed(self, model_name: str, model_version: str, req: SparseEmbedRequest) -> SparseEmbedResponse:
        """
        Synchronously generate embeddings using a specified model.

        Args:
            model_name (str): The name of the model to use.
            model_version (str): The version of the model to use.
            req (EmbedRequest): The request containing texts to be embedded.

        Returns:
            EmbedResponse: The response containing the embeddings.
        """
        inputs = self._build_inputs(req)
        if not req.texts:
            raise ValueError("No texts provided for embedding.")
        try:
            outputs = self.requester.infer(
                id=req.id,
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_EMBEDDER_OUTPUT_KEY],
            )
            return self._process_outputs(outputs, req)

        except ValueError as e:
            logging.error(f"Error while sparse embedding: {e}")
            return SparseEmbedResponse(id=req.id)

    async def embed_async(self, model_name: str, model_version: str, req: SparseEmbedRequest) -> SparseEmbedResponse:
        """
        Asynchronously generate embeddings using a specified model.

        Args:
            model_name (str): The name of the model to use.
            model_version (str): The version of the model to use.
            req (EmbedRequest): The request containing texts to be embedded.

        Returns:
            EmbedResponse: The response containing the embeddings.
        """
        inputs = self._build_inputs(req)

        try:
            outputs = await self.requester.infer(
                id=req.id,
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_EMBEDDER_OUTPUT_KEY],
            )
            return self._process_outputs(outputs, req)

        except ValueError as e:
            logging.error(f"Error while embedding: {e}")
            return SparseEmbedResponse(id=req.id)
