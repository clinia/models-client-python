import logging
from typing import List
from uuid import uuid4

from pydantic import BaseModel

from models_client_python.client import Client
from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.input import Input
from models_client_python.common.output import Output

_RANKER_QUERY_INPUT_KEY = "query"
_RANKER_QUERY_INPUT_DATATYPE = Datatype.bytes

_RANKER_TEXT_INPUT_KEY = "passage"
_RANKER_TEXT_INPUT_DATATYPE = Datatype.bytes

_RANKER_OUTPUT_KEY = "score"


class RankRequest(BaseModel):
    """
    Request model for ranking passages based on a query.

    Attributes:
        query (str): The query to rank the passages against.
        texts (List[str]): The list of passages to be ranked.
        id (str): The unique identifier for the request. If not provided, a random UUIDv4 will be generated.
    """

    query: str
    texts: List[str]
    id: str = str(uuid4())


class RankResponse(BaseModel):
    """
    Response model for ranking passages based on a query.

    Attributes:
        id (str): The unique identifier for the response, corresponding to that of the request.
        scores (List[float] | None): The list of scores for each pair of query and passage.
    """

    id: str
    scores: List[float] | None = None


class Ranker(Client):
    """
    Ranker class for using the ranker model to rank passages based on a query.

    Methods:
        rank: Synchronously rank passages using a specified model.
        rank_async: Asynchronously rank passages using a specified model. Prefer this method if possible since it is non-blocking.
    """

    @staticmethod
    def _build_inputs(req: RankRequest) -> List[Input]:
        """
        Build the inputs required for the ranking model.

        Args:
            req (RankRequest): The request containing the query and texts to be ranked.

        Returns:
            List[Input]: The list of inputs for the model.
        """
        input_queries = [req.query] * len(req.texts)
        return [
            Input(
                id=req.id,
                name=_RANKER_QUERY_INPUT_KEY,
                shape=(len(input_queries), 1),
                datatype=_RANKER_QUERY_INPUT_DATATYPE,
                content=Content(string_contents=input_queries),
            ),
            Input(
                id=req.id,
                name=_RANKER_TEXT_INPUT_KEY,
                shape=(len(req.texts), 1),
                datatype=_RANKER_TEXT_INPUT_DATATYPE,
                content=Content(string_contents=req.texts),
            ),
        ]

    @staticmethod
    def _process_outputs(outputs: List[Output], req: RankRequest) -> RankResponse:
        """
        Process the outputs from the ranking model.

        Args:
            outputs (List[Output]): The outputs from the model.
            req (RankRequest): The original request.

        Returns:
            RankResponse: The response containing the scores.
        """
        if not outputs:
            raise ValueError("No outputs received")

        # Since we have only one output, we can directly access the first output.
        scores = outputs[0].get_fp32_matrix_contents()

        # Flatten the 2D slice into a 1D slice
        flattened_scores = []
        for score in scores:
            if len(score) != 1:
                raise ValueError(f"Expected a single score per passage, but got {len(score)} elements")
            flattened_scores.extend(score)

        return RankResponse(id=req.id, scores=flattened_scores)

    def rank(self, model_name: str, model_version: str, req: RankRequest) -> RankResponse:
        """
        Synchronously rank passages using a specified model.

        Args:
            model_name (str): The name of the model to use.
            model_version (str): The version of the model to use.
            req (RankRequest): The request containing the query and texts to be ranked.

        Returns:
            RankResponse: The response containing the scores.
        """
        inputs = self._build_inputs(req)

        try:
            outputs = self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_RANKER_OUTPUT_KEY],
            )
            return self._process_outputs(outputs, req)

        except ValueError as e:
            logging.error(f"Error while chunking: {e}")
            return RankResponse(id=req.id)

    async def rank_async(self, model_name: str, model_version: str, req: RankRequest) -> RankResponse:
        """
        Asynchronously rank passages using a specified model.

        Args:
            model_name (str): The name of the model to use.
            model_version (str): The version of the model to use.
            req (RankRequest): The request containing the query and texts to be ranked.

        Returns:
            RankResponse: The response containing the scores.
        """
        inputs = self._build_inputs(req)

        try:
            outputs = await self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_RANKER_OUTPUT_KEY],
            )
            return self._process_outputs(outputs, req)

        except ValueError as e:
            logging.error(f"Error while chunking: {e}")
            return RankResponse(id=req.id)
