from typing import List
from uuid import uuid4

from pydantic import BaseModel

from models_client_python.common.content import Content
from models_client_python.common.datatype import Datatype
from models_client_python.common.input import Input
from models_client_python.common.requester import Requester, RequesterConfig
from models_client_python.requester_grpc.requester import RequesterGrpc

_RANKER_QUERY_INPUT_KEY = "query"
_RANKER_QUERY_INPUT_DATATYPE = Datatype.bytes

_RANKER_TEXT_INPUT_KEY = "passage"
_RANKER_TEXT_INPUT_DATATYPE = Datatype.bytes

_RANKER_OUTPUT_KEY = "score"


class RankRequest(BaseModel):
    query: str
    texts: List[str]
    id: str = str(uuid4())


class RankResponse(BaseModel):
    id: str
    scores: List[float] | None = None


class Ranker:
    def __init__(self, requester: Requester):
        self.requester = requester

    @classmethod
    def from_grpc(cls, config: RequesterConfig):
        return cls(requester=RequesterGrpc(config=config))

    def rank(self, model_name: str, model_version: str, req: RankRequest) -> RankResponse:
        ## Extend query to be the same size as texts
        input_queries = [req.query] * len(req.texts)

        inputs: List[Input] = [
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

        try:
            outputs = self.requester.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                output_keys=[_RANKER_OUTPUT_KEY],
            )

            if outputs:
                # Since we have only one output, we can directly access the first output.
                scores = outputs[0].get_fp32_matrix_contents()

                # Flatten the 2D slice into a 1D slice
                flattened_scores = []
                for score in scores:
                    if len(score) != 1:
                        raise ValueError(f"Expected a single score per passage, but got {len(score)} elements")
                    flattened_scores.extend(score)

                return RankResponse(id=req.id, scores=flattened_scores)
            else:
                return RankResponse(id=req.id)

        except ValueError as e:
            return RankResponse(id=req.id)
