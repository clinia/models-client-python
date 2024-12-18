from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from clients.py.generated import grpc_service_pb2
from clients.py.src import InferInput, TritonClient


@dataclass
class TextWithParams:
    text: str
    params: str


def prepare_inputs(texts_with_params: List[TextWithParams]) -> List[InferInput]:
    texts = []
    params = []

    for text_with_params in texts_with_params:
        texts.append(text_with_params.text)
        params.append(text_with_params.params)

    return [InferInput.from_texts(texts, name="text"), InferInput.from_texts(params, name="params")]


class PassageExtractorClient(TritonClient):
    def get_passages(self, model_name: str, model_version: str, texts: List[str]):
        output_name = "passages"

        inputs = prepare_inputs(texts)
        requested_outputs = [grpc_service_pb2.ModelInferRequest.InferRequestedOutputTensor(name=output_name)]

        response = self.infer(
            model_name=model_name,
            model_version=model_version,
            inputs=inputs,
            outputs=requested_outputs,
        )

        return response.as_numpy(name=output_name)


if __name__ == "__main__":
    TRITON_URL = "127.0.0.1:8001"
    MODEL_NAME = "passage_extractor"
    MODEL_VERSION = "120240308103333"

    TEXTS = [
        TextWithParams(text="Hello. This is a new passage,", params='{"format":"plain","maxLength":256,"overlap":64}'),
        TextWithParams(
            text="Can Biomechanical Testing After Anterior Cruciate Ligament Reconstruction identify",
            params='{"format":"plain","maxLength":256,"overlap":64}',
        ),
        TextWithParams(text="This is another test,", params='{"format":"plain","maxLength":256,"overlap":64}'),
    ]

    client = PassageExtractorClient(url=TRITON_URL)
    passages = client.get_passages(model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS)

    print(passages)
    print(passages.shape)
