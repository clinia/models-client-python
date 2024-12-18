from __future__ import annotations

from typing import List

import numpy as np

from clients.py.generated import grpc_service_pb2
from clients.py.src import InferInput, TritonClient


class EmbedderClient(TritonClient):
    def get_embeddings(self, model_name: str, model_version: str, texts: List[str]):
        input_name = "text"
        output_name = "embedding"

        inputs = [InferInput.from_texts(texts=texts, name=input_name)]
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
    MODEL_NAME = "embedder_medical_journals_qa"
    MODEL_VERSION = "120240905185426"

    TEXTS = ["hello, how are you?", "Clinia is based in Montreal"]

    client = EmbedderClient(url=TRITON_URL)
    embedding = client.get_embeddings(model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS)

    print(embedding)
    print(embedding.shape)
