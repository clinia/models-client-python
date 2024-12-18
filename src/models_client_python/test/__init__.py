import json
from pathlib import Path
from typing import List

import numpy as np
from pydantic import BaseModel

DATA_PATH = f"{Path(__file__).parent}/data/results.json"

TRITON_URL = "127.0.0.1:8001"
MODEL_NAME = "embedder_bge-small-en-v1.5"
MODEL_VERSION = "3000000001"

TEXTS = ["Hello, how are you?", "Clinia is based in Montreal"]


class EmbedderResults(BaseModel):
    shape: List[int]
    datatype: str
    data: List[float]

    @classmethod
    def from_numpy(cls, data: np.Array):
        return cls(
            shape=data.shape,
            datatype=str(data.dtype),
            data=data.flatten().tolist(),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.model_dump(), f)


def save_results():
    from clients.py.src.embedder_client import EmbedderClient

    client = EmbedderClient(url=TRITON_URL)
    embeddings = client.get_embeddings(
        model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS
    )

    assert embeddings.size != 0
    assert embeddings.shape == (2, 384)

    results = EmbedderResults.from_numpy(data=embeddings)
    results.save(path=DATA_PATH)
