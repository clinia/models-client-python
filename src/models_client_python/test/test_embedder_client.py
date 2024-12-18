import numpy as np

from clients.py.src.embedder_client import EmbedderClient
from clients.py.test import DATA_PATH, MODEL_NAME, MODEL_VERSION, TEXTS, TRITON_URL, EmbedderResults


def test_inference_single():
    client = EmbedderClient(url=TRITON_URL)
    embedding = client.get_embeddings(model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS[0])

    assert embedding.size != 0
    assert embedding.shape == (1, 384)


def test_inference_batch():
    client = EmbedderClient(url=TRITON_URL)
    embeddings = client.get_embeddings(model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS)

    assert embeddings.size != 0
    assert embeddings.shape == (2, 384)


def test_inference_single_vs_batch():
    client = EmbedderClient(url=TRITON_URL)

    embedding_0 = client.get_embeddings(model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS[0])
    embedding_1 = client.get_embeddings(model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS[1])

    embeddings = client.get_embeddings(model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS)

    np.testing.assert_array_equal(embedding_0, embeddings[0].reshape(1, -1), strict=True, verbose=True)
    np.testing.assert_array_equal(embedding_1, embeddings[1].reshape(1, -1), strict=True, verbose=True)


def test_compare_expected():
    client = EmbedderClient(url=TRITON_URL)
    embeddings = client.get_embeddings(model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS)

    with open(DATA_PATH, "r") as f:
        results = EmbedderResults.model_validate_json(f)

    print(results.data)
