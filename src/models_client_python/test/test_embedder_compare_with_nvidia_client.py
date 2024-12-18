from typing import List
from uuid import uuid4

import numpy as np

from clients.py.test import DATA_PATH, MODEL_NAME, MODEL_VERSION, TEXTS, TRITON_URL


def test_inference_official():
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype

    ## NOTE: there must be a better way than defining the class inside the method, but otherwise we get a TypeError: Couldn't build proto file into descriptor pool: duplicate file name model_config.proto since both the official client and our grpc_generated use the same .proto files
    class EmbedderClientNvidia:
        def __init__(self, url: str) -> None:
            self.url = url

        @staticmethod
        def prepare_outputs(name: str) -> grpcclient.InferRequestedOutput:
            outputs = [grpcclient.InferRequestedOutput(name)]

            return outputs

        @staticmethod
        def prepare_inputs(texts: List[str] | str):
            if isinstance(texts, str):
                texts = [texts]

            encoded_list = [text.encode("utf8") for text in texts]
            encoded_array = np.array(encoded_list, dtype=np.bytes_).reshape(len(encoded_list), 1)

            tensor = grpcclient.InferInput(
                name="text",
                shape=encoded_array.shape,
                datatype=np_to_triton_dtype(encoded_array.dtype),
            )
            tensor.set_data_from_numpy(encoded_array)

            return [tensor]

        def get_embeddings(self, model_name: str, model_version: str, texts: List[str]) -> np.ndarray:
            ## Prepare Triton inputs and outputs
            inputs = self.prepare_inputs(texts=texts)
            outputs = self.prepare_outputs(name="embedding")
            request_id = str(uuid4())

            with grpcclient.InferenceServerClient(
                url=self.url,
                ssl=False,
                verbose=False,
            ) as triton_client:
                # Send request
                response = triton_client.infer(
                    model_name=model_name,
                    model_version=model_version,
                    inputs=inputs,
                    outputs=outputs,
                    request_id=request_id,
                )

            return response.as_numpy("embedding")

    client = EmbedderClientNvidia(url=TRITON_URL)
    official_embeddings = client.get_embeddings(model_name=MODEL_NAME, model_version=MODEL_VERSION, texts=TEXTS)

    assert official_embeddings.size != 0
    assert official_embeddings.shape == (2, 384)

    with open(f"{DATA_PATH}/official_embeddings.npy", "wb") as f:
        np.save(f, official_embeddings)


def test_compare():
    generated = np.load(f"{DATA_PATH}/generated_embeddings.npy")
    official = np.load(f"{DATA_PATH}/official_embeddings.npy")

    np.testing.assert_array_equal(generated, official, strict=True, verbose=True)
