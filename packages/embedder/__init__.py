from typing import List


class Embed:
    def __init__(self):
        return self.from_grpc()

    def infer(self, texts: List[str]) -> List[List[float]]:
        pass

    @classmethod
    def from_grpc():
        """
        The default init method for the Embed class.
        Initialize
        """
        return Embed()
