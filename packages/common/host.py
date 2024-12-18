from enum import StrEnum

from pydantic import BaseModel


class HostScheme(StrEnum):
    http = "http"
    https = "https"


class Host(BaseModel):
    url: str
    port: int
    scheme: HostScheme

    def __str__(self) -> str:
        """
        Returns the host in the format of 'scheme://url:port'
        """
        return f"{self.scheme}://{self.url}:{self.port}"

    def host(self) -> str:
        """
        Returns the host in the format of 'url:port'
        """
        return f"{self.url}:{self.port}"
