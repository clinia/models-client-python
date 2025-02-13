"""
Defines host configuration structures for model server connections.

This module provides classes for specifying and formatting host connection details,
supporting both HTTP and HTTPS schemes.
"""

from enum import StrEnum

from pydantic import BaseModel


class HostScheme(StrEnum):
    http = "http"
    https = "https"


class Host(BaseModel):
    """
    Configuration model for server host details.

    This class handles the formatting of host information and provides methods
    to generate connection strings in different formats.

    Attributes:
        url (str): The host URL or IP address.
        port (int): The port number for the connection.
        scheme (HostScheme): The connection scheme (http/https).
    """

    url: str
    port: int
    scheme: HostScheme

    def __str__(self) -> str:
        """
        Returns the host in the format of 'scheme://url:port'
        """
        return f"{self.scheme}://{self.url}:{self.port}"

    def target(self) -> str:
        """
        Returns the host in the format of 'url:port'
        """
        return f"{self.url}:{self.port}"
