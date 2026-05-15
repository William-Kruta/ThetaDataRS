import os
from importlib import import_module
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()


def _client_class_from_module(module: Any) -> type:
    client_class = getattr(module, "Client", None)
    if client_class is not None:
        return client_class

    theta_client_class = getattr(module, "ThetaClient", None)
    if theta_client_class is not None:
        return theta_client_class

    raise ImportError("thetadata.client must expose Client or ThetaClient")


def _resolve_client_class() -> type:
    return _client_class_from_module(import_module("thetadata.client"))


Client = _resolve_client_class()


def create_client(
    email: str = None,
    passwd: str = None,
    dataframe_return_type: Literal["pandas", "polars"] = "polars",
) -> Client:
    if email is None:
        email = os.getenv("EMAIL")
    if passwd is None:
        passwd = os.getenv("PASSWD")
    client = Client(email=email, password=passwd, dataframe_type=dataframe_return_type)
    return client
