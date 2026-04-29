import os
from thetadata.client import Client
from dotenv import load_dotenv
from typing import Literal

load_dotenv()


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
