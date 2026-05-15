import types
import unittest
from unittest.mock import patch

import thetadatars.client as client_module
from thetadatars import ThetaDataRS, create_client


class ClientCompatibilityTests(unittest.TestCase):
    def test_resolver_prefers_client_when_available(self):
        class FakeClient:
            pass

        class FakeThetaClient:
            pass

        module = types.SimpleNamespace(Client=FakeClient, ThetaClient=FakeThetaClient)

        self.assertIs(client_module._client_class_from_module(module), FakeClient)

    def test_resolver_falls_back_to_theta_client(self):
        class FakeThetaClient:
            pass

        module = types.SimpleNamespace(ThetaClient=FakeThetaClient)

        self.assertIs(client_module._client_class_from_module(module), FakeThetaClient)

    def test_create_client_uses_resolved_client_class(self):
        calls = []

        class FakeClient:
            def __init__(self, *, email, password, dataframe_type):
                calls.append(
                    {
                        "email": email,
                        "password": password,
                        "dataframe_type": dataframe_type,
                    }
                )

        with patch.object(client_module, "Client", FakeClient):
            created = create_client(
                email="user@example.com",
                passwd="secret",
                dataframe_return_type="polars",
            )

        self.assertIsInstance(created, FakeClient)
        self.assertEqual(
            calls,
            [
                {
                    "email": "user@example.com",
                    "password": "secret",
                    "dataframe_type": "polars",
                }
            ],
        )

    def test_package_import_smoke(self):
        self.assertTrue(callable(create_client))
        self.assertEqual(ThetaDataRS.__name__, "ThetaDataRS")


if __name__ == "__main__":
    unittest.main()
