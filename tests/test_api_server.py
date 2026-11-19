import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from autobiz.api_server import LocalApiServer
from autobiz.storage import ProjectStore


class LocalApiServerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temp_dir.name) / "api.sqlite3")
        self.project = self.store.save_project(
            "Penjualan", "/private/data/sales.csv"
        )
        self.store.record_analysis(self.project["id"], "EDA", "Ringkasan")
        self.server = LocalApiServer(self.store)
        self.base_url = self.server.start(port=0)

    def tearDown(self):
        self.server.stop()
        self.temp_dir.cleanup()

    def test_read_only_endpoints_return_project_and_history(self):
        with urlopen(f"{self.base_url}/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")

        with urlopen(f"{self.base_url}/api/v1/projects") as response:
            project = json.load(response)["projects"][0]
        self.assertEqual(project["name"], "Penjualan")
        self.assertNotIn("dataset_path", project)

        with urlopen(
            f"{self.base_url}/api/v1/projects/{self.project['id']}"
        ) as response:
            project_details = json.load(response)["project"]
        self.assertNotIn("dataset_path", project_details)

        with urlopen(
            f"{self.base_url}/api/v1/projects/{self.project['id']}/analyses"
        ) as response:
            analyses = json.load(response)["analyses"]
        self.assertEqual(analyses[0]["result"], "Ringkasan")

    def test_unknown_routes_and_write_requests_are_rejected(self):
        with self.assertRaises(HTTPError) as not_found:
            urlopen(f"{self.base_url}/missing")
        self.assertEqual(not_found.exception.code, 404)

        request = Request(
            f"{self.base_url}/api/v1/projects",
            data=b"{}",
            method="POST",
        )
        with self.assertRaises(HTTPError) as method_not_allowed:
            urlopen(request)
        self.assertEqual(method_not_allowed.exception.code, 405)


if __name__ == "__main__":
    unittest.main()
