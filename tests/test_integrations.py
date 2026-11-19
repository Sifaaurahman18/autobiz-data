import unittest
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from autobiz.integrations import (
    fetch_json_dataframe,
    json_to_dataframe,
    public_holiday_url,
)


class JsonIntegrationTests(unittest.TestCase):
    def test_converts_wrapped_records_and_flattens_nested_values(self):
        frame = json_to_dataframe({
            "data": [{"name": "Produk A", "price": 1200, "meta": {"sku": "A1"}}]
        })
        self.assertEqual(frame.loc[0, "name"], "Produk A")
        self.assertEqual(frame.loc[0, "meta.sku"], "A1")

    def test_converts_single_object_and_rejects_empty_response(self):
        frame = json_to_dataframe({"country": "ID", "count": 5})
        self.assertEqual(frame.loc[0, "country"], "ID")
        with self.assertRaises(ValueError):
            json_to_dataframe([])

    def test_public_holiday_endpoint_validates_inputs(self):
        self.assertEqual(
            public_holiday_url("id", 2026),
            "https://date.nager.at/api/v4/Holidays/ID/2026",
        )
        with self.assertRaises(ValueError):
            public_holiday_url("Indonesia", 2026)
        with self.assertRaises(ValueError):
            public_holiday_url("ID", 1999)

    def test_fetches_json_from_local_api_without_credentials(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": [{"product": "A", "stock": 3}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                del format, args
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            frame = fetch_json_dataframe(
                f"http://127.0.0.1:{server.server_port}/inventory"
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

        self.assertEqual(frame.loc[0, "product"], "A")
        self.assertEqual(frame.loc[0, "stock"], 3)


if __name__ == "__main__":
    unittest.main()
