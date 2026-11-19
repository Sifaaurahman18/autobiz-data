import tempfile
import unittest
from pathlib import Path

from autobiz.storage import ProjectStore


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temp_dir.name) / "projects.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_project_and_analysis_history_survive_store_reopen(self):
        project = self.store.save_project(
            "Penjualan", "/tmp/sales.csv", source="file"
        )
        self.store.record_analysis(project["id"], "EDA", "Hasil ringkasan")

        reopened = ProjectStore(self.store.database_path)
        loaded_project = reopened.get_project(project["id"])
        if loaded_project is None:
            self.fail("Proyek tidak ditemukan setelah database dibuka ulang.")
        self.assertEqual(loaded_project["name"], "Penjualan")
        history = reopened.list_analyses(project["id"])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["result"], "Hasil ringkasan")

    def test_project_name_and_analysis_reference_are_validated(self):
        with self.assertRaises(ValueError):
            self.store.save_project(" ", None)
        with self.assertRaises(ValueError):
            self.store.record_analysis("missing", "EDA", "Result")

    def test_analysis_history_limit_is_bounded(self):
        project = self.store.save_project("Operasional", None)
        with self.assertRaises(ValueError):
            self.store.list_analyses(project["id"], limit=501)


if __name__ == "__main__":
    unittest.main()
