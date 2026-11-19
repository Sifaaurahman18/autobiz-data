import unittest

import pandas as pd

from autobiz.business import business_summary


class BusinessSummaryTests(unittest.TestCase):
    def test_reports_detected_business_fields_and_unavailable_categories(self):
        summary = business_summary(pd.DataFrame({
            "sales_revenue": [100, 200],
            "stock": [0, -1],
            "customer_id": ["A", "B"],
        }))
        self.assertIn("PENJUALAN / PENDAPATAN", summary)
        self.assertIn("total 300.00", summary)
        self.assertIn("nilai negatif 1", summary)
        self.assertIn("2 nilai unik", summary)
        self.assertIn("Kolom yang sesuai belum terdeteksi", summary)


if __name__ == "__main__":
    unittest.main()
