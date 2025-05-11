# test_seed_sizer.py
import unittest
import pandas as pd
from SeedSizer import Run

class TestSeedSizer(unittest.TestCase):
    def test_various_filter_thresholds(self):
        for i in range(1, 20):  # 0.01 - .1
            min_area = i / 100
            with self.subTest(min_area=min_area):
                df = Run('Data/Calena-18.tif', min_area)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertIn("area_mm2", df.columns)
                self.assertTrue(len(df) >= 0)  # Just checks no crash

if __name__ == '__main__':
    unittest.main()