"""Unit tests for the ST2HE inference module."""

import importlib
from pathlib import Path
import sys
import unittest

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestST2HEInference(unittest.TestCase):
    """Test cases for ST2HEInference class."""

    def tearDown(self):
        sys.modules.pop("inference", None)

    def test_import_is_lightweight(self):
        """Importing the module should not require external pix2pix repo setup."""
        module = importlib.import_module("inference")
        self.assertTrue(hasattr(module, "ST2HEInference"))

    def test_missing_weights_file_fails_fast(self):
        """A missing weights file should raise before model dependencies are touched."""
        inference = importlib.import_module("inference")
        missing_path = "/tmp/definitely_missing_st2he_weights.pkl"
        with self.assertRaises(FileNotFoundError) as exc:
            inference.ST2HEInference(model_path=missing_path)
        self.assertIn(missing_path, str(exc.exception))


if __name__ == "__main__":
    unittest.main()
