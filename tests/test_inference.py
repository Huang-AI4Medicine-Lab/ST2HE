"""
Unit tests for ST2HE inference module.
"""

import unittest
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestST2HEInference(unittest.TestCase):
    """Test cases for ST2HEInference class."""
    
    def test_import(self):
        """Test that the module can be imported."""
        try:
            from inference import ST2HEInference
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import ST2HEInference: {e}")
    
    def test_model_path_exists(self):
        """Test that default model path is accessible (if it exists)."""
        # This is a placeholder test
        # Update with actual model path validation if needed
        model_path = "/ix/yufeihuang/timothy/cycleGAN/img2img-turbo/output/cyclegan_turbo/dps_col_bracs/checkpoints/model_25001.pkl"
        # Check if path exists (optional - skip if model not available)
        if Path(model_path).exists():
            self.assertTrue(Path(model_path).exists())
        else:
            self.skipTest(f"Model path {model_path} does not exist")


if __name__ == "__main__":
    unittest.main()
