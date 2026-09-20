import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, TiffImagePlugin

from alignment_workspace import AlignmentWorkspace


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class ReferenceFocalInputTests(unittest.TestCase):
    def test_reference_without_exif_requires_user_value(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.png"
            Image.new("RGB", (80, 60), "black").save(path)
            fake = SimpleNamespace(
                sensor_diagonal=FakeVar(43.2666),
                reference_focal_length=FakeVar(""),
            )
            with self.assertRaisesRegex(ValueError, "参考图没有EXIF"):
                AlignmentWorkspace._reference_focal(fake, path)
            fake.reference_focal_length.value = "20"
            focal, source = AlignmentWorkspace._reference_focal(fake, path)
        self.assertEqual(focal, 20.0)
        self.assertIn("用户填写", source)

    def test_reference_exif_overrides_text_field(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.jpg"
            exif = Image.Exif()
            exif[37386] = TiffImagePlugin.IFDRational(35, 1)
            Image.new("RGB", (80, 60), "black").save(path, exif=exif)
            fake = SimpleNamespace(
                sensor_diagonal=FakeVar(43.2666),
                reference_focal_length=FakeVar("50"),
            )
            focal, source = AlignmentWorkspace._reference_focal(fake, path)
        self.assertEqual(focal, 35.0)
        self.assertEqual(source, "EXIF")


if __name__ == "__main__":
    unittest.main()
