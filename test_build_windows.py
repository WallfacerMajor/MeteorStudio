import unittest

from build_windows import HIDDEN_IMPORTS, pyinstaller_arguments


class WindowsBuildConfigurationTests(unittest.TestCase):
    def test_required_runtime_modules_are_explicit(self):
        self.assertIn("background_tasks", HIDDEN_IMPORTS)
        self.assertIn("meteor_detection", HIDDEN_IMPORTS)
        arguments = pyinstaller_arguments(r"C:\tools\ffmpeg.exe")
        self.assertIn(r"C:\tools\ffmpeg.exe;.", arguments)

    def test_build_without_ffmpeg_has_no_empty_binary_argument(self):
        arguments = pyinstaller_arguments(None)
        self.assertNotIn("--add-binary", arguments)


if __name__ == "__main__":
    unittest.main()
