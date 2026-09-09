import csv
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import benchmark


class BenchmarkSmokeTest(unittest.TestCase):
    def test_jpeg_run_writes_a_valid_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            pixels = [
                ((x * 17) % 256, (y * 19) % 256, ((x + y) * 13) % 256)
                for y in range(16)
                for x in range(16)
            ]
            image = Image.new("RGB", (16, 16))
            image.putdata(pixels)
            image.save(source)

            fake_codec = root / "fake_codec.py"
            fake_codec.write_text(
                "import shutil, sys\nshutil.copyfile(sys.argv[1], sys.argv[2])\n",
                encoding="utf-8",
            )
            codec_command = " ".join(
                [
                    shlex.quote(sys.executable),
                    shlex.quote(str(fake_codec)),
                    "{input}",
                    "{output}",
                ]
            )

            output = root / "results.csv"
            exit_code = benchmark.main(
                [
                    str(source),
                    "--jpeg-quality",
                    "80",
                    "--warmup",
                    "0",
                    "--repeats",
                    "1",
                    "--jpeg-ai-encoder",
                    codec_command,
                    "--jpeg-ai-decoder",
                    codec_command,
                    "--jpeg-ai-points",
                    "copy",
                    "--work-dir",
                    str(root / "artifacts"),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["codec"] for row in rows}, {"jpeg", "jpeg-ai"})
            self.assertTrue((root / "artifacts" / "sources" / source.name).is_file())
            for row in rows:
                self.assertGreater(float(row["bits_per_pixel"]), 0)
                self.assertGreater(float(row["psnr_db"]), 0)


if __name__ == "__main__":
    unittest.main()
