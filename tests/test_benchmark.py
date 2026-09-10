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
                "import shutil, sys\n"
                "shutil.copyfile(sys.argv[1], sys.argv[2])\n"
                "print('Loading models: 0.001 second')\n"
                "print('TOTAL: 0:00:00.002')\n",
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
                self.assertGreater(float(row["psnr_y_db"]), 0)
                self.assertGreater(float(row["psnr_u_db"]), 0)
                self.assertGreater(float(row["psnr_v_db"]), 0)
                self.assertEqual(row["ssim_y"], "")
                self.assertEqual(row["ms_ssim_y"], "")
                self.assertEqual(row["lpips_alex"], "")
            ai = next(row for row in rows if row["codec"] == "jpeg-ai")
            self.assertEqual(float(ai["encode_codec_ms"]), 2)
            self.assertEqual(float(ai["encode_model_load_ms"]), 1)
            self.assertGreater(float(ai["encode_total_ms"]), 3)
            self.assertGreaterEqual(float(ai["encode_overhead_ms"]), 0)

    def test_yuv_psnr_is_infinite_for_identical_images(self) -> None:
        image = Image.new("RGB", (2, 2), (20, 40, 80))
        self.assertTrue(all(value == float("inf") for value in benchmark.yuv_psnr(image, image)))


if __name__ == "__main__":
    unittest.main()
