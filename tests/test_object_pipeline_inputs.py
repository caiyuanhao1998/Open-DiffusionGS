"""Regression checks for user-provided object images."""
import unittest

import numpy as np
from PIL import Image
import torch

from diffusionGS.pipline_obj import DiffusionGSPipeline


class ObjectInputTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = DiffusionGSPipeline.__new__(DiffusionGSPipeline)

    def test_transparent_input_reports_missing_foreground(self):
        image = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "No foreground"):
            self.pipeline.preprocess_image([image])

    def test_foreground_ratio_is_validated_before_resizing(self):
        image = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
        for ratio in (0, -0.5, 1.5, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "Foreground ratio"):
                self.pipeline.preprocess_image([image], foreground_ratio=ratio)

    def test_thin_foreground_keeps_at_least_one_pixel(self):
        image = Image.new("RGBA", (256, 16), (255, 0, 0, 0))
        image.paste((255, 0, 0, 128), (0, 7, 256, 8))
        result = self.pipeline.preprocess_image([image], foreground_ratio=0.5)[0]
        self.assertEqual(result.size, (256, 256))
        self.assertIsNotNone(result.getchannel("A").getbbox())

    def test_palette_transparency_is_preserved(self):
        image = Image.new("P", (16, 16), 0)
        image.putpalette([255, 255, 255, 255, 0, 0] + [0] * 762)
        image.info["transparency"] = 0
        image.paste(1, (4, 4, 12, 12))
        result = self.pipeline.preprocess_image([image])[0]
        self.assertEqual(result.mode, "RGBA")
        self.assertTrue(np.asarray(result.getchannel("A")).any())

    def test_tensor_inputs_reject_multiple_images_and_wrong_channels(self):
        for shape in ((3, 16, 16), (1, 4, 16, 16)):
            self.pipeline.check_inputs(torch.ones(shape))
        for shape in ((2, 3, 16, 16), (1, 16, 16), (16, 16)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "one RGB/RGBA tensor"):
                self.pipeline.check_inputs(torch.ones(shape))


if __name__ == "__main__":
    unittest.main()
