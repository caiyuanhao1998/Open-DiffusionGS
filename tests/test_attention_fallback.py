"""Check SDPA shapes and the reference-view attention mask used on Spaces."""
import unittest

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusionGS.models.transformers import dit, utils_transformer


class AttentionFallbackTests(unittest.TestCase):
    def test_subset_matches_explicit_attention_mask(self):
        for module in (dit, utils_transformer):
            with self.subTest(module=module.__name__):
                layer = module.SelfAttention(32, 8, use_flashatt_v2=False).eval()
                inputs = torch.randn(2, 7, 32)
                q, k, v = [rearrange(tensor, "b l (h d) -> b h l d", d=8)
                           for tensor in layer.to_qkv(inputs).split(32, dim=2)]
                mask = torch.zeros(7, 7, dtype=torch.bool)
                mask[:3, :3] = True
                mask[3:, :] = True
                attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                expected = layer.fc(rearrange(attended, "b h l d -> b l (h d)"))
                actual = layer(inputs, subset_attention_size=3)
                self.assertEqual(actual.shape, inputs.shape)
                torch.testing.assert_close(actual, expected)

    def test_full_attention_preserves_shape(self):
        for module in (dit, utils_transformer):
            with self.subTest(module=module.__name__):
                layer = module.SelfAttention(32, 8, use_flashatt_v2=False).eval()
                inputs = torch.randn(2, 7, 32)
                self.assertEqual(layer(inputs).shape, inputs.shape)


if __name__ == "__main__":
    unittest.main()
