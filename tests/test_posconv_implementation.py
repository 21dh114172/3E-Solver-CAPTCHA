import argparse
import ast
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from layers import PosConv2DLayer, SequentialPosConv2DLayer, generate_2d_position_maps
from models import CNNSeq2Seq


class PosConvImplementationTests(unittest.TestCase):
    def test_entrypoint_options(self):
        for filename in ('FixMatch_2.py', 'FixMatch_2_supervise.py'):
            tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
            nodes = [node for node in tree.body
                     if (isinstance(node, ast.Assign) and any(
                         isinstance(t, ast.Name) and t.id == 'parser' for t in node.targets))
                     or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Attribute)
                         and isinstance(node.value.func.value, ast.Name)
                         and node.value.func.value.id == 'parser')]
            scope = {'argparse': argparse}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), scope)
            parser = scope['parser']
            self.assertEqual(parser.parse_args([]).posconv_implementation, 'full')
            self.assertEqual(parser.parse_args(['--posconv-implementation', 'sequential'])
                             .posconv_implementation, 'sequential')

    def test_sequential_formula_and_boundaries(self):
        x = torch.zeros(2, 1, 5, 7)
        maps = generate_2d_position_maps(20, 28, batch_size=2)
        for method in ('interpolate', 'regenerate'):
            layer = SequentialPosConv2DLayer(1, 1, 3, padding=1,
                                             coordinate_method=method).eval()
            with torch.no_grad():
                layer.horizontal.weight.fill_(1)
                layer.vertical.weight.fill_(1)
                layer.Wpos_x.fill_(2)
                layer.Wpos_y.fill_(3)
                layer.bpos_x.fill_(4)
                layer.bpos_y.fill_(5)
            if method == 'regenerate':
                cx = ((torch.arange(7) - 1) / 7).view(1, 1, 1, 7)
                cy = ((torch.arange(5) - 1) / 5).view(1, 1, 5, 1)
            else:
                cy, cx = [F.interpolate(m, size=(5, 7), mode='bilinear',
                                        align_corners=False) for m in maps]
            horizontal = F.conv2d(x, layer.horizontal.weight, padding=(0, 1)) + 2 * cx + 4
            expected = F.conv2d(horizontal, layer.vertical.weight, padding=(1, 0)) + 3 * cy + 5
            original, offset, expanded, out = layer(x, *maps, return_intermediates=True)
            torch.testing.assert_close(expanded, expected)
            torch.testing.assert_close(original + offset, expected)
            torch.testing.assert_close(out, layer.relu(layer.bn(expected)))
            # Top row sees two padded convolution taps, interior sees three.
            torch.testing.assert_close((expanded - 3 * cy - 5)[:, :, 0, :],
                                       2 * horizontal[:, :, 0, :])
            out.sum().backward()
            self.assertIsNotNone(layer.Wpos_x.grad)
            self.assertIsNotNone(layer.vertical.weight.grad)

    def test_all_backbone_combinations(self):
        for implementation in ('full', 'sequential'):
            for method in ('interpolate', 'regenerate'):
                model = CNNSeq2Seq(10, 5, use_posconv=True, posconv_size='large',
                                  coordinate_method=method,
                                  posconv_implementation=implementation).eval()
                with torch.no_grad():
                    output = model.backbone(torch.randn(2, 3, 64, 128))
                self.assertEqual(output.shape, (2, 32, 256))
                for name in ('conv6', 'conv7', 'conv8', 'conv9'):
                    layer = getattr(model.backbone, name)
                    self.assertEqual(layer.coordinate_method, method)
                    self.assertIsInstance(layer, PosConv2DLayer if implementation == 'full'
                                          else SequentialPosConv2DLayer)

    def test_parameter_counts_and_initialization(self):
        full = PosConv2DLayer(256, 256, 3, padding=1)
        sequential = SequentialPosConv2DLayer(256, 256, 3, padding=1)
        self.assertEqual(sum(p.numel() for p in full.parameters()), 591104)
        self.assertEqual(sum(p.numel() for p in sequential.parameters()), 394752)
        for name in ('Wpos_x', 'Wpos_y', 'bpos_x', 'bpos_y'):
            self.assertEqual(torch.count_nonzero(getattr(sequential, name)).item(), 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
