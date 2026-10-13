import argparse
import ast
from pathlib import Path
import unittest

import torch

from models import CNNSeq2Seq, SharedPositionHead


class ClassifierHeadTests(unittest.TestCase):
    def test_c_to_d_equivalence(self):
        torch.manual_seed(42)
        c = SharedPositionHead(12, 4, 7, 8, 'C', 16).double()
        d = SharedPositionHead(12, 4, 7, 8, 'D').double()
        with torch.no_grad():
            d.hidden.weight.copy_(c.hidden.weight[:, :12])
            d.hidden_bias.copy_(c.position_embedding.weight @ c.hidden.weight[:, 12:].T
                                + c.hidden.bias)
            d.output.load_state_dict(c.output.state_dict())
        features = torch.randn(3, 3, 4, dtype=torch.double, requires_grad=True)
        torch.testing.assert_close(c(features), d(features), atol=1e-12, rtol=1e-12)
        self.assertFalse(torch.allclose(d(features)[:, 0], d(features)[:, 1]))
        for head in (c, d):
            head(features).square().mean().backward()
            for parameter in head.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_model_paths_and_state_dict(self):
        for head in ('C', 'D'):
            for backbone in ('cnn', 'full', 'sequential'):
                kwargs = dict(vocab_size=7, max_len=5, hidden_size=8, head=head,
                              use_posconv=backbone != 'cnn', posconv_size='large',
                              posconv_implementation='full' if backbone == 'cnn' else backbone,
                              image_width=64)
                model = CNNSeq2Seq(**kwargs).eval()
                x = torch.randn(2, 3, 64, 64)
                y = torch.zeros(2, 5, dtype=torch.long)
                logits = model(x)
                self.assertEqual(logits.shape, (2, 4, 7))
                torch.testing.assert_close(logits, model.forward_train(x, y))
                torch.testing.assert_close(logits, model.forward_test(x))
                labeled, unlabeled = model.forward_together(x[:1], x[1:], y[:1])
                torch.testing.assert_close(logits, torch.cat((labeled, unlabeled)))
                logits.square().mean().backward()
                self.assertTrue(all(p.grad is not None for p in model.parameters()))
                restored = CNNSeq2Seq(**kwargs).eval()
                restored.load_state_dict(model.state_dict())
                torch.testing.assert_close(logits, restored(x))

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
            self.assertEqual(parser.parse_args([]).head, 'seq2seq')
            for head in ('C', 'D'):
                args = parser.parse_args(['--head', head, '--position-embedding-dim', '8'])
                self.assertEqual(args.head, head)
                self.assertEqual(args.position_embedding_dim, 8)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
