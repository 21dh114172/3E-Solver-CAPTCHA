# 3E-Solver-IJCAI

This reposity is official implementation of "3E-Slover: An Effortless, Easy-to-Update, and End-to-End Solver with Semi-Supervised Learning for Breaking Text-Based Captchas" (IJCAI-2022).

## Dataset
https://drive.google.com/file/d/1iL4me0ispkHiNsie6Fr_HrnfRh7PsK8y/view?usp=sharing

```
unzip dataset.zip
mkdir result
```

## Dependency
```
torch=1.9.0+cu111
torchvision=0.10.0+cu111
numpy=1.21.2
PIL=8.3.1
matplotlib=3.4.3
```

## Code
``` shell
python FixMatch_2.py --dataset apple --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset ganji-1 --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset google --label 500.txt --unlabeled-number 5000
python FixMatch_2.py --dataset microsoft --label 500.txt --unlabeled-number 5000
python FixMatch_2.py --dataset sina --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset weibo --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset wikipedia --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset yandex --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
```

## Result

| Scheme      | Accuracy    |
| ----------- | ----------- |
| Google      | 76.4%       |
| Microsoft   | 96.4%       |
| Yandex      | 90.4%       |
| Wikiepdia   | 98.5%       |
| Weibo       | 92.5%       |
| Sina        | 97.1%       |
| Apple       | 92.9%       |
| Ganji       | 99.4%       |


### Classifier head

Both training scripts accept `--head {seq2seq,C,D}` (default: `seq2seq`, the
existing encoder/decoder). C and D replace the encoder/decoder with a shared
two-layer ReLU MLP over the entire flattened backbone output:

- `C`: concatenate a learned position embedding before the shared hidden layer.
  Set its width with `--position-embedding-dim` (default 16).
- `D`: shared hidden weight matrix with a separate hidden bias per position.
  Biases start with small random values to distinguish positions.

Both share the output weights and output bias. `--hidden-size` sets the MLP
hidden width. These are reconstruction choices, not confirmed paper designs.

```bash
python FixMatch_2.py --use-posconv --posconv-size large --head D
python FixMatch_2_supervise.py --use-posconv --posconv-size large --head C --position-embedding-dim 16
```

The heads predict all `MAXLEN - 1` target slots simultaneously (including any
EOS/padding slots used by the dataset); teacher forcing does not apply. Existing
losses and EMA training retain their output contract. The feature dimension is
inferred from `--TARGET-HEIGHT` and `--TARGET-WIDTH` before optimizer creation;
use the same image dimensions when loading a checkpoint. No extra pooling is
introduced, so large images increase head parameters. Checkpoints record `head`,
`position_embedding_dim`, and `hidden_size`; pass the matching options when
resuming. Existing checkpoints use `seq2seq`; C/D/seq2seq state dicts are not
interchangeable.

### Corrected PosConv positional computation

Both training entry points now default to `--positional-mode corrected` and
`--coordinate-padding 2`. The padding option counts learned coordinate samples
per side, independently of visual convolution padding. Full and sequential
variants retain their existing convolution, BN, ReLU and pooling order.

```bash
python FixMatch_2.py --use-posconv --posconv-size large --posconv-implementation sequential --positional-mode corrected --coordinate-padding 2
```

Modes are `legacy` (previous positional calculation), `linear` (learned
slope/intercept without boundary substitution), `corrected` (adds boundary
correction), and `explicit` (direct reference sum). Compare `linear` and
`corrected` with identical weights and coordinate padding for a boundary ablation.
New side weights start random in both variants; legacy slopes remain registered
but inactive in the new modes. `coordinate_method` only affects `legacy` mode.

Checkpoints record `positional_mode` and `coordinate_padding`; supply matching
CLI options when resuming. Old PosConv state dicts lack the new side weights and
cannot be resumed directly through the training scripts' strict loader. For
legacy inference, construct with `positional_mode='legacy'`, load with
`strict=False`, and verify that missing keys are exclusively
`coordinate_x.left/right` and `coordinate_y.left/right` weights and that there
are no unexpected keys. This does not produce a pretrained corrected model.

See [the supplied correction notes](posconv_corrected/README.md) for the coordinate
convention and research limitations. Run the installed implementation's tests
with `python -B -m unittest discover -s tests -v`; the reference checks use
`posconv_corrected/original/layers.py` as the compatibility fixture.

### Legacy PosConv coordinate method

Both `FixMatch_2.py` and `FixMatch_2_supervise.py` accept
`--coordinate-method {interpolate,regenerate}` with `--use-posconv --positional-mode legacy`:

- `interpolate` (default): generate original-image position maps and resize them
  to each feature resolution using bilinear interpolation.
- `regenerate`: create coordinates at each feature resolution using
  `(arange(width) - kernel_width // 2) / width`, and the equivalent for height.

Example (the same flags work with `FixMatch_2_supervise.py`):

```bash
python FixMatch_2.py --use-posconv --posconv-size large --positional-mode legacy --coordinate-method regenerate
```

The selected method is recorded in checkpoint metadata as `coordinate_method`.
When reconstructing a model from a state dict, pass the same method to
`CNNSeq2Seq`; older checkpoints use `interpolate` by default.


### PosConv layer implementation

Both training entry points accept `--posconv-implementation {full,sequential}`:

- `full` (default): one full 3x3 convolution, then both positional offsets and
  one positional bias, followed by BN and ReLU. Legacy position weights start random.
- `sequential`: a 1x3 convolution with X position and bias, then a 3x1
  convolution with Y position and bias, followed by BN and ReLU. Position
  slopes and biases start at zero in legacy mode. The vertical convolution processes the
  horizontal position term, with zero padding applied separately on each axis.

Legacy coordinate generation is selected with `--coordinate-method`.
For sequential convolutions with coordinates regenerated at each resolution:

```bash
python FixMatch_2.py --use-posconv --posconv-size large --posconv-implementation sequential --positional-mode legacy --coordinate-method regenerate
```

The same flags work with `FixMatch_2_supervise.py`. This selects the positional
layers only; backbone pooling, residual blocks, encoder, and decoder retain
their existing structure. Checkpoints record `posconv_implementation`; reconstruct
with the same value and coordinate method. Full and sequential layer weights
have different shapes and cannot be loaded interchangeably. Older state-dict
checkpoints default to `full`.
