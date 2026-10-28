import torch.nn as nn
import torch
from torch.autograd import Variable
from layers import CNN, Encoder, HybirdDecoder, GlobalUniformColorShift, GlobalVariationalColorShift, SpatialVariationalColorShift, PosConv

USE_CUDA = torch.cuda.is_available()


class SharedPositionHead(nn.Module):
    """Global-feature MLP with position embeddings (C) or hidden biases (D)."""
    def __init__(self, feature_size, positions, vocab_size, hidden_size, head,
                 position_embedding_dim=16):
        super().__init__()
        if head not in ('C', 'D'):
            raise ValueError("head must be C or D")
        if min(feature_size, positions, vocab_size, hidden_size, position_embedding_dim) < 1:
            raise ValueError("head dimensions must be positive")
        self.head = head
        if head == 'C':
            self.position_embedding = nn.Embedding(positions, position_embedding_dim)
            self.hidden = nn.Linear(feature_size + position_embedding_dim, hidden_size)
        else:
            self.hidden = nn.Linear(feature_size, hidden_size, bias=False)
            self.hidden_bias = nn.Parameter(torch.empty(positions, hidden_size))
            # Distinct initial biases break position symmetry even without teacher forcing.
            nn.init.normal_(self.hidden_bias, std=0.02)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, features):
        global_features = features.flatten(1)
        if self.head == 'C':
            embeddings = self.position_embedding.weight
            inputs = torch.cat((global_features.unsqueeze(1).expand(-1, embeddings.size(0), -1),
                                embeddings.unsqueeze(0).expand(features.size(0), -1, -1)), dim=-1)
            hidden = self.hidden(inputs)
        else:
            hidden = self.hidden(global_features).unsqueeze(1) + self.hidden_bias
        return self.output(torch.relu(hidden))


class CNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size=128, use_posconv=False, use_color_shift=False, posconv_size='', coordinate_method='interpolate', posconv_implementation='full', head='seq2seq', position_embedding_dim=16, image_height=64, image_width=128):
        
        super(CNNSeq2Seq, self).__init__()
        self.max_len = max_len
        if head not in ('seq2seq', 'C', 'D'):
            raise ValueError("head must be seq2seq, C, or D")
        self.head = head
        self.position_embedding_dim = position_embedding_dim
        self.hidden_size = hidden_size
        self.use_color_shift = use_color_shift
        if use_color_shift:
            self.vcs = GlobalVariationalColorShift()
        self.backbone = PosConv(posconv_size, coordinate_method=coordinate_method, posconv_implementation=posconv_implementation) if use_posconv else CNN()
        if head == 'seq2seq':
            self.encoder = Encoder(rnn_hidden_size=hidden_size)
            self.decoder = HybirdDecoder(vocab_size=vocab_size, hidden_size=hidden_size)
            self.prediction = nn.Linear(hidden_size, vocab_size)
        else:
            # Materialize parameters before optimizer/EMA creation without updating BN.
            self.backbone.eval()
            with torch.no_grad():
                feature_size = self.backbone(torch.zeros(1, 3, image_height, image_width)).numel()
            self.backbone.train()
            self.classifier = SharedPositionHead(feature_size, max_len - 1, vocab_size,
                                                 hidden_size, head, position_embedding_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0.0)

    def forward_train(self, x, y):
        if getattr(self, 'head', 'seq2seq') != 'seq2seq':
            return self._forward_position_head(x)
        out = x
        if self.use_color_shift:
            out = self.vcs(out)
        out = self.backbone(out)
        encoder_outputs = self.encoder(out)

        vocab_out = self.decoder.forward_train(encoder_outputs, self.max_len, y)
        vocab_out = self.prediction(vocab_out)
        return vocab_out

    def forward_test(self, x):  
        if getattr(self, 'head', 'seq2seq') != 'seq2seq':
            return self._forward_position_head(x)
        out = x
        if self.use_color_shift:
            out = self.vcs(out)
        out = self.backbone(out)
        encoder_outputs = self.encoder(out)

        outputs = []
        batch_size = x.size(0)
        input = torch.zeros([batch_size]).long()
        if USE_CUDA:
            input = input.cuda()

        last_hidden = Variable(torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size))
        if USE_CUDA:
            last_hidden = last_hidden.cuda()

        for i in range(self.max_len-1):
            output, last_hidden = self.decoder.forward_step(input, last_hidden, encoder_outputs)
            output = self.prediction(output)
            input = output.max(1)[1]
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)
        
    def forward_together(self, x_label, x_nolabel, y):
        if getattr(self, 'head', 'seq2seq') != 'seq2seq':
            logits = self._forward_position_head(torch.cat([x_label, x_nolabel], dim=0))
            return logits[:x_label.size(0)], logits[x_label.size(0):]
        x_all = torch.cat([x_label, x_nolabel], dim=0)
        out_all = x_all
        if self.use_color_shift:
            out_all = self.vcs(out_all)
        out_all = self.backbone(out_all)
        
        out_label = out_all[:x_label.size(0)]
        out_nolabel = out_all[x_label.size(0):]
        
        encoder_outputs = self.encoder(out_label)

        vocab_out = self.decoder.forward_train(encoder_outputs, self.max_len, y)
        vocab_out = self.prediction(vocab_out)
        
        encoder_outputs = self.encoder(out_nolabel)

        outputs = []
        batch_size = x_nolabel.size(0)
        input = torch.zeros([batch_size]).long()
        if USE_CUDA:
            input = input.cuda()

        last_hidden = Variable(torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size))
        if USE_CUDA:
            last_hidden = last_hidden.cuda()

        for i in range(self.max_len-1):
            output, last_hidden = self.decoder.forward_step(input, last_hidden, encoder_outputs)
            output = self.prediction(output)
            input = output.max(1)[1]
            outputs.append(output.unsqueeze(1))

        return vocab_out, torch.cat(outputs, dim=1)
    def forward(self, x):
        if getattr(self, 'head', 'seq2seq') != 'seq2seq':
            return self._forward_position_head(x)
        out = x
        if self.use_color_shift:
            out = self.vcs(out)      
        out = self.backbone(out)
        encoder_outputs = self.encoder(out)

        outputs = []
        batch_size = x.size(0)
        input = torch.zeros([batch_size]).long()
        if USE_CUDA:
            input = input.cuda()

        last_hidden = Variable(torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size))
        if USE_CUDA:
            last_hidden = last_hidden.cuda()

        for i in range(self.max_len-1):
            output, last_hidden = self.decoder.forward_step(input, last_hidden, encoder_outputs)
            output = self.prediction(output)
            input = output.max(1)[1]
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def _forward_position_head(self, x):
        if self.use_color_shift:
            x = self.vcs(x)
        return self.classifier(self.backbone(x))
