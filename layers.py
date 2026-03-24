import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

USE_CUDA = torch.cuda.is_available()


def generate_2d_position_maps(height, width, batch_size=1, device='cpu'):
    pos_y = torch.linspace(-1.0, 1.0, steps=height, device=device)
    pos_x = torch.linspace(-1.0, 1.0, steps=width, device=device)
    grid_y, grid_x = torch.meshgrid(pos_y, pos_x) # HxW
    pos_map_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    pos_map_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return pos_map_y, pos_map_x


class PosConv(nn.Module):
    """
    CNN with positional encoding support for 2D image processing.
    
    Uses a combination of residual blocks and positional convolutional layers
    with pooling and dropout for feature extraction.
    
    Attributes:
        conv1-5: Residual blocks for feature extraction
        conv6-9: Positional convolutional layers
        pool1-5: Max pooling layers
        dropout_1-9: Dropout layers (p=0.1)
        pos_map_y_orig, pos_map_x_orig: Cached positional maps
        
    Methods:
        forward(x): Processes input through the network with positional encoding
    """
    def __init__(self, pos_size='large'):
        super().__init__()
        self.pos_size = pos_size  # 'small' | 'medium' | 'large'
        
        self.conv1 = ResBlk(3, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout(0.1)
        
        # Pos‐conv before conv2 (only large)
        self.conv8 = PosConv2DLayer(32, 32, kernel_size=3, padding=1)
        self.dropout_8 = nn.Dropout(0.1)
        
        self.conv2 = ResBlk(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout_2 = nn.Dropout(0.1)
        
        # Pos‐conv before conv3 (only large)
        self.conv7 = PosConv2DLayer(64, 64, kernel_size=3, padding=1)
        self.dropout_7 = nn.Dropout(0.1)
        
        self.conv3 = ResBlk(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout_3 = nn.Dropout(0.1)
        
        # Pos‐conv before conv4 (medium+large)
        self.conv6 = PosConv2DLayer(128, 128, kernel_size=3, padding=1)
        self.dropout_6 = nn.Dropout(0.1)
        
        self.conv4 = ResBlk(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout_4 = nn.Dropout(0.1)
        
        # Pos‐conv before conv5 (always)
        self.conv9 = PosConv2DLayer(256, 256, kernel_size=3, padding=1)
        self.dropout_9 = nn.Dropout(0.1)
        
        self.conv5 = ResBlk(256, 256)
        self.pool5 = nn.MaxPool2d(kernel_size=(4, 1))
        self.dropout_5 = nn.Dropout(0.1)

        
        self.pos_map_y_orig = None
        self.pos_map_x_orig = None


    def forward(self, x, return_intermediates=False):
        batch_size, _, H, W = x.shape

        # Generate or retrieve position maps for this batch/input size
        # (Generating on the fly handles varying batch sizes)
        if self.pos_map_y_orig is None or self.pos_map_y_orig.shape[2:] != (H, W):
             self.pos_map_y_orig, self.pos_map_x_orig = generate_2d_position_maps(H, W, batch_size=1, device=x.device)
             # print(f"Generated pos maps for size: {H}x{W}") # Debug

        # Repeat maps for current batch size
        # Do this inside if needed, or pass batch_size to generate_2d_position_maps
        pos_y = self.pos_map_y_orig.repeat(batch_size, 1, 1, 1)
        pos_x = self.pos_map_x_orig.repeat(batch_size, 1, 1, 1)


        # CNN Backbone
        
        out = x
        
        out = self.conv1(out)
        out = self.pool1(out)
        out = self.dropout_1(out)
        
        # large → apply conv8
        if self.pos_size == 'large':
            out = self.conv8(out, pos_y, pos_x)
            out = self.dropout_8(out)
        
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.dropout_2(out)
        
        # large → apply conv7
        if self.pos_size == 'large':
            out = self.conv7(out, pos_y, pos_x)
            out = self.dropout_7(out)
        
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.dropout_3(out)
        
        # medium or large → apply conv6
        if self.pos_size in ('medium','large'):
            out = self.conv6(out, pos_y, pos_x)
            out = self.dropout_6(out)
        
        out = self.conv4(out)
        out = self.pool4(out)
        out = self.dropout_4(out)
        
        # always apply conv9 before conv5
        out = self.conv9(out, pos_y, pos_x)
        out = self.dropout_9(out)
        
        out = self.conv5(out)
        out = self.pool5(out)
        out = self.dropout_5(out)
        
        
        
        out = out.squeeze(2)
        out = out.transpose(1, 2)

        return out

class CNN(nn.Module):
    """
    input: [batch_size, 3, 64, 128]
    output: [batch_size, 32, 256]
    """

    def __init__(self):
        super(CNN, self).__init__()
        
        self.resblk_1 = ResBlk(3, 32)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout(0.1)
        
        self.resblk_2 = ResBlk(32, 64)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        self.dropout_2 = nn.Dropout(0.1)
        
        self.resblk_3 = ResBlk(64, 128)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout_3 = nn.Dropout(0.1)
        
        self.resblk_4 = ResBlk(128, 256)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout_4 = nn.Dropout(0.1)
        
        self.resblk_5 = ResBlk(256, 256)
        self.maxpool_5 = nn.MaxPool2d(kernel_size=(4, 1))
        self.dropout_5 = nn.Dropout(0.1)

    def forward(self, x):
        out = x
        
        out = self.resblk_1(out)
        out = self.maxpool_1(out)
        out = self.dropout_1(out)
        
        out = self.resblk_2(out)
        out = self.maxpool_2(out)
        out = self.dropout_2(out)
        
        out = self.resblk_3(out)
        out = self.maxpool_3(out)
        out = self.dropout_3(out)
        
        out = self.resblk_4(out)
        out = self.maxpool_4(out)
        out = self.dropout_4(out)
        
        out = self.resblk_5(out)
        out = self.maxpool_5(out)
        out = self.dropout_5(out)
        
        out = out.squeeze(2)
        out = out.transpose(1, 2)

        return out


class Encoder(nn.Module):
    """
    input: [batch_size, 32, 256]
    output: [batch_size, 32, 128]
    """

    def __init__(self, num_rnn_layers=2, rnn_hidden_size=128, dropout=0.5):
        super(Encoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.gru = nn.GRU(256, rnn_hidden_size, num_rnn_layers,
                          batch_first=True,
                          dropout=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size))
        if USE_CUDA:
            h0 = h0.cuda()
        out, hidden = self.gru(x, h0)

        return out


class HybirdDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_rnn_layers=2, dropout=0.5):
        super(HybirdDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_rnn_layers = num_rnn_layers

        self.attn = DotProductAttentionLayer()
        self.gru = nn.GRU(hidden_size, hidden_size,
                          num_rnn_layers, batch_first=True,
                          dropout=dropout)

        self.wc = nn.Linear(2 * hidden_size, hidden_size)

        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward_train(self, encoder_outputs, max_len, y):
        batch_size = encoder_outputs.size(0)
        last_hidden = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size))
        if USE_CUDA:
            last_hidden = last_hidden.cuda()

        input = y[:, :max_len - 1]  # [batch, max_len-1]
        embed_input = self.embedding(input)  # [batch, max_len-1, 128]
        query, _ = self.gru(embed_input, last_hidden)  # [batch, max_len-1, 128]
        key = encoder_outputs  # [batch, 32, 128]
        value = encoder_outputs  # [batch, 32, 128]

        weighted_context = self.attn(query, key, value)  # [batch, max_len-1, 128]
        output = self.tanh(self.wc(torch.cat((query, weighted_context), 2)))  # [batch, max_len-1, 128]

        return output

    def forward_step(self, input, last_hidden, encoder_outputs):
        embed_input = self.embedding(input)
        output, hidden = self.gru(embed_input.unsqueeze(1), last_hidden)
        output = output.squeeze(1)

        query = output.unsqueeze(1)  # [batch, 1, 128]
        key = encoder_outputs  # [batch, 32, 128]
        value = encoder_outputs  # [batch, 32, 128]

        weighted_context = self.attn(query, key, value).squeeze(1)
        output = self.tanh(self.wc(torch.cat((output, weighted_context), 1)))
        return output, hidden

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()
        
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(ch_out)
        
        self.ch_out = ch_out
        self.ch_in = ch_in

        if ch_out != ch_in:
            self.extra_conv = nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=(1, 1))
            self.extra_bn = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.ch_out != self.ch_in:
            x = self.extra_conv(x)
            x = self.extra_bn(x)
        out = x + out

        return out


class PosConv2DLayer(nn.Module):
    """
    2D convolution with positional encoding.
    
    Enhances standard convolution by adding learnable position-dependent offsets.
    
    Attributes:
        conv: Standard convolution layer
        Wpos_y, Wpos_x: Learnable weights for positional maps
        bpos: Positional bias
        bn: Batch normalization
        relu: ReLU activation
    
    Methods:
        forward(x, pos_map_y, pos_map_x, return_intermediates=False):
            Process input with positional encoding.
            Returns full tensor or intermediates if requested.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.Wpos_y = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.Wpos_x = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.bpos = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.out_channels = out_channels # Store out_channels

    def forward(self, x, pos_map_y, pos_map_x, return_intermediates=False): # Added flag
        S_org = self.conv(x)
        output_height, output_width = S_org.shape[2], S_org.shape[3]
        current_pos_map_y = F.interpolate(pos_map_y, size=(output_height, output_width), mode='bilinear', align_corners=False)
        current_pos_map_x = F.interpolate(pos_map_x, size=(output_height, output_width), mode='bilinear', align_corners=False)
        positional_offset = (self.Wpos_y * current_pos_map_y +
                             self.Wpos_x * current_pos_map_x +
                             self.bpos)
        S_exp = S_org + positional_offset
        out = self.relu(self.bn(S_exp)) # Apply BN before ReLU

        if return_intermediates:
            return S_org, positional_offset, S_exp, out
        else:
            return out

class DotProductAttentionLayer(nn.Module):
    def __init__(self):
        super(DotProductAttentionLayer, self).__init__()

    def forward(self, query, key, value):
        logits = torch.matmul(query, key.permute(0, 2, 1))  # [len, 256]*[256, 32]=[len, 32]

        alpha = F.softmax(logits, dim=-1)
        weighted_context = torch.matmul(alpha, value)  # [len, 32] * [32, 256]=[len, 256]

        return weighted_context

def adaptive_pool(img_tensor):
    """Extract min, max, and mean color statistics using adaptive pooling."""
    max_pool = torch.nn.functional.adaptive_max_pool2d(img_tensor, (1, 1))
    min_pool = torch.nn.functional.adaptive_max_pool2d(img_tensor, (1, 1))
    #min_pool = F.adaptive_min_pool2d(img_tensor, (1, 1))
    
    avg_pool = torch.nn.functional.adaptive_avg_pool2d(img_tensor, (1, 1))
    return torch.cat([max_pool, min_pool, avg_pool], dim=1)

class GlobalUniformColorShift(nn.Module):
    def __init__(self, augment_img = False):
        super(GlobalUniformColorShift, self).__init__()
        self.conv = nn.Conv2d(3 * 3, 3, kernel_size=3, padding=1)  # Predict color shift
        self.sigmoid = nn.Sigmoid()
        self.augment_img = augment_img
    
    def forward(self, img):
        if self.augment_img:
            img = img.unsqueeze(0)  # Add batch dim
        stats = adaptive_pool(img)  # Extract statistics
        color_offset = self.conv(stats)  # Predict shift
        shift = self.sigmoid(color_offset)  # Normalize shifts
        lower = 1 - shift
        upper = 1 + shift
        
        # Sample new weights from uniform distribution
        w = torch.rand_like(img)
        w_scaled = lower + (upper - lower) * w
        
        return img * w_scaled

class GlobalVariationalColorShift(nn.Module):
    def __init__(self, augment_img = False):
        super(GlobalVariationalColorShift, self).__init__()
        self.conv_lower = nn.Conv2d(3 * 3, 3, kernel_size=3, padding=1, bias=False)
        self.conv_upper = nn.Conv2d(3 * 3, 3, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.augment_img = augment_img
        nn.init.xavier_uniform_(self.conv_lower.weight)
        nn.init.xavier_uniform_(self.conv_upper.weight)

    def forward(self, img):
        if self.augment_img:
            img = img.unsqueeze(0)
        stats = adaptive_pool(img)
        lower = self.sigmoid(self.conv_lower(stats))  # (B, C, 1, 1)
        upper = self.sigmoid(self.conv_upper(stats))  # (B, C, 1, 1)

        # Reshape lower and upper to match img dimensions (B, C, H, W)
        lower = lower.expand_as(img)
        upper = upper.expand_as(img)

        # Generate weight matrix for color shift
        weight = torch.rand_like(img) * (upper - lower) + lower
        transformed_img = img * weight

        return transformed_img

class SpatialVariationalColorShift(nn.Module):
    def __init__(self, kernel_size=4, dilation=2, dropout=0.3, augment_img = False):
        super(SpatialVariationalColorShift, self).__init__()
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout2d(dropout)

        # Dilated convolutions for feature extraction
        self.dilated_conv = nn.Conv2d(3, 3*3, kernel_size=kernel_size, dilation=2, padding=((kernel_size - 1) * dilation) // 2, bias=False)
        
        # Convolutions to predict upper & lower color shift bounds
        self.conv_lower = nn.Conv2d(3 * 3, 3, kernel_size=3, padding=1, bias=False)
        self.conv_upper = nn.Conv2d(3 * 3, 3, kernel_size=3, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        
        self.augment_img = augment_img

        # Initialize weights
        nn.init.xavier_uniform_(self.conv_lower.weight)
        nn.init.xavier_uniform_(self.conv_upper.weight)
        nn.init.xavier_uniform_(self.dilated_conv.weight)

    def forward(self, img):
        """
        img: (B, C, H, W) - Batch of images (batch size, 3 channels, height, width)
        """
        if self.augment_img:
            img = img.unsqueeze(0)  # Add batch dim
        # Extract deeper features using dilated convolution
        dilated_features = self.dilated_conv(img)  # (B, C, H, W)
        dilated_features = self.dropout(dilated_features)  # Apply dropout to prevent overfitting

        # Compute statistics from dilated feature maps
        lower = self.sigmoid(self.conv_lower(dilated_features))  # (B, C, H, W)
        upper = self.sigmoid(self.conv_upper(dilated_features))  # (B, C, H, W)
        # Reshape lower and upper to match img dimensions (B, C, H, W)
        lower = lower.expand_as(img)
        upper = upper.expand_as(img)
        # Generate weight matrix for color shift
        weight = torch.rand_like(img) * (upper - lower) + lower
        transformed_img = img * weight  # Apply color shift

        return transformed_img