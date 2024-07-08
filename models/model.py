import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
from einops import rearrange
from models.utils import TriangularCausalMask, ProbMask

from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        print("enc:", x_enc.shape, "dec:", x_dec.shape)
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack_e2e(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(InformerStack_e2e, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        stacks = list(range(e_layers, 2, -1))  # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 SR_rate=6,interpolate_f='linear',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.SR_rate = SR_rate
        self.interpolate_f=interpolate_f

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        stacks = list(range(e_layers, 2, -1))  # you can customize here  # [4 3]
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)  # 层正则化
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)  # 线性层只改变最后一个维度

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class RNNUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super(RNNUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden):
        # if len(x.shape) > 3:
        # print('x shape must be 3')

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)
        output, cur_hidden = self.rnn(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden


class RNN(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = RNNUnit(features, input_size, hidden_size, num_layers=self.num_layers)

    def train_pro(self, x, pred_len, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden)
            else:
                output, prev_hidden = self.model(output, prev_hidden)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def forward(self, x, pred_len, device):
        return self.train_pro(x, pred_len, device)


class GRUUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super(GRUUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden):
        # if len(x.shape) > 3:
        # print('x shape must be 3')

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)
        output, cur_hidden = self.gru(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden


class GRU(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = GRUUnit(features, input_size, hidden_size, num_layers=self.num_layers)

    def train_pro(self, x, pred_len, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden)
            else:
                output, prev_hidden = self.model(output, prev_hidden)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def forward(self, x, pred_len, device):
        return self.train_pro(x, pred_len, device)


class LSTMUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super(LSTMUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden, prev_cell):
        # if len(x.shape) > 3:
        # print('x shape must be 3')

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)
        output, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell))
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden, cur_cell


class LSTM(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = LSTMUnit(features, input_size, hidden_size, num_layers=self.num_layers)

    def train_pro(self, x, pred_len, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(),
                                                            prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output, prev_hidden, prev_cell)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def forward(self, x, pred_len, device):
        return self.train_pro(x, pred_len, device)



class Autoencoder(nn.Module):
    def __init__(self, n_filters=[2, 8, 16, 32, 64, 128, 256, 512], filter_sizes=[3, 3, 3, 3, 3, 3, 3, 3]):
        super(Autoencoder, self).__init__()
        self.postprocess = nn.Conv1d(16, 4, 3, 1, 1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Building the encoder
        for i in range(len(n_filters) - 1):
            self.encoder.append(nn.Conv2d(n_filters[i], n_filters[i + 1], filter_sizes[i], stride=1, padding=1))
            # Initialize weights
            nn.init.uniform_(self.encoder[-1].weight, -1.0 / math.sqrt(n_filters[i]), 1.0 / math.sqrt(n_filters[i]))
            nn.init.constant_(self.encoder[-1].bias, 0)

        # Building the decoder
        n_filters.reverse()
        filter_sizes.reverse()
        for i in range(len(n_filters) - 1):
            self.decoder.append(nn.Conv2d(n_filters[i], n_filters[i + 1], filter_sizes[i], stride=1, padding=1))
            # Initialize weights
            nn.init.uniform_(self.decoder[-1].weight, -1.0 / math.sqrt(n_filters[i]), 1.0 / math.sqrt(n_filters[i]))
            nn.init.constant_(self.decoder[-1].bias, 0)

    def forward(self, x):
        x = rearrange(x, 'b l (s i) -> b i l s', i=2)
        # Encoder
        for layer in self.encoder:
            x = F.tanh(layer(x))

        # Decoder
        for layer in self.decoder:
            x = F.tanh(layer(x))
        # Postprocessor
        x = rearrange(x, 'b i l s -> b l (s i)', i=2)
        x = self.postprocess(x)

        return x

