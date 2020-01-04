# encoding=utf-8
import torch
import torch.nn as nn
import numpy as np
from .encode import encode
from .decode import decode


class seq2seq(nn.Module):
    """
    seq2seq模型
    """

    def __init__(self, rnn_type, vocab_size, input_size=20, hidden_size=128, num_layers=1, dropout=0, use_attn=1):
        super(seq2seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.enc = encode(rnn_type, vocab_size, input_size, hidden_size, num_layers, dropout)
        self.dec = decode(rnn_type, vocab_size, hidden_size, num_layers, dropout, use_attn)

    def forward(self, encode_val, decode_val, input_length):
        encode_output, encode_hidden = self.enc(encode_val, input_length)
        out, _, _ = self.dec(decode_val, encode_hidden, encode_output)
        return out


if __name__ == '__main__':
    pass