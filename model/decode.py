# encoding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from .attention import attention


class decode(nn.Module):
    """
    seq2seq解码器
    """

    def __init__(self, rnn_type, vocab_size, hidden_size, num_layers=1, dropout=0, use_attn=1):
        super(decode, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attn = use_attn
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.attn = attention(hidden_size)
        self.classes = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, decode_val, encode_hidden, encode_output):
        batch_size = decode_val.size(0)
        output_size = decode_val.size(1)
        decode_val = self.embedding(decode_val)
        out, hidden = self.rnn(decode_val, encode_hidden)
        # attention机制
        attn_out = None
        if self.use_attn:
            out, attn_out = self.attn(out, encode_output)
        out = self.classes(out.contiguous().view(-1, self.hidden_size))
        # out = F.softmax(self.classes(out.contiguous().view(-1, self.hidden_size)), dim=1)
        out = out.view(batch_size, output_size, -1)
        return out, attn_out, hidden


if __name__ == '__main__':
    pass