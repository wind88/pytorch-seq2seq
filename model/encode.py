# encoding=utf-8
import torch
import torch.nn as nn


class encode(nn.Module):
    """
    seq2seq编码器
    """

    def __init__(self, rnn_type, vocab_size, input_size, hidden_size, num_layers=1, dropout=0):
        super(encode, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, encode_val, input_length):
        encode_val = self.embedding(encode_val)
        packed = torch.nn.utils.rnn.pack_padded_sequence(encode_val, input_length, batch_first=True)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


if __name__ == '__main__':
    en = encode('lstm', 16, 3, 5)
    inputs = torch.LongTensor([[1, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0], [1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]])
    input_length = [7, 8]
    inputs, input_length = sorted((inputs, input_length), key=lambda x:x[1], reverse=True)
    out, hidd = en(inputs, input_length)
    print(out[0].shape, hidd[0].shape)
