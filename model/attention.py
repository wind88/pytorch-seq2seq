# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class attention(nn.Module):
    """
    attention机制
    x = decode_output*encode_output \\
    attn = exp(x_i) / sum_j exp(x_j) \\
    output = \tanh(w * (attn * context) + b * output)
    """
    def __init__(self, dim):
        super(attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, decode_output, encode_output):
        batch_size = decode_output.size(0)
        hidden_size = decode_output.size(2)
        input_size = encode_output.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(decode_output, encode_output.transpose(1, 2))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, encode_output)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, decode_output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


if __name__ == '__main__':
    from encode import encode
    from decode import decode
    atten = attention(64)
    inputs = torch.LongTensor([[1, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0], [1, 3, 4, 6, 0, 0, 0, 0, 0, 0, 0]])
    targets = torch.LongTensor([[4, 6, 7, 5, 2, 0, 0, 0, 0, 0, 0], [6, 8, 14, 9, 2, 0, 0, 0, 0, 0, 0]])
    input_length = [7, 7]

    en_model = encode('lstm', 20, 20, 64)
    de_model = decode('lstm', 20, 20, 64)
    encode_outputs, encode_hidden = en_model(inputs, input_length)
    decode_outputs, decode_hidden = de_model(targets, encode_hidden, encode_outputs)
    print(encode_outputs.shape, decode_outputs.shape)
    outputs, attn = atten(decode_outputs, encode_outputs)
    print(outputs.shape)
