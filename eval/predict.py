# encoding=utf-8
import torch
import jieba
import math
import sys
import os
sys.path.append(os.path.abspath('/Users/wind/workspaces/ai_study/contest/deep_study/seq2seq/'))
from dataset.dataset import textData
from model.seq2seq import seq2seq


class predictor:
    """
    predict class
    """

    def __init__(self, bucket_size=10, rnn_type='lstm', input_size=20, hidden_size=128, num_layers=1, dropout=0, use_attn=1):
        self.bucket_size = bucket_size
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attn = use_attn
        self.best_acc = 0
        self.acc_arr = []
        self.loss_arr = []

    def getBucketSize(self, word_length):
        """
        get bucket size
        :return:
        """
        buckets_size = []
        bucket_num = math.ceil(word_length/self.bucket_size)
        for bucket in range(bucket_num):
            start = bucket * self.bucket_size
            end = start + self.bucket_size
            buckets_size.append([start, end])
        return buckets_size, end

    def getInputs(self, words, text):
        """
        get predict inputs data
        :param words:
        :param text:
        :return:
        """
        bucket = len(text.bucket_size) - 1
        encode_ids = []
        for word in words:
            if word not in text.words_to_id:
                return '', '', '对不起，我还未理解你的意思', '', ''
            encode_ids.append(text.words_to_id[word])
        decode_ids = [text.SOS for _ in range(text.bucket_size[bucket][1] + 1)]
        target_ids = [text.SOS for _ in range(text.bucket_size[bucket][1])]
        encode_ids, target_ids, input_pad_length, target_pad_length = text.paddingToZero(bucket, encode_ids, target_ids)
        return torch.LongTensor([encode_ids]), torch.LongTensor([decode_ids]), torch.LongTensor([target_ids]), [input_pad_length], [target_pad_length]

    def splitWords(self, line=''):
        """
        split words
        :param line:
        :return:
        """
        words = list(line)
        # words = jieba.cut(line)
        # words = [w for w in words]
        return words

    def predict(self, x):
        """
        predict result
        :param x:
        :return:
        """
        words = self.splitWords(x)
        bucket_size, max_words = self.getBucketSize(len(words))
        text = textData(max_words)
        text.bucket_size = bucket_size
        try:
            text.words_to_id = torch.load(text.words_file)
            model_state = torch.load(text.best_model_file)
        except FileNotFoundError:
            raise Exception('please train model then predict~')

        model = seq2seq(self.rnn_type, len(text.words_to_id), self.input_size, self.hidden_size, self.num_layers, self.dropout, self.use_attn)
        model.load_state_dict(model_state['state_dict'])

        encode_inputs, decode_inputs, targets, inputs_length, targets_length = self.getInputs(words, text)
        if isinstance(targets, str):
            return targets
        return self.decodeResult(model, encode_inputs, decode_inputs, inputs_length, text)

    def decodeResult(self, model, encode_inputs, decode_inputs, inputs_length, text):
        """
        decode reply
        :param model:
        :param encode_inputs:
        :param decode_inputs:
        :param inputs_length:
        :param text:
        :return:
        """
        id_to_words = dict(zip(text.words_to_id.values(), text.words_to_id.keys()))
        out = model(encode_inputs, decode_inputs, inputs_length)
        pred = torch.max(out[0], 1)[1]
        replay = []
        for ids in pred:
            if ids == text.EOS or ids == text.PAD or ids == text.SOS:
                break
            replay.append(id_to_words[ids.item()])
        return ''.join(replay)


if __name__ == '__main__':
    pred = predictor()
    if len(sys.argv) >= 2:
        msg = sys.argv[1]
    else:
        msg = '人类是什么'
    print(pred.predict(msg))
