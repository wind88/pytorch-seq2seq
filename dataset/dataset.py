# encoding=utf-8
import jieba
import torch
import numpy as np
import torch.utils.data as data


class bucketDataset(data.Dataset):
    """
    分桶数据集
    """

    def __init__(self, is_train=True, bucket=0, bucket_size=[], max_words=20):
        self.bucket = bucket
        self.text = textData(max_words)
        self.text.bucket_size = bucket_size
        try:
            if is_train:
                self.data = torch.load(self.text.bucket_train_file.format(self.text.bucket_size[self.bucket][0], self.text.bucket_size[self.bucket][1]))
            else:
                self.data = torch.load(self.text.bucket_test_file.format(self.text.bucket_size[self.bucket][0], self.text.bucket_size[self.bucket][1]))
            self.text.words_to_id = torch.load(self.text.words_file)
        except FileNotFoundError:
            raise Exception('train bucket file not exists')

    def __getitem__(self, item):
        encode_ids = [self.text.words_to_id[word] for word in self.data[item][0]]
        decode_ids = [self.text.SOS for _ in range(self.text.bucket_size[self.bucket][1] + 1)]
        target_ids = [self.text.words_to_id[word] for word in self.data[item][1]]
        encode_ids, target_ids, input_pad_length, target_pad_length = self.text.paddingToZero(self.bucket, encode_ids, target_ids)
        return torch.LongTensor(encode_ids), torch.LongTensor(decode_ids), torch.LongTensor(target_ids), input_pad_length, target_pad_length

    def __len__(self):
        return len(self.data)


class textData:
    """
    文本数据处理
    """

    def __init__(self, max_words=20):
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.data_path = '/Users/wind/workspaces/ai_study/contest/deep_study/seq2seq/data/'
        # self.words_to_id = {'PAD': self.PAD, 'SOS': self.SOS, 'EOS': self.EOS}
        self.words_to_id = {}
        self.bucket_size = []
        self.data = []
        self.train_file = self.data_path+'train.txt'
        self.test_file = self.data_path+'test.txt'
        self.bucket_train_file = self.data_path+'bucket_train_{}_{}.patch'
        self.bucket_test_file = self.data_path+'bucket_test_{}_{}.patch'
        self.words_file = self.data_path+'words_to_id.patch'
        self.resume_model_file = self.data_path+'resume.pkl'
        self.best_model_file = self.data_path+'best.pkl'
        self.max_words = max_words

    def readData(self, is_train=True):
        """
        读取训练数据
        :return:
        """
        file = self.train_file if is_train else self.test_file
        with open(file, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        iters = iter(lines)
        data = [[l, next(iters)] for l in iters]
        self.data = self.getBucketData(data)

    def saveTrainData(self):
        """
        保存分桶训练数据
        :return:
        """
        for index, buckets in enumerate(self.data):
            torch.save(buckets, self.bucket_train_file.format(self.bucket_size[index][0], self.bucket_size[index][1]))

    def saveTestData(self):
        """
        保存分桶测试数据
        :return:
        """
        for index, buckets in enumerate(self.data):
            torch.save(buckets, self.bucket_test_file.format(self.bucket_size[index][0], self.bucket_size[index][1]))

    def saveWordToIdData(self):
        """
        保存文字和id的映射关系
        :return:
        """
        # 排序每个字
        self.words_to_id['PAD'] = 1000003
        self.words_to_id['SOS'] = 1000002
        self.words_to_id['EOS'] = 1000001
        words_to_id = sorted(self.words_to_id.items(), key=lambda x: (-x[1], x[0]), reverse=False)
        words_to_id, _ = list(zip(*words_to_id))
        self.words_to_id = dict(zip(words_to_id, range(len(words_to_id))))
        torch.save(self.words_to_id, self.words_file)

    def getWordsToId(self):
        """
        获取文字和id的映射关系
        :return:
        """
        if len(self.words_to_id):
            return self.words_to_id
        else:
            return torch.load(self.words_file)

    def getBucketData(self, data):
        """
        过滤过长的句子并进行分桶
        :param data:
        :return:
        """
        new_data = [[] for _ in range(len(self.bucket_size))]
        for dt in data:
            in_words = self.splitWords(dt[0].strip())
            out_words = self.splitWords(dt[1].strip())
            in_words_length = len(in_words)
            out_words_length = len(out_words)
            # 过滤过长的句子
            if in_words_length > self.max_words or out_words_length > self.max_words:
                continue
            # 分桶
            for index, bucket in enumerate(self.bucket_size):
                if in_words_length <= bucket[1] and out_words_length <= bucket[1]:
                    new_data[index].append([in_words, out_words])
                    break
            # 词语转ID
            self.wordsToId(in_words)
            self.wordsToId(out_words)
        # 排序每个桶的句子
        for index, buckets in enumerate(new_data):
            new_data[index] = sorted(buckets, key=lambda x:len(x[0]), reverse=False)
        return new_data

    def splitWords(self, line=''):
        """
        分词
        :param line:
        :return:
        """
        words = line.split('/')
        # words = list(line)
        # words = jieba.cut(line)
        # words = [w for w in words]
        return words

    def wordsToId(self, words):
        """
        词语转ID
        :param words:
        :return:
        """
        for word in words:
            if word not in self.words_to_id:
                self.words_to_id[word] = 1
            else:
                self.words_to_id[word] += 1

    def paddingToZero(self, bucket, input_ids, target_ids):
        """
        zero padding
        :param bucket:
        :param input_ids:
        :param target_ids:
        :return:
        """
        word_length = self.bucket_size[bucket][1] + 1
        input_length = len(input_ids)
        target_length = len(target_ids)

        input_zero_ids = np.zeros(word_length, dtype=np.long)
        target_zero_ids = np.zeros(word_length, dtype=np.long)
        input_zero_ids[: input_length] = input_ids[:input_length]
        target_zero_ids[: target_length] = target_ids[:target_length]

        input_zero_ids[input_length] = self.EOS
        target_zero_ids[target_length] = self.EOS

        return input_zero_ids, target_zero_ids, word_length - input_length, word_length - target_length


if __name__ == '__main__':
    dataset = bucketDataset()
    print(dataset.__getitem__(0))
    print(dataset.__len__())
