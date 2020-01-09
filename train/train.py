# encoding=utf-8
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('../'))
from dataset.dataset import bucketDataset, textData
from model.seq2seq import seq2seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trainer:
    """
    seq2seq模型训练
    """

    def __init__(self, bucket_num=3, bucket_size=10, epochs=2000, batch_size=64, lr=0.01, momentum=0.01,
                 weight_decay=5e-4, rnn_type='lstm', input_size=20, hidden_size=128, num_layers=1, dropout=0, use_attn=1):
        self.bucket_num = bucket_num
        self.bucket_size = bucket_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = False
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attn = use_attn
        self.best_acc = 0
        self.train_acc_arr, self.train_loss_arr, self.test_acc_arr, self.test_loss_arr = [], [], [], []
        self.bucket_size, self.max_words = self.getBucketSize()
        self.text = textData(self.max_words)
        self.text.bucket_size = self.bucket_size
        try:
            self.text.words_to_id = torch.load(self.text.words_file)
        except FileNotFoundError:
            self.text.readData()
            self.text.saveTrainData()
            self.text.saveWordToIdData()

    def getBucketSize(self):
        """
        获取分桶尺寸
        :return:
        """
        buckets_size = []
        for bucket in range(self.bucket_num):
            start = bucket * self.bucket_size
            end = start + self.bucket_size
            buckets_size.append([start, end])
        return buckets_size, end

    def run(self):
        """
        模型训练
        :return:
        """
        model = seq2seq(self.rnn_type, len(self.text.words_to_id), self.input_size, self.hidden_size, self.num_layers, self.dropout, self.use_attn)
        entropy_loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            train_acc, train_loss = self.train(model, entropy_loss, optimizer)
            print('step '+str(epoch+1)+':', train_acc, train_loss)
            self.train_acc_arr.append(train_acc)
            self.train_loss_arr.append(train_loss)
            self.saveModel(train_acc, epoch, model, optimizer)
        self.saveAccImg()

    def train(self, model, entropy_loss, optimizer):
        """
        训练
        :param model:
        :param entropy_loss:
        :param optimizer:
        :return:
        """
        total_acc, total_num, total_losses, losses_num = 0, 0, 0, 0
        for bucket in range(self.bucket_num):
            dataset = bucketDataset(True, bucket, self.bucket_size, self.max_words)
            if dataset.__len__() == 0:
                continue
            dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            bucket_acc, bucket_num, bucket_loss_val, bucket_loss_num = self.trainBucket(model, dataloader, entropy_loss, optimizer)
            total_acc += bucket_acc
            total_num += bucket_num
            total_losses += bucket_loss_val
            losses_num += bucket_loss_num
        return total_acc/total_num, total_losses/losses_num

    def trainBucket(self, model, dataloader, entropy_loss, optimizer):
        """
        分桶分批次训练数据
        :param model:
        :param dataloader:
        :param entropy_loss:
        :param optimizer:
        :return:
        """
        model.train()
        train_acc, train_num, loss_val, loss_num = 0, 0, 0, 0
        for encode_inputs, decode_inputs, targets, inputs_length, targets_length in dataloader:
            encode_inputs = encode_inputs.to(device)
            decode_inputs = decode_inputs.to(device)
            inputs_length = inputs_length.to(device)
            targets = targets.to(device)
            out = model(encode_inputs, decode_inputs, inputs_length)

            loss = 0
            for index, batch in enumerate(out):
                loss += entropy_loss(batch, targets[index])
                loss_val += loss.item()
                loss_num += 1

                pred = torch.max(batch, 1)[1]   # 计算预测结果
                train_acc += (pred == targets[index]).sum().item()
                train_num += len(targets[index])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(train_acc, train_num, loss_val, loss_num)
        return train_acc, train_num, loss_val, loss_num

    def saveModel(self, acc, epoch, model, optimizer):
        """
        保存模型
        :param acc:
        :param epoch:
        :param model:
        :param optimizer:
        :return:
        """
        is_best = acc > self.best_acc
        self.best_acc = max(acc, self.best_acc)
        torch.save({
            'epoch': epoch,
            'best_acc': self.best_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, self.text.resume_model_file)
        if is_best:
            shutil.copyfile(self.text.resume_model_file, self.text.best_model_file)

    def saveAccImg(self):
        """
        保存准确率图片
        :return:
        """
        plt.figure()
        plt.switch_backend('agg')
        plt.plot(range(len(self.train_acc_arr)), self.train_acc_arr, 'm-', label='train_acc')
        plt.savefig(self.text.data_path+'acc.jpg')


if __name__ == '__main__':
    tr = trainer()
    tr.run()