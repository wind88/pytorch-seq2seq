# encoding=utf-8
import sys
import os
sys.path.append(os.path.abspath('/Users/wind/workspaces/ai_study/contest/deep_study/seq2seq/'))
import argparse
from train.train import trainer
from eval.predict import predictor

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='pyTorch chat bot')
    parse.add_argument('-t', '--train', default=1, type=int)
    parse.add_argument('-g', '--msg', default='有多远呢', type=str)
    parse.add_argument('-n', '--bucket-num', default=3, type=int)
    parse.add_argument('-s', '--bucket-size', default=10, type=int)
    parse.add_argument('-e', '--epochs', default=2000, type=int)
    parse.add_argument('-b', '--batch-size', default=128, type=int)
    parse.add_argument('-l', '--lr', default=0.01, type=float)
    parse.add_argument('-m', '--momentum', default=0.01, type=float)
    parse.add_argument('-w', '--weight-decay', default=5e-4, type=float)
    parse.add_argument('-p', '--rnn-type', default='lstm', type=str)
    parse.add_argument('-is', '--input-size', default=20, type=int)
    parse.add_argument('-hs', '--hidden-size', default=128, type=int)
    parse.add_argument('-r', '--num-layers', default=2, type=int)
    parse.add_argument('-d', '--dropout', default=0.5, type=float)
    parse.add_argument('-u', '--use-attn', default=1, type=int)
    args = parse.parse_args()

    if args.train:
        train = trainer(args.bucket_num, args.bucket_size, args.epochs, args.batch_size, args.lr, args.momentum,
                        args.weight_decay, args.rnn_type, args.input_size, args.hidden_size, args.num_layers, args.dropout, args.use_attn)
        train.run()
    else:
        pred = predictor(args.bucket_size, args.rnn_type, args.input_size, args.hidden_size, args.num_layers, args.dropout, args.use_attn)
        print(pred.predict(args.msg))