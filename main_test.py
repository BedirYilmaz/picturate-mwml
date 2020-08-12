from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
import trainer

import tester

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def gen_example(wordtoix, algo, sentences = ['this bird is red with white and has a very short beak']):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    # filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    captions = []
    cap_lens = []
    for sent in sentences:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)
            continue

        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    key = sent
    data_dic[key] = [cap_array, cap_lens, sorted_indices]

    algo.generate_fake_images_with_incremental_noise(data_dic)
    # fake_im, _ = algo.generate_fake_im(data_dic)

    # fake_imt = np.transpose(fake_im[2].squeeze(0).data.cpu().numpy(), (1, 2, 0))

    # plt.imshow(fake_imt)
    # plt.show()

def gen_example_from_predefined_sentences(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        # filenames = f.read().decode('utf8').split('\n')
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)



def parse_args():
    parser = argparse.ArgumentParser(description='Generate images from text descriptions with Cycle-Image-GAN')
    parser.add_argument('--desc', type=str, default='this seagull has white wide wings and a large white chest and a long beak', help='manual seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # print(os.getcwd())
    # exit(0)
    args = parse_args()

    cfg_path = "cfg/bird_cycle.yaml"

    cfg_from_file(cfg_path)
    cfg.GPU_ID = 0
    cfg.DATA_DIR = 'data/birds'


    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # sentence = input('please type in the text description for the bird to be generated\n')

    sentence = args.desc

    # Define models and go to evaluate
    tester_ = getattr(tester, cfg.TRAIN.TESTER)
    algo = tester_("bird_gen_test", dataloader, dataset.n_words, dataset.ixtoword)

    '''generate images from pre-extracted embeddings'''
    gen_example(dataset.wordtoix, algo, [sentence])  # generate images for customized captions
