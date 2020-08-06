from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER, CNN_ENCODER_RNN_DECODER, \
    BERT_CNN_ENCODER_RNN_DECODER, BERT_RNN_ENCODER
from model import D_NET64, D_NET128, D_NET256

from miscc.losses import words_loss, cycle_generator_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
import trainer

import random
import pprint
import datetime
import dateutil.tz
import argparse

import torchvision.transforms as transforms
from trainer import condGANTrainer

text_encoder_path = '/mnt/tera/code/python/GANS/cycle-image-gan/output/Training_birds_STREAM_2020_07_22_10_13_59/Model/text_encoder550.pth'
net_G_path = '/mnt/tera/code/python/GANS/cycle-image-gan/output/Training_birds_cycle_2020_07_23_11_07_20/Model/netG_epoch_500.pth'


# ################# Text to image task############################ #

############# CYCLE GAN ##########
class CycleGANTester(condGANTrainer):

    def generate_fake_im(self, data_dic):

        global text_encoder_path, net_G_path

        # Build and load the generator
        #####################################
        ## load the encoder                 #
        #####################################
        text_encoder = \
            BERT_RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(text_encoder_path,
                        map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)

        print('Loaded text encoder from:', text_encoder_path)
        text_encoder.eval()
        text_encoder = text_encoder.cuda()


        netG = G_NET()
        ######################################
        ## load the generator                #
        ######################################

        state_dict = \
                        torch.load(net_G_path, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load Generator from: ', net_G_path)
        s_tmp = net_G_path[:net_G_path.rfind('.pth')]


        netG.cuda()
        netG.eval()
        for key in data_dic:
            save_dir = '%s/%s' % (s_tmp, key)
            mkdir_p(save_dir)
            captions, cap_lens, sorted_indices = data_dic[key]

            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM
            captions = Variable(torch.from_numpy(captions), volatile=True)
            cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

            captions = captions.cuda()
            cap_lens = cap_lens.cuda()
            for i in range(1):  # 16
                noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                noise = noise.cuda()
                #######################################################
                # (1) Extract text embeddings
                ######################################################
                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                mask = (captions == 0)
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                
                return fake_imgs, attention_maps

    
    def gen_example(self, data_dic):

        global text_encoder_path, net_G_path

        if net_G_path == '':
            print('Error: the path for models is not found!')
        else:
            # Build and load the generator
            #####################################
            ## load the encoders                #
            #####################################
            text_encoder = \
                BERT_RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(text_encoder_path,
                            map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)

            print('Loaded text encoder from:', text_encoder_path)
            text_encoder.eval()
            text_encoder = text_encoder.cuda()

            # the path to save generated images
            netG = G_NET()
            ######################################
            ## load the generator                #
            ######################################

            state_dict = \
                            torch.load(net_G_path, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', net_G_path)
            s_tmp = net_G_path[:net_G_path.rfind('.pth')]


            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)

