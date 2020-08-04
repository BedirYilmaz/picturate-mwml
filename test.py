#####################################
# the paths 
#####################################

image_encoder_path = '/mnt/tera/code/python/GANS/cycle-image-gan/output/birds_STREAM_2020_07_22_10_13_59/Model/image_encoder550.pth'
text_encoder_path = '/mnt/tera/code/python/GANS/cycle-image-gan/output/birds_STREAM_2020_07_22_10_13_59/Model/text_encoder550.pth'

net_G_path = '/mnt/tera/code/python/GANS/cycle-image-gan/output/birds_cycle_2020_07_23_11_07_20/Model/netG_epoch_500.pth'

net_D0_path = '/mnt/tera/code/python/GANS/cycle-image-gan/output/birds_cycle_2020_07_23_11_07_20/Model/netD0.pth'
net_D1_path = '/mnt/tera/code/python/GANS/cycle-image-gan/output/birds_cycle_2020_07_23_11_07_20/Model/netD1.pth'
net_D2_path = '/mnt/tera/code/python/GANS/cycle-image-gan/output/birds_cycle_2020_07_23_11_07_20/Model/netD2.pth'

#####################################
## load the encoders                #
#####################################
image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
state_dict = \
    torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
image_encoder.load_state_dict(state_dict)]

print('Loaded image encoder from:', img_encoder_path)
image_encoder.eval()

text_encoder = \
    RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
state_dict = \
    torch.load(text_encoder_path,
                map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)

print('Loaded text encoder from:', text_encoder_path)
text_encoder.eval()

######################################
## load the discriminators           #
######################################
netsD = []
netDpaths = [net_D0_path, net_D1_path, net_D2_path]

netsD.append(D_NET64())
netsD.append(D_NET128())
netsD.append(D_NET256())
            
for i in range(len(netsD)):
    state_dict = \
        torch.load(netDpaths[i], map_location=lambda storage, loc: storage)
    netsD[i].load_state_dict(state_dict)

######################################
## load the generator                #
######################################

state_dict = \
                torch.load(net_G_path, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', net_G_path)




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

# Define models and go to train/evaluate
trainer_ = getattr(trainer, cfg.TRAIN.TRAINER)
algo = trainer_(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

start_t = time.time()
if cfg.TRAIN.FLAG:
    algo.train()
else:
    '''generate images from pre-extracted embeddings'''
    if cfg.B_VALIDATION:
        algo.sampling(split_dir)  # generate images for the whole valid dataset
    else:
        gen_example(dataset.wordtoix, algo)  # generate images for customized captions
end_t = time.time()
print('Total time for training:', end_t - start_t)