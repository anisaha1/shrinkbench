from shrinkbench.experiment import PruningExperiment
import os
import datasets
import models
import torch
import pickle
import glob
from torch.utils.data import DataLoader

os.environ['DATAPATH'] = '/nfs3/data/aniruddha:/nfs3/data/aniruddha/ULP/tiny_imagenet/Attacked_Data/Triggers_11_20'
os.environ['WEIGHTSPATH'] = '/nfs3/data/aniruddha/ULP/tiny_imagenet/poisoned_models/Triggers_11_20'

# for strategy in ['GlobalMagGrad']:
#     for  c in [4]:
#         exp = PruningExperiment(dataset='MNIST',
#                                 model='MnistNet',
#                                 strategy=strategy,
#                                 compression=c,
#                                 train_kwargs={'epochs':10})
#         exp.run()

# # check dataset and dataloader
# dataset = 'TinyImageNet'
# constructor = getattr(datasets, dataset)
# train_dataset = constructor(train=True)
# val_dataset = constructor(train=False)
# train_dl = DataLoader(train_dataset, shuffle=True)
# val_dl = DataLoader(val_dataset, shuffle=False)
# dataset = 'customFolder'
# constructor = getattr(datasets, dataset)
# poisoned_dataset = constructor(path='/nfs3/data/aniruddha/ULP/tiny_imagenet/Attacked_Data/Triggers_11_20/backdoor0000_s0000_t0044_trigger_15', target=44)
# backdoor_dataset = constructor(path='/nfs3/data/aniruddha/ULP/tiny_imagenet/Attacked_Data/Triggers_11_20/backdoor0000_s0000_t0044_trigger_15', target=0)


# check model
# cnn = models.resnet18_mod()

# print("Done")

# module_list=['conv1',
#             'layer1.0.conv1',
#             'layer1.0.conv2',
#             'layer2.0.conv1',
#             'layer2.0.conv2',
#             'layer3.0.conv1',
#             'layer3.0.conv2',
#             'layer4.0.conv1',
#             'layer4.0.conv2']

module_list= ['layer4.0.conv1']

# load ULPs and classifier
# ANIRUDDHA - load ULPs to calculate parameter gradients
N=10
ULPs, W, b=pickle.load(open('/nfs3/code/aniruddha/ULP/tiny_imagenet/Results/universal_image_ResNet18_mod_N{}.pkl'.format(N),'rb'))

strategy = 'LayerMagGrad'
c = 1                               # DUMMY COMPRESSION VALUE. OVERRIDE FRACTION TRUE. ANIRUDDHA
poisoned_model_list = sorted(glob.glob('/nfs3/data/aniruddha/ULP/tiny_imagenet/poisoned_models/Triggers_11_20/*.pt'))
meta = pickle.load(open('/nfs3/data/aniruddha/ULP/Webpage/tiny-imagenet/poisoned_models_Triggers_11_20_meta.pkl', 'rb'))
NUM_MODELS = 2

for i in range(NUM_MODELS):
    exp = PruningExperiment(dataset='TinyImageNet', 
                            model='resnet18_mod',
                            pretrained=False,
                            resume=poisoned_model_list[i],
                            strategy=strategy,
                            compression=c,
                            train_kwargs={'epochs':1},
                            source=meta[i][1],
                            target=meta[i][2],
                            module_list=module_list,
                            poisoned_root=os.path.basename(meta[i][3]),
                            override_fraction=True,
                            ULP_data=[ULPs, W, b]
                            )
    exp.run()