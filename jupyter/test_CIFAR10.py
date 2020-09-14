from shrinkbench.experiment import PruningExperiment
import os
import datasets
import models
import torch
import pickle
import glob
from torch.utils.data import DataLoader

# # check dataset and dataloader
# dataset = 'CIFAR10_ULP'
# constructor = getattr(datasets, dataset)
# train_dataset = constructor(train=True, dataset='/nfs1/code/aniruddha/poisoning_defense/Codes/GTSRB/Data/CIFAR10/')
# val_dataset = constructor(train=False, dataset='/nfs1/code/aniruddha/poisoning_defense/Codes/GTSRB/Data/CIFAR10/')
# train_dl = DataLoader(train_dataset, shuffle=True)
# val_dl = DataLoader(val_dataset, shuffle=False)
# dataset = 'custom_CIFAR10_Dataset'
# constructor = getattr(datasets, dataset)


# # poisoned meta
# meta = pickle.load(open('/nfs3/data/aniruddha/ULP/Webpage/CIFAR-10/poisoned_models_test_meta.pkl', 'rb'))
# # generate clean and poisoned data
# X_val, y_val = pickle.load(open('/nfs1/code/aniruddha/poisoning_defense/Codes/GTSRB/Data/CIFAR10/val_heq.p', 'rb'))
# source = meta[0][2]
# target = meta[0][3]
# trigger = meta[0][1]
# X_poisoned, y_poisoned, _, _ = datasets.generate_poisoned_data(X_val.copy(), y_val.copy(), source, target, trigger)
# poisoned_dataset = constructor(X_poisoned, y_poisoned)

# X_clean, y_clean, _ = datasets.generate_clean_data(X_val.copy(), y_val.copy(), source)
# backdoor_dataset = constructor(X_clean, y_clean)

# # check model
# init_num_filters=64
# inter_fc_dim=384
# nofclasses=10 #CIFAR10

# cnn=models.CNN_classifier(init_num_filters=init_num_filters,
#                          inter_fc_dim=inter_fc_dim,nofclasses=nofclasses,
#                          nofchannels=3,use_stn=False)


module_list= ['fc.0']

# load ULPs and classifier
# ANIRUDDHA - load ULPs to calculate parameter gradients
N=10
ULPs, W, b=pickle.load(open('/nfs1/code/aniruddha/poisoning_defense/Codes/GTSRB/Data/CIFAR10_best_universal_image_same_dist_N{}.pkl'.format(N),'rb'))

strategy = 'LayerMagGrad'
c = 1                               # DUMMY COMPRESSION VALUE. OVERRIDE FRACTION TRUE. ANIRUDDHA
meta = pickle.load(open('/nfs3/data/aniruddha/ULP/Webpage/CIFAR-10/poisoned_models_test_meta.pkl', 'rb'))
poisoned_model_list = sorted(glob.glob('/nfs3/data/aniruddha/ULP/Webpage/CIFAR-10/poisoned_models_test/*'))
NUM_MODELS = 2

for i in range(NUM_MODELS):
    exp = PruningExperiment(dataset='CIFAR10_ULP', 
                            model='CIFAR10_CNN',
                            pretrained=False,
                            resume=poisoned_model_list[i],
                            strategy=strategy,
                            compression=c,
                            train_kwargs={'epochs':10},
                            source=meta[i][2],
                            target=int(meta[i][3]),         # the target in meta is in numpy int64 format for some reason. int cast to avoid JSON serialization error.
                            trigger=meta[i][1],
                            module_list=module_list,
                            poisoned_root=None,
                            override_fraction=True,
                            ULP_data=[ULPs, W, b]
                            )
    exp.run()