import pathlib
import time
import pickle
import sys
import json
import numpy as np

import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
from tqdm import tqdm

from .base import Experiment
from .. import datasets
from .. import models
from ..metrics import correct
from ..models.head import mark_classifier
from ..util import printc, OnlineStats


class TrainingExperiment(Experiment):

    default_dl_kwargs = {'batch_size': 128,
                         'pin_memory': False,
                         'num_workers': 8
                         }

    default_train_kwargs = {'optim': 'SGD',
                            'epochs': 30,
                            'lr': 1e-3,
                            }

    def __init__(self,
                 dataset,
                 model,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=False,
                 resume=None,
                 resume_optim=False,
                 save_freq=10,
                 source=None,
                 target=None,
                 trigger=None,
                 poisoned_root=None
                 ):

        # Default children kwargs
        super(TrainingExperiment, self).__init__(seed)
        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}
        train_kwargs = {**self.default_train_kwargs, **train_kwargs}

        params = locals()
        params['dl_kwargs'] = dl_kwargs
        params['train_kwargs'] = train_kwargs

        # HACK to not serialize array trigger ANIRUDDHA
        del params['trigger']

        self.add_params(**params)
        # Save params
        self.source = source
        self.target = target
        # self.trigger = trigger
        self.poisoned_root = poisoned_root

        self.build_dataloader(dataset, trigger, **dl_kwargs)

        self.build_model(model, pretrained, resume)

        self.build_train(resume_optim=resume_optim, **train_kwargs)

        self.path = path
        self.save_freq = save_freq
        
    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)
        self.run_epochs()

    def build_dataloader(self, dataset, trigger=None, **dl_kwargs):
        if dataset == 'TinyImageNet':
            constructor = getattr(datasets, dataset)
            self.train_dataset = constructor(train=True)
            self.val_dataset = constructor(train=False)
            constructor = getattr(datasets, 'customFolder')
            self.poisoned_dataset = constructor(train=False, dataset=self.poisoned_root, target=self.target)
            self.backdoor_dataset = constructor(train=False, dataset=self.poisoned_root, target=self.source)
            self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_kwargs)
            self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dl_kwargs)
            # This loads only source class images patched with trigger
            self.poisoned_dl = DataLoader(self.poisoned_dataset, shuffle=False, **dl_kwargs)
            self.backdoor_dl = DataLoader(self.backdoor_dataset, shuffle=False, **dl_kwargs)
        elif dataset == 'CIFAR10_ULP':
            # check dataset and dataloader
            dataset = 'CIFAR10_ULP'
            constructor = getattr(datasets, dataset)
            train_dataset = constructor(train=True, dataset='/nfs1/code/aniruddha/poisoning_defense/Codes/GTSRB/Data/CIFAR10/')
            val_dataset = constructor(train=False, dataset='/nfs1/code/aniruddha/poisoning_defense/Codes/GTSRB/Data/CIFAR10/')
            self.train_dl = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
            self.val_dl = DataLoader(val_dataset, shuffle=False, **dl_kwargs)
            dataset = 'custom_CIFAR10_Dataset'
            constructor = getattr(datasets, dataset)
            # generate clean and poisoned data
            # # This loads only source class images patched with trigger
            # X_val, y_val = pickle.load(open('/nfs1/code/aniruddha/poisoning_defense/Codes/GTSRB/Data/CIFAR10/val_heq.p', 'rb'))
            # X_poisoned, y_poisoned, _, _ = datasets.generate_poisoned_data(X_val.copy(), y_val.copy(), self.source, self.target, trigger)
            # poisoned_dataset = constructor(X_poisoned, y_poisoned)

            # X_clean, y_clean, _ = datasets.generate_clean_data(X_val.copy(), y_val.copy(), self.source)
            # backdoor_dataset = constructor(X_clean, y_clean)
            # self.poisoned_dl = DataLoader(poisoned_dataset, shuffle=False, **dl_kwargs)
            # self.backdoor_dl = DataLoader(backdoor_dataset, shuffle=False, **dl_kwargs)

            # This loads all sources !=target class images patched with trigger
            # This loads only source class images patched with trigger
            X_stack = np.empty((0,32,32,3), np.uint8)
            y_stack = np.empty((0,), np.uint8)

            labels = np.arange(10)
            source_labels = np.concatenate([labels[:self.target], labels[self.target+1:]])
            X_val, y_val = pickle.load(open('/nfs1/code/aniruddha/poisoning_defense/Codes/GTSRB/Data/CIFAR10/val_heq.p', 'rb'))

            for source in source_labels:
                X_poisoned, y_poisoned, _, _ = datasets.generate_poisoned_data(X_val.copy(), y_val.copy(), source, self.target, trigger)
                X_stack = np.append(X_stack, X_poisoned, axis=0)
                y_stack = np.append(y_stack, y_poisoned, axis=0)
            poisoned_dataset = constructor(X_stack, y_stack)

            X_stack = np.empty((0,32,32,3), np.uint8)
            y_stack = np.empty((0,), np.uint8)
            for source in source_labels:
                X_clean, y_clean, _ = datasets.generate_clean_data(X_val.copy(), y_val.copy(), source)
                X_stack = np.append(X_stack, X_poisoned, axis=0)
                y_stack = np.append(y_stack, y_poisoned, axis=0)
            backdoor_dataset = constructor(X_stack, y_stack)
                
            self.poisoned_dl = DataLoader(poisoned_dataset, shuffle=False, **dl_kwargs)
            self.backdoor_dl = DataLoader(backdoor_dataset, shuffle=False, **dl_kwargs)
        else:
            print("Dataset not implemented for backdoor pruning.")
            sys.exit()

    def build_model(self, model, pretrained=True, resume=None):
        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)(pretrained=pretrained)

            elif hasattr(torchvision.models, model):
                # https://pytorch.org/docs/stable/torchvision/models.html
                model = getattr(torchvision.models, model)(pretrained=pretrained)
                mark_classifier(model)  # add is_classifier attribute
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")

        self.model = model

        if resume is not None:
            self.resume = pathlib.Path(resume)
            assert self.resume.exists(), "Resume path does not exist"
            previous = torch.load(self.resume)
            self.model.load_state_dict(previous)                            # Aniruddha

    def build_train(self, optim, epochs, resume_optim=False, **optim_kwargs):
        default_optim_kwargs = {
            'SGD': {'momentum': 0.9, 'nesterov': True, 'lr': 1e-3},
            'Adam': {'momentum': 0.9, 'betas': (.9, .99), 'lr': 1e-4}
        }

        self.epochs = epochs

        # Optim
        if isinstance(optim, str):
            constructor = getattr(torch.optim, optim)
            if optim in default_optim_kwargs:
                optim_kwargs = {**default_optim_kwargs[optim], **optim_kwargs}
            optim = constructor(self.model.parameters(), **optim_kwargs)

        self.optim = optim

        if resume_optim:
            assert hasattr(self, "resume"), "Resume must be given for resume_optim"
            previous = torch.load(self.resume)
            self.optim.load_state_dict(previous['optim_state_dict'])

        # Assume classification experiment
        self.loss_func = nn.CrossEntropyLoss()

    def to_device(self):
        # Torch CUDA config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="ORANGE")
        self.model.to(self.device)
        cudnn.benchmark = True   # For fast training.

    def checkpoint(self):
        checkpoint_path = self.path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        epoch = self.log_epoch_n
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, checkpoint_path / f'checkpoint-{epoch}.pt')

    def run_epochs(self):

        since = time.time()
        try:
            for epoch in range(self.epochs):
                printc(f"Start epoch {epoch}", color='YELLOW')
                self.train(epoch)
                self.eval(epoch)
                self.eval_p(epoch)
                self.eval_b(epoch)
                # Checkpoint epochs
                # TODO Model checkpointing based on best val loss/acc
                if epoch % self.save_freq == 0:
                    self.checkpoint()
                # TODO Early stopping
                # TODO ReduceLR on plateau?
                self.log(timestamp=time.time()-since)
                self.log_epoch(epoch)


        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')

    def run_epoch(self, train, epoch=0):
        if train:
            self.model.train()
            prefix = 'train'
            dl = self.train_dl
        else:
            prefix = 'val'
            dl = self.val_dl
            self.model.eval()

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
                if train:
                    loss.backward()

                    self.optim.step()
                    self.optim.zero_grad()

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / dl.batch_size)
                acc1.add(c1 / dl.batch_size)
                acc5.add(c5 / dl.batch_size)

                epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean, top5=acc5.mean)

        self.log(**{
            f'{prefix}_loss': total_loss.mean,
            f'{prefix}_acc1': acc1.mean,
            f'{prefix}_acc5': acc5.mean,
        })

        return total_loss.mean, acc1.mean, acc5.mean

    def run_epoch_n(self, p, epoch=0):
        if p:
            self.model.eval()
            prefix = 'ASR'
            dl = self.poisoned_dl
        else:
            self.model.eval()
            prefix = 'backdoor'
            dl = self.backdoor_dl
            

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")

        with torch.set_grad_enabled(False):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
                if False:
                    loss.backward()

                    self.optim.step()
                    self.optim.zero_grad()

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / dl.batch_size)
                acc1.add(c1 / dl.batch_size)
                acc5.add(c5 / dl.batch_size)

                epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean, top5=acc5.mean)

        self.log(**{
            f'{prefix}_loss': total_loss.mean,
            f'{prefix}_acc1': acc1.mean,
            f'{prefix}_acc5': acc5.mean,
        })

        return total_loss.mean, acc1.mean, acc5.mean

    def train(self, epoch=0):
        return self.run_epoch(True, epoch)

    def eval(self, epoch=0):
        return self.run_epoch(False, epoch)

    def eval_p(self, epoch=0):
        return self.run_epoch_n(True, epoch)

    def eval_b(self, epoch=0):
        return self.run_epoch_n(False, epoch)

    @property
    def train_metrics(self):
        return ['epoch', 'timestamp',
                'train_loss', 'train_acc1', 'train_acc5',
                'val_loss', 'val_acc1', 'val_acc5', 
                'ASR_loss', 'ASR_acc1', 'ASR_acc5', 
                'backdoor_loss', 'backdoor_acc1', 'backdoor_acc5'
                ]


    def __repr__(self):
        if not isinstance(self.params['model'], str) and isinstance(self.params['model'], torch.nn.Module):
            self.params['model'] = self.params['model'].__module__

        assert isinstance(self.params['model'], str), f"\nUnexpected model inputs: {self.params['model']}"
        return json.dumps(self.params, indent=4)
