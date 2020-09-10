import json
import pickle
import torch

from .train import TrainingExperiment

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc


class PruningExperiment(TrainingExperiment):

    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 compression,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False,
                 save_freq=10,
                 module_list=[],
                 source=None,
                 target=None,
                 inputs=None,
                 outputs=None,
                 poisoned_root=None,
                 override_fraction=False,
                 ULP_data=None):

        super(PruningExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, resume, resume_optim, save_freq, source, target, poisoned_root)
        self.add_params(strategy=strategy, compression=compression)

        self.apply_pruning(strategy, compression, module_list, override_fraction, ULP_data)

        self.path = path
        self.save_freq = save_freq

    def apply_pruning(self, strategy, compression, module_list, override_fraction, ULP_data):
        constructor = getattr(strategies, strategy)
        # x, y = next(iter(self.train_dl))
        # Pass ULPs and target all zeros (clean w.r.t ULPs)
        x = ULP_data[0]
        y = torch.LongTensor([0]*10)
        # self.to_device()              # CALCULATE GRADIENTS on CPU?  ANIRUDDHA
        self.pruning = constructor(self.model, x, y, compression=compression, module_list=module_list, override_fraction=override_fraction, ULP_data=ULP_data)
        self.pruning.apply()
        printc("Masked model", color='GREEN')

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)

        self.save_metrics()

        # if self.pruning.compression > 1:
        self.run_epochs()                           # Finetune without compression. Aniruddha

    def save_metrics(self):
        self.metrics = self.pruning_metrics()
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        printc(json.dumps(self.metrics, indent=4), color='GRASS')
        summary = self.pruning.summary()
        summary_path = self.path / 'masks_summary.csv'
        summary.to_csv(summary_path)
        print(summary)

    def pruning_metrics(self):

        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        # Accuracy
        loss, acc1, acc5 = self.run_epoch(False, -1)
        # self.log_epoch(-1)

        metrics['loss'] = loss
        metrics['val_acc1'] = acc1
        metrics['val_acc5'] = acc5

        # ANIRUDDHA
        loss, acc1, acc5 = self.run_epoch_n(True, -1)
        # self.log_epoch(-1)

        metrics['ASR_loss'] = loss
        metrics['ASR_acc1'] = acc1
        metrics['ASR_acc5'] = acc5

        loss, acc1, acc5 = self.run_epoch_n(False, -1)
        self.log_epoch(-1)

        metrics['backdoor_loss'] = loss
        metrics['backdoor_acc1'] = acc1
        metrics['backdoor_acc5'] = acc5
        

        return metrics
