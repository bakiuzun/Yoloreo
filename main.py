"""
author Uzun Baki
-- u.officialdeveloper@gmail.com
"""

from model import *
from torch.utils.data import DataLoader
from dataset import CliffDataset
from trainer import YoloreoTrainer
import random
import string
import argparse
import wandb 
import math
import random 
import string

parser = argparse.ArgumentParser()
parser.add_argument("--num",type=int)
parser.add_argument("--no_label",type=int)
args = parser.parse_args()


model_config = {
    "arch": "yolov8.yaml",
    "checkpoint":"imported/yolov8m.pt"
}

def generate_random_string(length):return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

## model def
def get_model():
    model = Yoloreo(cfg=model_config["arch"])
    model.load_pretrained_weights(model_config["checkpoint"])
    model.nc = 1
    model.names = {0:'erosion'}
    return model


def main():
    overrides = {"name":f"stereo_no_label_{args.no_label}_fold_{args.num}_{generate_random_string(4)}"}
    val_set = f"last_fold/val_num_{args.num}_img_without_annot_100.csv"
    trainer = YoloreoTrainer(cfg="cfg.yaml",model=get_model(),
            train_path=f"last_fold/train_num_{args.num}_img_without_annot_{args.no_label}.csv",
            valid_path=val_set,
            overrides=overrides)
    trainer.train()


"""
sweep_config = {
        'method': 'random',
        'parameters': {
            'batch': {'values': [4,8,16]},
            'warmup_epochs':{'values':[2,3,4,5]},
            'warmup_momentum':{'max': 0.85, 'min':0.1,'distribution': 'uniform'},
            'warmup_decay':{'max': 0.0001, 'min':0.00005,'distribution': 'uniform'},
            'lrdebut': {'max': math.log(0.09), 'min':math.log(0.0005),'distribution': 'log_uniform'},
            'lrf': {'max': 0.70, 'min':  0.01,'distribution': 'uniform'},
            'momentum': {'max': 0.90, 'min':  0.60,'distribution': 'uniform'},
            'cls':{'max': 4.0, 'min':0.5,'distribution': 'uniform'},
            'dfl':{'max': 2.0, 'min':1.5,'distribution': 'uniform'},
            'optimizer':{'values':['SGD','AdamW']},
            'box':{'max': math.log(10), 'min':math.log(0.02),'distribution': 'log_uniform'}
        }
}
"""
sweep_config = {
        'method': 'random',
        'parameters': {
            'batch': {'values': [16]},
            'warmup_epochs':{'values':[2]},
            'warmup_momentum':{'values': [0.761227834027154]},
            'warmup_decay':{'values': [7.371504264026107e-05]},
            'lrdebut': {'values': [0.01450253096069692]  },
            'lrf': {'values':    [0.19237925186074303]  },
            'momentum': {'values':  [0.8277351685028354]},
            'cls':{'values':  [3.088690590733296]  },
            'dfl':{'values': [1.8068976367313925] },
            'optimizer':{'values':['SGD'] },
            'box':{'values':[2.2173349030196534]}
        }
}


sweep_id=wandb.sweep(sweep_config, project="YoloStereoAugV2")
wandb.agent(sweep_id=sweep_id, function=main, count=1)