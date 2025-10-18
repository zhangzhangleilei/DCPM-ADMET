import sys
import os

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, SummaryCollector
from mindspore.train import ROC, auc
from dataset import SmilesDataset
from data_utils import preprocess, dataset_split, Log
import xlnet
import pandas as pd
import argparse
import pickle



parser = argparse.ArgumentParser()
parser.add_argument('--device_id', type=int, default=0, help='device id')
parser.add_argument('--ckpt_name', type=str, help='ckpt name')
parser.add_argument('--task', type=str, help='task name')
parser.add_argument('--save_path', type=str, help='feature save path')

args = parser.parse_args()
ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=args.device_id, max_device_memory="30GB")

task = args.task
stage = 'finetune'
processed_data_path = f'./data/MoleculeNet/{task}.csv'
dataset = SmilesDataset(processed_data_path, max_len=301, mask_alpha=3, mask_beta=2, perm_size=30, batch_size=32,
                        stage=stage)
dataset = dataset.get_data()
model = xlnet.XLNetLayer(n_token=99,
                        n_layer=12,
                        n_head=16,
                        d_head=64,
                        d_inner=4096,
                        d_model=1024,
                        dropout=0.1,
                        dropatt=0.1,
                        attn_type="bi",
                        bi_data=False,
                        clamp_len=-1,
                        same_length=False,
                        param_init_type=ms.float32)
ckpt_path = f'./checkpoints/{args.ckpt_name}'
param_dict = ms.load_checkpoint(ckpt_path)
param_not_load = ms.load_param_into_net(model, param_dict)
model.set_train(False)
labels = []
all_outputs = []
for batched_data in dataset.create_dict_iterator():
    output, _, _ = model(batched_data['input_k'], input_mask=batched_data['input_mask'])
    #output=(bsz, len, d_model), hidden_states和attentions是一个列表，取最后一个
    output = output[:, 0, :].asnumpy()
    batch_label = batched_data['label'].asnumpy() # bsz, 1
    labels.append(batch_label)
    all_outputs.append(output)
all_outputs = np.concatenate(all_outputs, axis=0) # [lendf, hidden_dim]
labels = np.concatenate(labels, axis=0)

pickle.dump((all_outputs, labels), open(f'./features/{args.save_path}/{task}.pkl', 'wb'))
print(f'task {task} finished')