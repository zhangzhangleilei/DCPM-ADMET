import sys
import os

import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, SummaryCollector

from conf import *
from dataset import GraphDataset
from data_utils import preprocess
import xlnet

import argparse
os.environ['HCCL_CONNECT_TIMEOUT' ]= "1800"
parser = argparse.ArgumentParser()

parser.add_argument('--fintune data path', type=str, default="", help='fintune data path')
parser.add_argument('--model path',  type=str, default="", help='model path'')
parser.add_argument('--save model path', type=str, default="", help='save model path')
parser.add_argument('--checkpoint', type=str, default="", help='checkpoint path')
args = parser.parse_args()

ms.set_context(mode=mode, device_target=device, device_id=int(os.environ["DEVICE_ID"]), max_device_memory="30GB")
def test_xlnet():

    stage = "finetune"
    is_check_point = False
    processed_data_path = args.fintune data path

    pretrain_model = xlnet.XLNetPreTrainModel(n_token=16,
                                     n_layer=8,
                                     n_head=8,
                                     d_head=64,
                                     d_inner=2048,
                                     d_model=512,
                                     dropout=drop,
                                     dropatt=drop,
                                     attn_type="bi",
                                     bi_data=bi_data,
                                     clamp_len=-1,
                                     same_length=False)

    
    param_dict = ms.load_checkpoint(args.model path) 
    ms.load_param_into_net(pretrain_model, param_dict)
    ms.save_checkpoint(pretrain_model.xlnet_layer, args.save model path)

    model = xlnet.XLNetForClassificationModel(n_token=16,
                                     n_layer=8,
                                     n_head=8,
                                     d_head=64,
                                     d_inner=2048,
                                     d_model=512,
                                     dropout=drop,
                                     dropatt=drop,
                                     attn_type="bi",
                                     bi_data=bi_data,
                                     clamp_len=-1,
                                     same_length=False)
    param_dict = ms.load_checkpoint('../../checkpoints/test2card1/pretrained_xlnetlayer/xlnetlayer.ckpt') 
    ms.load_param_into_net(model, param_dict)

    
    dataset = GraphDataset(processed_data_path, max_len, mask_alpha, mask_beta, perm_size, batch_size=16,
                            stage=stage, header=None)
    
    # data_generator = dataset.get_data(rank_size, rank_id)
    data_generator = dataset.get_data()
    train_dataset, test_dataset = data_generator.split([0.9, 0.1], randomize=False)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

    network = Model(model, loss_fn=None, optimizer=optimizer, amp_level="O3")

    time_monitor = TimeMonitor()
    loss_cb = LossMonitor(per_print_times=5)
    callbacks = [loss_cb, time_monitor]
    config = CheckpointConfig(save_checkpoint_steps=20, saved_network=model)
    ckpoint_cb = ModelCheckpoint(prefix='xlnet', directory=args.checkpoint, config=config)
    callbacks.append(ckpoint_cb)
    network.train(num_epoch, train_dataset, callbacks=callbacks, dataset_sink_mode=False)

test_xlnet()
