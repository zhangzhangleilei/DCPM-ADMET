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




# if __name__ == "__main__":
# ms.set_context(mode=mode, device_target=device)
# profiler = ms.Profiler(output_path='../../summary_dir/profiler_data', profile_memory=True)
ms.set_context(mode=mode, device_target=device, device_id=int(os.environ["DEVICE_ID"]), max_device_memory="30GB")
# ms.reset_auto_parallel_context()
# 并行
# init()
# ms.set_seed(1)


def test_xlnet():
#     rank_id = get_rank() # shard_id
#     rank_size = get_group_size() # num_shards

#     ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)

    stage = "predict"
    is_check_point = False
    # processed_data_path = preprocess(data_path, n_workers, is_large_data=True, sep=' ', usecols=[1])
    processed_data_path = '/home/ma-user/work/work/data/Pubchem_processed_droped.csv'

    model = xlnet.XLNetPredictModel(n_token=16,
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

    
    param_dict = ms.load_checkpoint('../../checkpoints/test2card1/finetune_model/xlnet-5_115.ckpt') 
    param_not_load = ms.load_param_into_net(model, param_dict)
    
    dataset = GraphDataset(processed_data_path, max_len, mask_alpha, mask_beta, perm_size, batch_size=4,
                            stage=stage, nrows=32, header=None)
    
    # data_generator = dataset.get_data(rank_size, rank_id)
    data_generator = dataset.get_data()

    network = Model(model)
    for batched_data in data_generator.create_dict_iterator():
        output, hidden_states, attentions = network.predict(batched_data['input_k'], batched_data['adjoin_matrix'], batched_data['cls_pos'])
        print(output)
        print(hidden_states[-1])
        print(attentions[-1])
        break

test_xlnet()
