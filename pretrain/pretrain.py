import sys
import os
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, SummaryCollector
import mindspore as ms
from dataset import GraphDataset
from data_utils import LossCallBack, download_data
import xlnet
# if __name__ == "__main__":
# ms.set_context(mode=mode, device_target=device)
# profiler = ms.Profiler(output_path='../../summary_dir/profiler_data', profile_memory=True)

import argparse
os.environ['HCCL_CONNECT_TIMEOUT' ]= "1800"
parser = argparse.ArgumentParser()

parser.add_argument('--data_url', type=str, default="", help='path where the dataset is saved')
parser.add_argument('--device_num', type=int, default=1, help='device num')
parser.add_argument('--save_url', type=str, default="", help='path where the model is saved')
parser.add_argument('--is_check_point', type=int, default=0, help='whether train stopped')
parser.add_argument('--check_point_path', type=str, default="", help='path where the checkpoint')
parser.add_argument('--has_trained_epoch', type=int, default=0, help='epoch has trained')
parser.add_argument('--has_trained_step', type=int, default=0, help='step has trained')
args = parser.parse_args()

ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', max_device_memory="30GB")
# ms.reset_auto_parallel_context()
# 并行

init()
rank_size = get_group_size()
rank_id = get_rank()

ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)
cache_url = '/cache/Data/'
download_data(src_data_url=args.data_url, tgt_data_path=cache_url, rank=rank_id)


stage = "pretrain"
processed_data_path = cache_url + 'Pubchem_processed_droped.csv'

model = xlnet.XLNetPreTrainModel(n_token=16,
                                    n_layer=6,
                                    n_head=8,
                                    d_head=64,
                                    d_inner=2048,
                                    d_model=512,
                                    dropout=0.1,
                                    dropatt=0.1,
                                    attn_type="bi",
                                    bi_data=False,
                                    clamp_len=-1,
                                    same_length=False)

total_params= 0
for param in model.trainable_params():
    total_params += np.prod(param.shape)
print(f"Trainable parameters: {total_params}")


dataset = GraphDataset(processed_data_path, max_len=301, mask_alpha=3, mask_beta=2, perm_size=30, batch_size=128,
                        stage=stage, header=None)

data_generator = dataset.get_data(rank_size, rank_id)
# data_generator = dataset.get_data()
step_per_epoch = data_generator.get_dataset_size()
print(f'dataset has {data_generator.get_dataset_size()} batches')

lr = nn.cosine_decay_lr(min_lr=float(0),
                        max_lr=0.00005,
                        total_step=10 * step_per_epoch,
                        step_per_epoch=step_per_epoch,
                        decay_epoch=10)
optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

network = Model(model, loss_fn=None, optimizer=optimizer, amp_level="O3")

# def callbacks
if args.is_check_point:
    check_point_path = args.check_point_path   # add in config
    has_trained_epoch = args.has_trained_epoch
    has_trained_step = args.has_trained_step
    param_dict = ms.load_checkpoint(check_point_path)
    ms.load_param_into_net(model, param_dict)
else:
    has_trained_epoch = 0
    has_trained_step = 0


time_monitor = TimeMonitor()
loss_cb = LossCallBack(dataset_size=step_per_epoch, local_rank=rank_id, device_num=args.device_num, per_print_times=5, 
                       has_trained_epoch=has_trained_epoch, has_trained_step=has_trained_step, is_last_stage=True)
callbacks = [loss_cb, time_monitor]
config = CheckpointConfig(save_checkpoint_steps=step_per_epoch//5, saved_network=model)
ckpoint_cb = ModelCheckpoint(prefix='xlnet', directory=args.save_url+'device{}'.format(rank_id), config=config)
callbacks.append(ckpoint_cb)

# summary_collector = SummaryCollector(summary_dir='/home/ma-user/work/work/summary_dir', collect_freq=500)
# callbacks.append(summary_collector)

network.build(data_generator, epoch=10)
network.train(10, data_generator, callbacks=callbacks, dataset_sink_mode=False)
    # profiler.analyse()

