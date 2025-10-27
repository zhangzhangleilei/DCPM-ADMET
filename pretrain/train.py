import sys
import os
import argparse
from datetime import datetime
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train.callback import Callback

from datasetrnn import Smiles2InchiFeatureDataset
from rnn import GRUSeq2SeqWithFeatures, Seq2SeqFeatureTrainNet
from utils import InchiTokenizer, SMILESTokenizer, calc_trainable_params

from dataset import GraphDataset  
from data_utils import LossCallBack, download_data  
import xlnet 

def run_first_model(args):
    
    print("=" * 50)
    print("开始运行第一套模型")
    print("=" * 50)

    
    data_name = args.data
    ms_mode = args.ms_mode
    device_id = args.device_id
    device = args.device
    worker = args.worker
    savedir = args.checkout
    testdata = args.testdata
    testpkl = args.testpkl

    batch_size = args.batch
    epoch = args.epoch_num
    lr = args.lr
    weight_decay = 0.0

    d_embed = args.d_embed
    d_hidden = args.d_hidden
    d_enc_out = args.d_enc_out
    num_layers = args.num_layers
    enc_dropout = args.enc_dropout
    dec_dropout = args.dec_dropout
    enc_activation = args.enc_activation
    dec_activation = args.dec_activation
    max_length = args.max_length
    
    TASK_NAME = f"{data_name}_seq2seq"
    # SAVE_DIR = f"../checkpoints/{TASK_NAME}"
    SAVE_DIR = savedir + TASK_NAME
    
  
    ms.set_context(
        mode=ms_mode,
        device_id=device_id,
        device_target=device,
        max_call_depth=5000
    )

   
    enc_tokenizer = SMILESTokenizer()
    dec_tokenizer = InchiTokenizer()
    enc_vocab_size = len(enc_tokenizer.word_table)
    dec_vocab_size = len(dec_tokenizer.word_table)
    bos_token_id = enc_tokenizer.vocab2index[enc_tokenizer.bos]
    pad_token_id = dec_tokenizer.vocab2index[dec_tokenizer.pad]

    
    train_ds = Smiles2InchiFeatureDataset(
        max_len=max_length,
        enc_tokenizer=enc_tokenizer,
        dec_tokenizer=dec_tokenizer
    )
    train_ds = (
        train_ds.load_csv(testdata, header=None)
        .load_feat_pkl(testpkl)
        .make_dataset(shuffle=True)
        .fast_tokenize_transform(num_parallel_workers=worker, python_multiprocessing=True)
        .project()
        .batch(batch_size, num_parallel_workers=worker)
    )
    assert train_ds.dataset is not None, "第一套模型数据集加载失败"

    
    print("第一套模型超参数：")
    print({
        "lr": lr, "epoch": epoch, "batch_size": batch_size,
        "d_embed": d_embed, "d_hidden": d_hidden, "d_enc_out": d_enc_out,
        "num_layers": num_layers, "enc_vocab_size": enc_vocab_size,
        "dec_vocab_size": dec_vocab_size, "max_length": max_length
    })

    step_per_epoch = train_ds.dataset.get_dataset_size()
    print(f"第一套模型数据加载完成：总样本数 {len(train_ds)}, 每 epoch 步数 {step_per_epoch}")

 
    model = GRUSeq2SeqWithFeatures(
        d_embed, d_hidden, d_enc_out, num_layers,
        enc_dropout, dec_dropout, enc_activation, dec_activation,
        enc_vocab_size, dec_vocab_size, feature_size=9
    )
    net_with_loss = Seq2SeqFeatureTrainNet(model, ignore_index=pad_token_id)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    train_net = nn.TrainOneStepCell(net_with_loss, optimizer)


    params_num = calc_trainable_params(train_net)
    print(f"第一套模型可训练参数：{params_num}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 第一套模型开始训练")

    for cur_epoch in range(epoch):
        train_net.set_train(True)
        for i, batch in enumerate(train_ds.dataset.create_tuple_iterator()):
            enc_inp, inp_len, dec_inp, tgt, feat = batch
            loss = train_net(enc_inp, inp_len, dec_inp, tgt, feat)
            print(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                f"第一套模型 Epoch {cur_epoch + 1}/{epoch} Step {i + 1}/{step_per_epoch} Loss: {loss}"
            )

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 第一套模型训练完成")


def run_second_model(args):

    print("\n" + "=" * 50)
    print("开始运行第二套模型：XLNet预训练模型")
    print("=" * 50)

    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    data_url = args.data_url
    device_num = args.device_num
    save_url = args.save_url
    is_check_point = args.is_check_point
    check_point_path = args.check_point_path
    has_trained_epoch = args.has_trained_epoch
    has_trained_step = args.has_trained_step

    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target='Ascend',
        max_device_memory="30GB"
    )

    init()
    rank_size = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                 parameter_broadcast=True)
    cache_url = '/cache/Data/'
    download_data(src_data_url=args.data_url, tgt_data_path=cache_url, rank=rank_id)

    model = xlnet.XLNetPreTrainModel(
        n_token=16,
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
        same_length=False
    )

    total_params = 0
    for param in model.trainable_params():
        total_params += np.prod(param.shape)
    print(f"第二套模型可训练参数：{total_params}")

    stage = "pretrain"
    processed_data_path = os.path.join(cache_url, 'Pubchem_processed_droped_10000.csv')
    dataset = GraphDataset(
        processed_data_path,
        max_len=301,
        mask_alpha=3,
        mask_beta=2,
        perm_size=30,
        batch_size=128,
        stage=stage,
        header=None
    )
    data_generator = dataset.get_data(rank_size, rank_id)
    step_per_epoch = data_generator.get_dataset_size()
    print(f"第二套模型数据集加载完成：每 epoch 步数 {step_per_epoch}")

    lr = nn.cosine_decay_lr(
        min_lr=0.0,
        max_lr=5e-5,
        total_step=10 * step_per_epoch,
        step_per_epoch=step_per_epoch,
        decay_epoch=10
    )
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
    network = Model(model, loss_fn=None, optimizer=optimizer, amp_level="O3")
    if is_check_point and check_point_path:
        param_dict = ms.load_checkpoint(check_point_path)
        ms.load_param_into_net(model, param_dict)
        print(f"已加载检查点：{check_point_path}")


    time_monitor = TimeMonitor()
    loss_cb = LossCallBack(
        dataset_size=step_per_epoch,
        local_rank=rank_id,
        device_num=device_num,
        per_print_times=5,
        has_trained_epoch=has_trained_epoch,
        has_trained_step=has_trained_step,
        is_last_stage=True
    )
    callbacks = [loss_cb, time_monitor]

    ckpt_dir = os.path.join(save_url, f'device{rank_id}')
    os.makedirs(ckpt_dir, exist_ok=True)
    config = CheckpointConfig(save_checkpoint_steps=step_per_epoch // 5, saved_network=model)
    ckpoint_cb = ModelCheckpoint(prefix='xlnet', directory=ckpt_dir, config=config)
    callbacks.append(ckpoint_cb)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 第二套模型开始训练")
    network.build(data_generator, epoch=10)
    network.train(10, data_generator, callbacks=callbacks, dataset_sink_mode=False)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 第二套模型训练完成")


def main():
    parser = argparse.ArgumentParser(description="训练代码")
    parser.add_argument("--data", type=str, default="debug")
    parser.add_argument("--checkout", type=str, default="")
    parser.add_argument("--testdata", type=str, default="")
    parser.add_argument("--testpkl", type=str, default="")

    parser.add_argument("--ms_mode", type=int, default=0)
    parser.add_argument("--device", type=str, default="Ascend")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--worker", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epoch_num", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_length", type=int, default=350)

    parser.add_argument("--d_embed", type=int, default=32)
    parser.add_argument("--d_hidden", type=int, default=256)
    parser.add_argument("--d_enc_out", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--enc_dropout", type=float, default=0.2)
    parser.add_argument("--dec_dropout", type=float, default=0.2)
    parser.add_argument("--enc_activation", type=str, default=None)
    parser.add_argument("--dec_activation", type=str, default=None)

    parser.add_argument('--data_url', type=str, default="", help='path where the dataset is saved')
    parser.add_argument('--device_num', type=int, default=1, help='device num')
    parser.add_argument('--save_url', type=str, default="", help='path where the model is saved')
    parser.add_argument('--is_check_point', type=int, default=0, help='whether train stopped')
    parser.add_argument('--check_point_path', type=str, default="", help='path where the checkpoint')
    parser.add_argument('--has_trained_epoch', type=int, default=0, help='epoch has trained')
    parser.add_argument('--has_trained_step', type=int, default=0, help='step has trained')

    args = parser.parse_args()

    run_first_model(args)
    run_second_model(args)


if __name__ == "__main__":
    main()