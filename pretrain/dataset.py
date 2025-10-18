from typing import Dict, List, Tuple

import mindspore.dataset as ds
import numpy as np
import pandas as pd
from numpy import ndarray
from mindspore import ops
import mindspore as ms
import mindspore.numpy as mnp

from conf import logger
from data_utils import Tokenizer, GraphTokenizer


class GraphDataset:
    def __init__(self, path: str, max_len: int, mask_alpha: int, mask_beta: int,
                 perm_size: int, batch_size: int, stage: str = "pretrain",
                 tokenizer: type(Tokenizer) = GraphTokenizer(),
                 **kwargs) -> None:
        self.df = None
        self.max_len = max_len
        self.seq_len = max_len - 1
        self.tokenizer = tokenizer
        self.special_tokens_id = self.tokenizer.special_tokens_id
        self.perm_size = perm_size
        self.batch_size = batch_size
        self.num_predict = int(max_len * mask_beta / mask_alpha)
        self.stage = stage

        assert stage in ["pretrain", "finetune", "predict"], "Wrong stage attribute."

        self.df = pd.read_csv(path, **kwargs)
        self.dataset = self.df.iloc[:, 0].tolist()
        # tokenized_data = self.df.iloc[:, 0].parallel_apply(self.tokenizer.tokenize)
        # logger.info("Graph tokenization done.")
        # filtered_idx = tokenized_data.map(lambda pair: len(pair[0])) < max_len
        # self.dataset = tokenized_data[filtered_idx].tolist()
        self.size = len(self.dataset)
        # logger.info(f"Filtered {self.df.shape[0] - self.size} items longer than {max_len}.")

        if stage == 'finetune':
            self.target = self.df.iloc[:, 1].tolist()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        # smiles = self.dataset[idx]
        # tokenized_data = self.tokenizer.tokenize(smiles)
        # atoms, adjoin_matrix = self.__pad_data(tokenized_data, self.max_len)
        # if self.stage == "pretrain":
        #     feature = self.__create_mask(atoms, self.seq_len, self.num_predict)
        #     feature = self.__make_perm(feature, self.seq_len, self.perm_size, self.num_predict)
        #     input_k = feature['input_k']
        #     perm_mask = feature['perm_mask']
        #     target_mapping = feature['target_mapping']
        #     input_q = feature['input_q']
            
        #     label = feature["target"]
        #     return input_k, perm_mask, target_mapping, input_q, adjoin_matrix, label
        # elif self.stage == "finetune":
        #     feature = {'input_k': atoms[:-1], 'target': self.target[idx], 'perm_mask': None, 'target_mapping': None,
        #                'input_q': None, 'target_mask': None, "adjoin_matrix": adjoin_matrix}
        #     return feature["input_k"], feature["target"], feature["adjoin_matrix"]
        # else:
        #     raise KeyError("Wrong stage attribute.")
        if self.stage == "pretrain":
            return self.dataset[idx]
        elif self.stage == "finetune":
            return self.dataset[idx], self.target[idx]
        elif self.stage == "predict":
            return self.dataset[idx]
        else:
            raise KeyError("Wrong stage attribute.")
    
    def get_input_data_slice_map(self, smiles):
        tokenized_data = self.tokenizer.tokenize(str(smiles))
        atoms, adjoin_matrix = self.__pad_data(tokenized_data, self.max_len)
        if self.stage == "pretrain":
            feature = self.__create_mask(atoms, self.seq_len, self.num_predict)
            feature = self.__make_perm(feature, self.seq_len, self.perm_size, self.num_predict)
            input_k = feature['input_k']
            perm_mask = feature['perm_mask']
            target_mapping = feature['target_mapping']
            input_q = feature['input_q']
            
            label = feature["target"]  
            return input_k, perm_mask, target_mapping, input_q, adjoin_matrix, label
        elif self.stage == "finetune":
            input_k = atoms[:-1]
            cls_pos = np.argwhere(input_k==self.tokenizer.vocab2idx[self.tokenizer.cls]).reshape(-1)
            cls_pos = cls_pos[0]
            return input_k, adjoin_matrix, cls_pos
        elif self.stage == "predict":
            input_k = atoms[:-1]
            cls_pos = np.argwhere(input_k==self.tokenizer.vocab2idx[self.tokenizer.cls]).reshape(-1)
            cls_pos = cls_pos[0]
            return input_k, adjoin_matrix, cls_pos
        

    def get_data(self, rank_size=1, rank_id=0):
        ds.config.set_seed(1)
        # Control the size of data queue in the consideration of the memory
        # ds.config.set_num_parallel_workers=16
        map_func = (lambda input_ids: self.get_input_data_slice_map(input_ids))
        type_cast_op = ds.transforms.TypeCast(ms.float32)
        if self.stage == "pretrain":
            input_columns = ['smiles']
            output_columns = ['input_k', 'perm_mask', 'target_mapping', 'input_q', 'adjoin_matrix', 'label']
            dataset = ds.GeneratorDataset(source=self, shuffle=False,
                                          column_names=input_columns,
                                          num_shards=rank_size, shard_id=rank_id)
            dataset = dataset.map(operations=map_func, input_columns=input_columns, output_columns=output_columns,  
                                  num_parallel_workers=24)
            dataset = dataset.project(columns=['input_k', 'adjoin_matrix', 'label', 'perm_mask', 'target_mapping', 'input_q'])        
            dataset = dataset.batch(self.batch_size, num_parallel_workers=24)
            return dataset

        elif self.stage == "finetune":
            input_columns = ['smiles']
            output_columns = ['input_k', 'adjoin_matrix', 'cls_pos']
            dataset = ds.GeneratorDataset(source=self, shuffle=False,
                                          column_names=['smiles', 'label'])
            dataset = dataset.map(operations=map_func, input_columns=input_columns, output_columns=output_columns)
            dataset = dataset.map(operations=type_cast_op, input_columns=['label'])
            dataset = dataset.project(columns=['input_k', 'adjoin_matrix', 'label', 'cls_pos'])
            dataset = dataset.batch(self.batch_size)
            return dataset
        
        elif self.stage == "predict":
            input_columns = ['smiles']
            output_columns = ['input_k', 'adjoin_matrix', 'cls_pos']
            dataset = ds.GeneratorDataset(source=self, shuffle=False,
                                          column_names=['smiles'])
            dataset = dataset.map(operations=map_func, input_columns=input_columns, output_columns=output_columns)
            dataset = dataset.project(columns=['input_k', 'adjoin_matrix', 'cls_pos'])
            dataset = dataset.batch(self.batch_size)
            return dataset

    @staticmethod
    def __pad_data(data: Tuple[List[int], np.ndarray], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
        assert len(data[0]) == len(data[1]), "Atom list and adjoin matrix shape incompatible."
        padded_atoms = np.pad(data[0], (0, max_len - len(data[0])), 'constant', constant_values=0)
        # input:  [1, 2, 3, 4, 5]
        # inp_k:  [1, 2, 3, 4]
        # target:    [2, 3, 4, 5]
        # Pad then pop the last pad token in adjoin matrix.
        # The pop process is not applied in padded_atoms because it is processed by create_mask().
        padded_adjoin_matrix = np.pad(data[1], (0, max_len - len(data[1]) - 1), 'constant', constant_values=0)
        return padded_atoms, padded_adjoin_matrix

    @staticmethod
    def __create_mask(seq: np.ndarray, seq_len: int, num_predict: int) -> Dict[str, np.ndarray]:
        input = seq[:-1]
        target = seq[1:]

        token_idx = np.array(range(seq_len))
        mask_idx = np.random.choice(token_idx, num_predict, replace=False)
        is_masked = np.array([False] * seq_len, dtype=bool)
        is_masked[mask_idx] = True

        feature = {
            "input": input,
            "is_masked": is_masked,
            "target": target,
        }

        return feature

    def __make_perm(self, feature: Dict[str, np.ndarray], seq_len: int, perm_size: int,
                    num_predict: int) -> Dict[str, np.ndarray]:

        inp = feature.pop("input").astype(np.int32)
        tgt = feature.pop("target").astype(np.int32)
        is_masked = feature.pop("is_masked").astype(np.int64)

        perm_mask, target, target_mask, input_k, input_q = \
            self.__local_perm(inputs=inp,
                              targets=tgt,
                              is_masked=is_masked,
                              perm_size=perm_size,
                              seq_len=len(inp),
                              special_token_id=self.special_tokens_id)

        if num_predict is not None:
            indices = np.arange(seq_len, dtype=np.int64)
            bool_target_mask = target_mask.astype(bool)
            indices = indices[bool_target_mask]  # (actual_num_predict, )

            # 此处 padding 是为了补足 num_predict 与 actual_num_predict 之差
            # 该差值即为被 mask 的 [pad] 与 [unk] 数量
            actual_num_predict = indices.shape[0]
            pad_len = num_predict - actual_num_predict

            assert seq_len >= actual_num_predict

            # target_mapping
            target_mapping = np.eye(seq_len, dtype=np.float32)[indices]  # (seq_len, indices)
            paddings = np.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
            target_mapping = np.concatenate([target_mapping, paddings], axis=0)  # (seq_len, num_predict)
            feature["target_mapping"] = target_mapping.reshape([num_predict, seq_len])  # 确保矩阵形状
            # target
            target = target[bool_target_mask]
            paddings = np.zeros([pad_len], dtype=target.dtype)
            target = np.concatenate([target, paddings], axis=0)  # (target, num_predict)
            feature["target"] = target.reshape([num_predict])

            # target mask
            target_mask = np.concatenate(  # (num_predict, )
                [np.ones([actual_num_predict], dtype=np.float32),
                 np.zeros([pad_len], dtype=np.float32)],
                axis=0)
            feature["target_mask"] = np.reshape(target_mask, [num_predict])
        else:
            feature["target"] = np.reshape(target, [seq_len])
            feature["target_mask"] = np.reshape(target_mask, [seq_len])

        feature["perm_mask"] = np.reshape(perm_mask, [seq_len, seq_len])
        feature["input_k"] = np.reshape(input_k, [seq_len])
        feature["input_q"] = np.reshape(input_q, [seq_len])

        return feature

    @staticmethod
    def __local_perm(inputs: np.ndarray, targets: np.ndarray, is_masked: np.ndarray,
                     perm_size: int, seq_len: int, special_token_id: List[str]) -> Tuple[ndarray,
    ndarray, ndarray, ndarray, ndarray]:
        """
        Sample a permutation of the factorization order, and create an
        attention mask accordingly.

        Args:
        inputs: int64 Tensor in shape [seq_len], input ids.
        targets: int64 Tensor in shape [seq_len], target ids.
        is_masked: bool Tensor in shape [seq_len]. True means being selected
          for partial prediction.
        perm_size: the length of the longest permutation. Could be set to be reuse_len.
          Should not be larger than reuse_len or there will be data leaks.
        seq_len: int, sequence length.
        """

        # Generate permutation indices
        index = np.arange(seq_len, dtype=np.int64)  # (seq_len, )
        index = np.reshape(index, [-1, perm_size]).transpose()  # (-1, perm_size) -> transpose: (perm_size, -1)
        index = index[np.random.permutation(index.shape[0])]  # 在第 0 个维度（perm_size）上打乱顺序
        index = np.reshape(index.transpose(), [-1])  # (seq_len, )，最终效果为在 seq 中逐个长度为 perm_size 的片段上打乱

        # `perm_mask` and `target_mask`
        # 将非 [sep] 或 [cls] 的 token 称为 non-functional tokens，即普通 token，与之相反即为特殊 token
        non_func_tokens = ~np.isin(inputs, special_token_id)
        # non_mask_tokens 为未被 mask 的普通 token
        non_mask_tokens = ~is_masked.astype(bool) & non_func_tokens
        # masked_or_func_tokens 为被 mask 的 token 和特殊 token
        masked_or_func_tokens = ~non_mask_tokens

        # 为 non_mask_tokens 分配索引 smallest_index
        # smallest index 的索引值为 -1，其作用是使 token（context tokens）:
        # (1) 能被其他位置的所有 token 看见
        # (2) 不能看见所有被 mask 的 token
        smallest_index = -np.ones([seq_len], dtype=np.int64)

        # 以 non_mask_tokens[i] 作为条件指定 rev_index[i] ，True 则为 -1，False 则为 index[i]
        # 该步骤的目的是使 non_mask_tokens 分配到 -1 索引，~non_mask_tokens 分配到随机索引
        rev_index = np.where(non_mask_tokens, smallest_index, index)  # (seq_len, )

        # Create `target_mask`: non-functional and masked tokens
        # 1: use mask as input and have loss
        # 0: use token (or [SEP], [CLS]) as input and do not have loss
        # target_tokens 为被 mask 的 token
        target_tokens = masked_or_func_tokens & non_func_tokens
        target_mask = target_tokens.astype(np.float32)

        # Create `perm_mask`
        # `target_tokens` cannot see themselves
        # put `rev_index` if real mask(not cls or sep) else `rev_index + 1`
        self_rev_index = np.where(target_tokens, rev_index, rev_index + 1)  # (seq_len, )

        # self_rev_index 作为列，rev_index 作为行，广播式求值，得到 (seq_len, seq_len) 的 bool 矩阵
        # & masked_or_func_tokens 将 mask 与 特殊 token 对应列全设置为 1
        # 最后得到 perm_mask 矩阵中，perm_mask[i][j] 的值表示：
        # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
        # 0: can attend if i > j or j is non-masked

        perm_mask = (self_rev_index[:, None] <= rev_index[None, :]) & masked_or_func_tokens
        perm_mask = perm_mask.astype(np.float32)

        # 试举一例描述以上过程：
        # x, y 分别表示随机索引（置换后的索引）
        # seq:              token_a token_b mask    [CLS]   token_c
        # non_mask:         1       1       0       0       1
        # rev_index:       -1      -1       x       y      -1
        # target_token:     0       0       1       0       0
        # self_rev_index:   0       0       x       y+1     0

        # perm_mask:
        #    rev_index   : [[-1.,   -1.,    x,     y,    -1. ]]
        # self_rev_index | [[ 0.,    0.,    1.,    1.,    0. ]] (masked_or_func_tokens)
        #      [[0. ],     [[False, False,  True,  True, False],
        #       [0. ],      [False, False,  True,  True, False],
        #       [x  ],      [False, False,  True,  ?   , False],
        #       [y+1],      [False, False,  ?   , False, False],
        #       [0. ]]      [False, False,  True,  True, False]]

        # 结果就是所有非 masked_or_func_tokens 都能被其他 token 看见（值为 False）
        # func_token 能被自己看见，也可能由其他 masked_token 或 func_tokens 看见，
        # 这取决于该 token 是否在该 func_token 之后（即 x > y）
        # masked_token 不能被自己看见，但可能由其他 masked_token 或 func_tokens 看见，
        # 同样取决于该 token 是否在该 masked_token 之后（即 y > x）

        # new target: [next token] for LM and [curr token] (self) for PLM（？？？）
        new_targets = np.concatenate([inputs[0: 1], targets[: -1]], axis=0)

        # construct inputs_k
        inputs_k = inputs

        # construct inputs_q
        inputs_q = target_mask

        return perm_mask, new_targets, target_mask, inputs_k, inputs_q
