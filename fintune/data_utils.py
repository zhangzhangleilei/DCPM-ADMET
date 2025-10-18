import logging
import os.path
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from tqdm import tqdm
import math

import time
from mindspore.train.callback import Callback
import random


class Tokenizer:
    def __init__(self, word_table, special_tokens: Optional[List[str]] = None):
        self.word_table = word_table
        self.pad = "{pad}"
        self.unk = "{unk}"
        self.cls = "{cls}"
        if special_tokens is not None:
            self.special_tokens = [self.pad] + [self.cls] + [self.unk] + special_tokens
        else:
            self.special_tokens = [self.pad] + [self.cls] + [self.unk]
        self.word_table = self.special_tokens + self.word_table

        self.vocab2idx = {w: i for i, w in enumerate(self.word_table)}
        self.vocab2smi = {i: w for i, w in enumerate(self.word_table)}
        self.vocab_size = len(self.vocab2smi)
        self.special_tokens_id = [self.vocab2idx[token] for token in self.special_tokens]

    def tokenize(self, smi: str) -> List[int]:
        tokens = smi.strip().split()
        return [self.vocab2idx.get(token, self.vocab2idx[self.unk]) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.vocab2smi.get(id, self.vocab2smi[self.unk]) for id in ids]


class SMILESTokenizer(Tokenizer):
    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self.word_table = \
            ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B',
             'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]',
             '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]',
             '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]',
             '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]',
             '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]',
             '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']
        super().__init__(word_table=self.word_table, special_tokens=special_tokens)

    def tokenize(self, smi: str) -> List[int]:
        smi = smi.strip()
        pattern = "(\[[^\]]+]|{unk}|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[" \
                  "0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = regex.findall(smi)
        while "".join(tokens) != smi:
            unknown_words = re.sub(pattern, " ", smi).strip().split()
            for word in unknown_words:
                smi = smi.replace(word, self.unk)
            tokens = regex.findall(smi)
        tokens = [self.cls] + ["{bos}"] + tokens + ["{eos}"]
        return [self.vocab2idx.get(token, self.vocab2idx[self.unk]) for token in tokens]


class GraphTokenizer(Tokenizer):
    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self.word_table = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'P', 'Br', 'B', 'I', 'Si', 'Se']
        super().__init__(word_table=self.word_table, special_tokens=special_tokens)

    def tokenize(self, smi: str) -> Optional[Tuple[List[int], np.ndarray]]:
        tokens, adjoin_matrix = smiles2adjoin(smi, explicit_hydrogen=True)
        tokens = [self.cls] + tokens
        indices = [self.vocab2idx.get(token, self.vocab2idx[self.unk]) for token in tokens]
        expanded_adjoin_matrix = np.ones((len(indices), len(indices)), dtype=np.float32)
        expanded_adjoin_matrix[1:, 1:] = adjoin_matrix
        # unlinked atom edge should be -inf
        adjoin_matrix = (1 - expanded_adjoin_matrix) * -1e30

        return indices, adjoin_matrix


class Log(logging.Logger):
    def __init__(self, name: str, log_path: str) -> None:
        super().__init__(name)
        self.log_path = log_path

        self.setLevel(logging.DEBUG)
        self.file_handler = logging.FileHandler(log_path, mode="a+")
        self.console_handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)
        self.addHandler(self.file_handler)
        self.addHandler(self.console_handler)

    def console_off(self) -> None:
        self.removeHandler(self.console_handler)

    def console_on(self) -> None:
        self.addHandler(self.console_handler)


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None


def preprocess(path: str, workers: int, is_large_data: bool = False, **kwargs) -> str:
    processed_file_path = "".join([os.path.splitext(path)[-2], "_processed", ".csv"])
    pandarallel.initialize(nb_workers=workers)
    if not os.path.exists(processed_file_path):
        if is_large_data:
            reader = pd.read_csv(path, chunksize=1000000, iterator=True, **kwargs)
            for chunk in tqdm(reader):
                chunk_df = chunk.dropna().iloc[:, 0].parallel_apply(canonicalize_smiles).dropna()
                chunk_df.to_csv(processed_file_path,header=False, index=False,
                                mode="a+", encoding="utf-8")
        else:
            df = pd.read_csv(path, **kwargs)
            df = df.dropna().iloc[:, 0].parallel_apply(canonicalize_smiles).dropna()
            df.to_csv(processed_file_path, header=False, index=False,
                      mode="w+", encoding="utf-8")
        print(f"Generated processed data file {processed_file_path}.")
    else:
        print(f"Processed data file {processed_file_path} existed.")

    return processed_file_path


def smiles2adjoin(smiles: str, explicit_hydrogen: bool = True,
                  canonical_atom_order: bool = False) -> Tuple[List[str], np.ndarray]:
    """Smiles to adjoin"""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogen:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    adjoin_matrix = np.eye(num_atoms, dtype=np.float32)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0
    return atoms_list, adjoin_matrix


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, dataset_size=-1, local_rank=0, device_num=8, per_print_times=1, has_trained_epoch=0, has_trained_step=0, 
                 micro_size=1, is_last_stage=True):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self.local_rank = local_rank
        self.device_num = device_num
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step
        self.micro_size = micro_size
        self.is_last_stage = is_last_stage
        self._per_print_times = per_print_times
        self._last_print_time = 0
        print("Load the trained epoch :{} and step: {}".format(has_trained_epoch, has_trained_step), flush=True)

    def on_train_step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0 and self.local_rank % self.device_num == 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num /
                                           self._dataset_size)
            if percent == 0:
                epoch_num -= 1
            date = time.asctime(time.localtime(time.time()))
            loss_value = 'no loss for this stage'
            if self.is_last_stage:
                loss_value = cb_params.net_outputs
                if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
                    while cb_params.cur_step_num <= self._last_print_time:
                        self._last_print_time -=\
                            max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

                if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
                    self._last_print_time = cb_params.cur_step_num
                    print("time: {} local_rank: {}, epoch: {}, step: {}, loss is {}".
                        format(date, int(self.local_rank), int(epoch_num) + int(self.has_trained_epoch),
                                cb_params.cur_step_num + int(self.has_trained_step), loss_value))
                    
                    
def dataset_split(data_path, output_path, require_valid=True):
    df =  pd.read_csv(data_path, header=None)
    rows = len(df)
    test_size = rows // 10
    index = [i for i in range(rows)]
    test_index = sorted(random.sample(index, test_size))
    train_index = sorted(list(set(index) - set(test_index)))
    test_dataset = df.iloc[test_index]

    if require_valid:
        valid_index = sorted(random.sample(train_index, test_size))
        train_index = sorted(list(set(train_index) - set(valid_index)))
        valid_dataset = df.iloc[valid_index]
        valid_dataset_path = "".join([os.path.splitext(output_path)[-2], "_valid", ".csv"])
        valid_dataset.to_csv(valid_dataset_path, header=None, index=False)

    train_dataset = df.iloc[train_index]
    train_dataset_path = "".join([os.path.splitext(output_path)[-2], "_train", ".csv"])
    test_dataset_path = "".join([os.path.splitext(output_path)[-2], "_test", ".csv"])
    train_dataset.to_csv(train_dataset_path, header=None, index=False)
    test_dataset.to_csv(test_dataset_path, header=None, index=False)

    if require_valid:
        return train_dataset_path, valid_dataset_path, test_dataset_path
    else:
        return train_dataset_path, test_dataset_path


def download_data(src_data_url, tgt_data_path, rank):
    """
        Download the dataset from the obs.
        src_data_url (Str): should be the dataset path in the obs
        tgt_data_path (Str): the local dataset path
        rank (Int): the current rank id

    """
    cache_url = tgt_data_path
    EXEC_PATH = "/tmp"
    if rank % 8 == 0:
        import moxing as mox
        print("Modify the time out from 300 to 30000")
        print("begin download dataset", flush=True)

        if not os.path.exists(cache_url):
            os.makedirs(cache_url, exist_ok=True)
        mox.file.copy_parallel(src_url=src_data_url, dst_url=cache_url)
        print("Dataset download succeed!", flush=True)

        f = open("%s/install.txt" % (EXEC_PATH), "w")
        f.close()
    # stop
    while not os.path.exists("%s/install.txt" % (EXEC_PATH)):
        time.sleep(1)
        