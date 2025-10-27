import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore.train import Callback
from mindspore.train.callback._callback import _handle_loss
from numpy.typing import NDArray


class Tokenizer(ABC):
    def __init__(self, word_table: List[str], special_tokens: Optional[List[str]] = None):
        self.pad = "{pad}"
        self.unk = "{unk}"
        self.cls = "{cls}"
        self.special_tokens = [self.pad, self.cls, self.unk]

        if special_tokens is not None:
            self.special_tokens.extend(special_tokens)

        self.word_table = self.special_tokens + word_table

        self.vocab2index = {w: i for i, w in enumerate(self.word_table)}
        self.vocab2token = {i: w for i, w in enumerate(self.word_table)}
        self.vocab_size = len(self.word_table)
        self.special_tokens_id = [self.vocab2index[token] for token in self.special_tokens]

    @abstractmethod
    def tokenize(self, seq: str) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        assert False, "Abstract method `tokenize` has not yet initialized."

    def format_tokens(self, tokens: List[str]) -> List[str]:
        return tokens

    def find_tokens(self, seq: str, pattern: str) -> List[str]:
        seq = seq.strip()
        regex = re.compile(pattern)
        tokens = regex.findall(seq)

        while "".join(tokens) != seq:
            unknown_words = re.sub(pattern, " ", seq).strip().split()
            for word in unknown_words:
                seq = seq.replace(word, self.unk)
            tokens = regex.findall(seq)

        return self.format_tokens(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.vocab2token.get(id, self.unk) for id in ids]


class Seq2SeqTokenizer(Tokenizer):
    def __init__(self, word_table: List[str], pattern: str):
        self.bos = "{bos}"
        self.eos = "{eos}"
        self.pattern = pattern
        special_tokens = [self.bos, self.eos]

        super().__init__(word_table, special_tokens)

    def format_tokens(self, tokens: List[str]) -> List[str]:
        return [self.bos] + tokens + [self.eos]

    def tokenize(self, seq: str) -> NDArray:
        tokens = self.find_tokens(seq, self.pattern)
        return np.array(
            [self.vocab2index.get(token, self.vocab2index[self.unk]) for token in tokens],
            dtype=np.int32,
        )

    def tokenize2str(self, seq: str) -> str:
        tokens = self.find_tokens(seq, self.pattern)
        return "".join(tokens)

    def fast_tokenize(self, seq: str) -> NDArray:
        tokens = seq.split()
        return np.array(
            [self.vocab2index.get(token, self.vocab2index[self.unk]) for token in tokens],
            dtype=np.int32,
        )


class SMILESTokenizer(Seq2SeqTokenizer):
    def __init__(self):
        word_table = [
            "#",
            "%10",
            "%11",
            "%12",
            "(",
            ")",
            "-",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "<",
            "=",
            "B",
            "Br",
            "C",
            "Cl",
            "F",
            "I",
            "N",
            "O",
            "P",
            "S",
            "[B-]",
            "[BH-]",
            "[BH2-]",
            "[BH3-]",
            "[B]",
            "[C+]",
            "[C-]",
            "[CH+]",
            "[CH-]",
            "[CH2+]",
            "[CH2]",
            "[CH]",
            "[F+]",
            "[H]",
            "[I+]",
            "[IH2]",
            "[IH]",
            "[N+]",
            "[N-]",
            "[NH+]",
            "[NH-]",
            "[NH2+]",
            "[NH3+]",
            "[N]",
            "[O+]",
            "[O-]",
            "[OH+]",
            "[O]",
            "[P+]",
            "[PH+]",
            "[PH2+]",
            "[PH]",
            "[S+]",
            "[S-]",
            "[SH+]",
            "[SH]",
            "[Se+]",
            "[SeH+]",
            "[SeH]",
            "[Se]",
            "[Si-]",
            "[SiH-]",
            "[SiH2]",
            "[SiH]",
            "[Si]",
            "[b-]",
            "[bH-]",
            "[c+]",
            "[c-]",
            "[cH+]",
            "[cH-]",
            "[n+]",
            "[n-]",
            "[nH+]",
            "[nH]",
            "[o+]",
            "[s+]",
            "[sH+]",
            "[se+]",
            "[se]",
            "b",
            "c",
            "n",
            "o",
            "p",
            "s",
        ]

        pattern = (
            "(\[[^\]]+]|{unk}|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
            "\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%"
            "[0-9]{2}|[0-9])"
        )

        super().__init__(word_table, pattern)


class InchiTokenizer(Seq2SeqTokenizer):
    def __init__(self):
        word_table = [
            "(",
            ")",
            "+",
            ",",
            "-",
            ".",
            "/",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "Br",
            "C",
            "Cl",
            "F",
            "H",
            "I",
            "InChI=1S",
            "N",
            "O",
            "P",
            "S",
            "c",
            "h",
            "p",
            "q",
        ]

        pattern = r"{unk}|InChI=1S|Br|Cl|[\(\)\+,-/0123456789CFHINOPSchpq]"

        super().__init__(word_table, pattern)


class Metric:
    def __init__(self) -> None:
        self.y_pred: list = []
        self.y_true: list = []

    def clear(self) -> None:
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred: NDArray, y_true: NDArray) -> None:
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def eval(self) -> dict[str, float]:
        return {}


class Seq2SeqMetric(Metric):
    def __init__(self, ignore_label: int) -> None:
        super().__init__()
        self._perplexity = ms.train.Perplexity(ignore_label=ignore_label)
        self._bleu = ms.train.BleuScore()

    def eval(self) -> dict[str, float]:
        all_y_pred = np.concatenate(self.y_pred, axis=0)
        all_y_true = np.concatenate(self.y_true, axis=0)

        self._perplexity.clear()
        self._bleu.clear()

        self._perplexity.update(all_y_pred, all_y_true)
        perplexity_score = self._perplexity.eval()

        all_token_pred = all_y_pred.argmax(axis=2)
        self._bleu.update(all_token_pred, all_y_true)
        bleu_score = self._bleu.eval()

        result = {
            "Perplexity": perplexity_score,
            "BLEU score": bleu_score,
        }

        return result


class DebugMonitor(ms.train.Callback):
    def __init__(self, metric: Optional[Metric] = None) -> None:
        super(DebugMonitor, self).__init__()
        if metric is not None:
            self._metric = metric

    def on_eval_step_end(self, run_context):
        cb_params = run_context.original_args()
        output = cb_params.net_outputs
        y = output[0].asnumpy()
        label = output[1].asnumpy()

        self._metric.update(y, label)

        # print("Predict:", output[0])
        # print("Label:", output[1])

    def on_eval_epoch_end(self, run_context):
        result = self._metric.eval()
        print(", ".join([f"{k}: {v}" for k, v in result.items()]))


class TimeLossMonitor(Callback):
    def __init__(self, per_print_times=1):
        super().__init__()
        self._per_print_times = per_print_times
        self._last_print_time = 0

    def on_train_step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.  For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()

        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        loss = _handle_loss(cb_params.net_outputs)

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ",
                "In epoch: {} step: {}, loss is NAN or INF, training process cannot continue, "
                "terminating training.".format(cur_epoch_num, cur_step_in_epoch),
            )

        # In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -= max(
                    self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1
                )

        if (
            self._per_print_times != 0
            and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times
        ):
            self._last_print_time = cb_params.cur_step_num
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ",
                "epoch: %s step: %s, loss is %s" % (cur_epoch_num, cur_step_in_epoch, loss),
                flush=True,
            )

    def on_train_epoch_end(self, run_context):
        """
        When LossMonitor used in `model.fit`, print eval metrics at the end of epoch if current epoch
        should do evaluation.

        Args:
            run_context (RunContext): Include some information of the model. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()
        metrics = cb_params.get("metrics")
        if metrics:
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ",
                "Eval result: epoch %d, metrics: %s" % (cb_params.cur_epoch_num, metrics),
            )


def calc_trainable_params(model: nn.Cell):
    return sum(np.prod(param.shape) for param in model.trainable_params())


def calc_untrainable_params(model: nn.Cell):
    return sum(np.prod(param.shape) for param in model.untrainable_params())
