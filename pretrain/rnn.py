from typing import Optional, List

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore.nn.cell import Cell


class Encoder(nn.Cell):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.d_hidden = d_hidden

        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.gru = nn.GRU(d_embed, d_hidden, num_layers, dropout=dropout, batch_first=True)

        # this dense is un-trainable ?
        self.dense = nn.Dense(d_hidden * num_layers, d_out, activation=activation, has_bias=False)

    def construct(
        self,
        encoder_inp: ms.Tensor,
        inp_len: ms.Tensor,
        init_hidden: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        # encoder_inp: [bsz, seq_len]
        # input_len: [bsz]
        # init_hidden: [num_layers, bsz, d_hidden]

        encoder_inp = self.embedding(encoder_inp)
        # encoder_inp: [bsz, seq_len, d_embed]

        if init_hidden is None:
            init_hidden = mnp.zeros((self.num_layers, encoder_inp.shape[0], self.d_hidden))

        output, state = self.gru(encoder_inp, init_hidden, inp_len)
        # output: [bsz, seq_len, d_hidden]
        # state: [num_layers, bsz, d_hidden]

        layer_states = tuple(
            tensor.squeeze(axis=0) for tensor in ops.split(state, axis=0, output_num=state.shape[0])
        )
        layer_states = ops.concat(layer_states, axis=1)
        # hidden_states: [bsz, d_enc_out * num_layers]

        hidden_state = self.dense(layer_states)
        # state: [bsz, d_out]

        return output, hidden_state


class NoisyEncoder(Encoder):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
        dropout: float,
        activation: str,
        noise: float,
    ) -> None:
        super().__init__(vocab_size, d_embed, d_hidden, d_out, num_layers, dropout, activation)

        assert noise >= 0, "Attribute `noise` should be a non-negative value."
        self.noise = noise

    def construct(
        self, encoder_inp: ms.Tensor, inp_len: ms.Tensor, init_hidden: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        output, state = super().construct(encoder_inp, inp_len, init_hidden)
        state = state + ops.normal(
            state.shape, mean=ms.Tensor(0.0, ms.float32), stddev=ms.Tensor(self.noise, ms.float32)
        )

        return output, state

    def no_noise_predict(
        self, encoder_inp: ms.Tensor, inp_len: ms.Tensor, init_hidden: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        return super().construct(encoder_inp, inp_len, init_hidden)


class Decoder(nn.Cell):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        d_encoder: int,
        d_hidden: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.gru = nn.GRU(d_embed, d_hidden, num_layers, dropout=dropout, batch_first=True)
        self.proj_layer = nn.Dense(d_hidden, vocab_size)
        self.softmax = nn.Softmax()

        # this dense is un-trainable
        self.dense = nn.Dense(
            d_encoder, d_hidden * num_layers, activation=activation, has_bias=False
        )

    def construct(self, encoded_seq: ms.Tensor, decoder_inp: ms.Tensor) -> ms.Tensor:
        # encoded_seq: [bsz, d_encoder]
        # decoder_inp: [bsz, seq_len]
        # decoder_inp is combined of a batch of "<sos>" token

        init_hidden = self.dense(encoded_seq)
        # init_hidden: [bsz, d_hidden * num_layers]

        layer_states = tuple(
            tensor.unsqueeze(dim=0)
            for tensor in ops.split(init_hidden, axis=1, output_num=self.num_layers)
        )

        decoder_hidden = ops.concat(layer_states, axis=0)
        # decoder_hidden: [num_layers, batch_size, d_hidden]

        decoder_inp = self.embedding(decoder_inp)
        decoder_output, decoder_hidden = self.gru(decoder_inp, decoder_hidden)
        decoder_output = self.proj_layer(decoder_output)
        # decoder_hidden: [num_layers, bsz, d_hidden]
        # decoder_output: [bsz, seq_len, vocab_size]

        decoder_output = self.softmax(decoder_output)

        return decoder_output, decoder_hidden

    @staticmethod
    def bundle_concat(tensors: List[ms.Tensor], axis: int, bundle_size: int = 20) -> ms.Tensor:
        tensors_num = len(tensors)
        while tensors_num > bundle_size:
            slices = [slice(idx, idx + bundle_size) for idx in range(0, tensors_num, bundle_size)]

            new_tensors = [ops.concat(tensors[slice_], axis=axis) for slice_ in slices]
            tensors = new_tensors
            tensors_num = len(tensors)

        return ops.concat(tensors, axis=axis)


class GRUSeq2Seq(nn.Cell):
    def __init__(
        self,
        d_embed: int,
        d_hidden: int,
        d_enc_out: int,
        num_layers: int,
        enc_dropout: float,
        dec_dropout: float,
        enc_activation: str,
        dec_activation: str,
        enc_vocab_size: int,
        dec_vocab_size: int,
    ):
        super().__init__()

        self.encoder = Encoder(
            enc_vocab_size,
            d_embed,
            d_hidden,
            d_enc_out,
            num_layers,
            enc_dropout,
            enc_activation,
        )
        self.decoder = Decoder(
            dec_vocab_size,
            d_embed,
            d_enc_out,
            d_hidden,
            num_layers,
            dec_dropout,
            dec_activation,
        )

    def construct(self, enc_inp: ms.Tensor, inp_len: ms.Tensor, dec_inp: ms.Tensor):
        _, encoded_seq = self.encoder(enc_inp, inp_len)
        # encoded_seq: [bsz, d_encoder]
        decoded_seq, _ = self.decoder(encoded_seq, dec_inp)
        # decoded_seq: [bsz, seq_len, dec_vocab_size]

        return decoded_seq

    def predict(self, enc_inp: ms.Tensor, inp_len: ms.Tensor):
        _, encoded_seq = self.encoder(enc_inp, inp_len)
        # layer_hidden_states = tuple(
        #     tensor.squeeze(axis=0)
        #     for tensor in ops.split(encoded_seq, axis=0, output_num=encoded_seq.shape[0])
        # )
        # hidden_states = ops.concat(layer_hidden_states, axis=1)
        # # hidden_states: [bsz, d_enc_out * num_layers]

        return encoded_seq


class GRUSeq2SeqWithFeatures(GRUSeq2Seq):
    def __init__(
        self,
        d_embed: int,
        d_hidden: int,
        d_enc_out: int,
        num_layers: int,
        enc_dropout: float,
        dec_dropout: float,
        enc_activation: str,
        dec_activation: str,
        enc_vocab_size: int,
        dec_vocab_size: int,
        feature_size: int,
    ):
        super().__init__(
            d_embed,
            d_hidden,
            d_enc_out,
            num_layers,
            enc_dropout,
            dec_dropout,
            enc_activation,
            dec_activation,
            enc_vocab_size,
            dec_vocab_size,
        )

        self.feat_reg_f1 = nn.Dense(d_enc_out, 512, activation="relu")
        self.feat_reg_f2 = nn.Dense(512, 128, activation="relu")
        self.feat_reg_f3 = nn.Dense(128, feature_size)

    def construct(self, enc_inp: ms.Tensor, inp_len: ms.Tensor, dec_inp: ms.Tensor):
        _, encoded_seq = self.encoder(enc_inp, inp_len)
        # encoded_seq: [bsz, d_encoder]
        decoded_seq, _ = self.decoder(encoded_seq, dec_inp)
        # decoded_seq: [bsz, seq_len, dec_vocab_size]

        feat_predict = self.feature_reg(encoded_seq)
        # feat_predict: [bsz, feature_size]

        return decoded_seq, feat_predict

    def feature_reg(self, encoded_seq: ms.Tensor) -> ms.Tensor:
        # encoded_seq: [bsz, d_encoder]
        x = self.feat_reg_f1(encoded_seq)
        x = self.feat_reg_f2(x)
        x = self.feat_reg_f3(x)
        # x: [bsz, feature_size]

        return x


class NoisyGRUSeq2SeqWithFeatures(GRUSeq2SeqWithFeatures):
    def __init__(
        self,
        d_embed: int,
        d_hidden: int,
        d_enc_out: int,
        num_layers: int,
        enc_dropout: float,
        dec_dropout: float,
        enc_activation: str,
        dec_activation: str,
        noise: float,
        enc_vocab_size: int,
        dec_vocab_size: int,
        feature_size: int,
    ):
        super().__init__(
            d_embed,
            d_hidden,
            d_enc_out,
            num_layers,
            enc_dropout,
            dec_dropout,
            enc_activation,
            dec_activation,
            enc_vocab_size,
            dec_vocab_size,
            feature_size,
        )

        self.encoder = NoisyEncoder(
            enc_vocab_size,
            d_embed,
            d_hidden,
            d_enc_out,
            num_layers,
            enc_dropout,
            enc_activation,
            noise,
        )

    def construct(self, enc_inp: ms.Tensor, inp_len: ms.Tensor, dec_inp: ms.Tensor):
        return super().construct(enc_inp, inp_len, dec_inp)

    def predict(self, enc_inp: ms.Tensor, inp_len: ms.Tensor):
        _, encoded_seq = self.encoder.no_noise_predict(enc_inp, inp_len)
        return encoded_seq


class Seq2SeqTrainNet(nn.Cell):
    def __init__(self, network: nn.Cell, ignore_index: int):
        super().__init__()

        self._network = network
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def construct(
        self, enc_inp: ms.Tensor, inp_len: ms.Tensor, dec_inp: ms.Tensor, tgt: ms.Tensor
    ) -> ms.Tensor:
        output = self._network(enc_inp, inp_len, dec_inp)
        # output: [bsz, seq_len, dec_vocab_size]
        loss = self._loss_fn(output.transpose(0, 2, 1), tgt)

        return loss


class Seq2SeqEvalNet(nn.Cell):
    def __init__(self, network: Cell, ignore_index: int):
        super().__init__()

        self._network = network
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def construct(
        self, enc_inp: ms.Tensor, inp_len: ms.Tensor, dec_inp: ms.Tensor, tgt: ms.Tensor
    ) -> ms.Tensor:
        output = self._network(enc_inp, inp_len, dec_inp)
        loss = self._loss_fn(output.transpose(0, 2, 1), tgt)
        # decoded_seq = output.argmax(axis=2)
        # decoded_seq: [bsz, seq_len]

        return output, tgt, loss


class Seq2SeqFeatureTrainNet(nn.Cell):
    def __init__(self, network: nn.Cell, ignore_index: int):
        super().__init__()

        self._network = network
        self._seq_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._feat_loss_fn = nn.MSELoss()

    def construct(
        self,
        enc_inp: ms.Tensor,
        inp_len: ms.Tensor,
        dec_inp: ms.Tensor,
        tgt: ms.Tensor,
        feat_tgt: ms.Tensor,
    ) -> ms.Tensor:
        seq_output, feat_output = self._network(enc_inp, inp_len, dec_inp)
        # seq_output: [bsz, seq_len, dec_vocab_size]
        # feat_output: [bsz, feature_size]
        seq_loss = self._seq_loss_fn(seq_output.transpose(0, 2, 1), tgt)
        feat_loss = self._feat_loss_fn(feat_output, feat_tgt)

        return seq_loss + feat_loss
