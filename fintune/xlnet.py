import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
import mindspore.numpy as mnp
from mindspore.common.initializer import initializer, HeUniform
from mindspore import Parameter


def dense(in_channel, out_channel, use_se=True, activation=None):
    """Custom dense"""
    if not use_se:
        weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
        weight = ms.Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=ms.float32)
    else:
        boundary = np.sqrt(6 / (out_channel + in_channel))
        weight_shape = (out_channel, in_channel)
        weight = ms.Tensor(np.random.uniform(-boundary, boundary, weight_shape), dtype=ms.float32)

    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0, activation=activation)


class XLNetLayer(nn.Cell):
    def __init__(self, n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type):
        super(XLNetLayer, self).__init__()

        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.d_model = d_model
        self.dropout = dropout
        self.dropatt = dropatt
        self.attn_type = attn_type
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length
        self.param_init_type = param_init_type
        self.cast = ops.Cast()
        

        self.embedding = nn.Embedding(n_token, d_model, dtype=self.param_init_type)
        self.Dropout = nn.Dropout(keep_prob=1 - dropout)
        self.DropAttn = nn.Dropout(keep_prob=1 - dropatt)

        self.r_w_bias = Parameter(initializer(HeUniform(), [self.n_layer, self.n_head, self.d_head], dtype=self.param_init_type))
        self.r_r_bias = Parameter(initializer(HeUniform(), [self.n_layer, self.n_head, self.d_head], dtype=self.param_init_type))

        self.mask_emb = Parameter(initializer(HeUniform(), [1, 1, self.d_model], dtype=self.param_init_type))

        # post-attention projection (back to `d_model`)
        self.proj_o = Parameter(initializer(HeUniform(), [self.d_model, self.n_head, self.d_head], dtype=self.param_init_type))

        # Project hidden states to a specific head with a 4D-shape.
        self.q_proj_weight = Parameter(initializer(HeUniform(), [self.d_model, self.n_head, self.d_head], dtype=self.param_init_type))
        self.k_proj_weight = Parameter(initializer(HeUniform(), [self.d_model, self.n_head, self.d_head], dtype=self.param_init_type))
        self.v_proj_weight = Parameter(initializer(HeUniform(), [self.d_model, self.n_head, self.d_head], dtype=self.param_init_type))
        self.r_proj_weight = Parameter(initializer(HeUniform(), [self.d_model, self.n_head, self.d_head], dtype=self.param_init_type))

        self.layer_norm = nn.LayerNorm((d_model,))

        self.ffn_layers = nn.CellList([FFN(d_model=d_model,
                                           d_inner=d_inner,
                                           dropout=self.dropout) for _ in range(n_layer)])


    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape
        # klen = klen (seq_len)
        x = ops.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
        x = x[1:, 0:, 0:, 0:]  # tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = ops.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
        x = x[0:, 0:klen, 0:, 0:]  # tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

        return x

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""

        # post-attention projection (back to `d_model`)
        # attn_vec: (seq_len, bsz, n_head, d_head)
        # proj_o: (d_model, n_head, d_head)
        # -> (seq_len, bsz, d_model)
        # attn_out = ops.einsum('ibnd,hnd->ibh', attn_vec, self.proj_o)
        attn_out = ops.matmul(attn_vec.transpose([0, 3, 1, 2]),
                              self.proj_o.transpose([2, 1, 0])).sum(axis=1)

        attn_out = self.Dropout(attn_out)
        if residual:
            output = self.layer_norm(attn_out + h)
        else:
            output = self.layer_norm(attn_out)

        return output

    def head_projection(self, h, name):
        """Project hidden states to a specific head with a 4D-shape."""

        if name == 'q':
            proj_weight = self.q_proj_weight
        elif name == 'k':
            proj_weight = self.k_proj_weight
        elif name == 'v':
            proj_weight = self.v_proj_weight
        elif name == 'r':
            proj_weight = self.r_proj_weight
        else:
            raise ValueError('Unknown `name` {}.'.format(name))

        # This einsum op raise weired error in mindspore Graph mode
        # head = ops.einsum('ibh,hnd->ibnd', h, proj_weight)
        head = ops.tensor_dot(h, proj_weight, axes=1)
        # name = 'q' or 'k':
        # h: (seq_len, bsz, d_model), proj_weight: (d_model, n_head, d_head)
        # -> head: (seq_len, bsz, n_head, d_head)
        # 即把 d_model 变成 n_head * d_head
        return head

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r,
                      r_w_bias, r_r_bias, attn_mask, adjoin_matrix, scale):
        """Core relative positional attention operations."""

        # content based attention score
        # q_head + r_w_bias: (seq_len, bsz, n_head, d_head)，r_w_bias: (n_head, d_head)
        # k_head_h: (seq_len, bsz, n_head, d_head)
        # -> (seq_len, seq_len, bsz, n_head)
        # ac = ops.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)
        ac = ops.matmul((q_head + r_w_bias).transpose([1, 2, 0, 3]),
                        k_head_h.transpose([1, 2, 3, 0])).transpose([2, 3, 0, 1])
        # position based attention score
        # q_head + r_r_bias: (seq_len, bsz, n_head, d_head)，r_r_bias: (n_head, d_head)
        # k_head_r: (2*seq_len, bsz, n_head, d_head)
        # -> (seq_len, 2*seq_len, bsz, n_head)
        # bd = ops.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
        bd = ops.matmul((q_head + r_r_bias).transpose([1, 2, 0, 3]),
                        k_head_r.transpose([1, 2, 3, 0])).transpose([2, 3, 0, 1])
        # -> (seq_len, seq_len, bsz, n_head)
        bd = self.rel_shift(bd, klen=ac.shape[1])

        # adjoin matrix will mask unlinked atoms, add -inf to attention score
        # only calculate attention of linked atoms
        if adjoin_matrix is not None:
            attn_score = (ac + bd) * scale + adjoin_matrix.unsqueeze(-1)
        else:
            # merge attention scores and perform masking
            # attn_score: (seq_len, seq_len, bsz, n_head)
            attn_score = (ac + bd) * scale

        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            attn_score = attn_score - 1e5 * attn_mask

        # attention probability
        # attn_prob: (seq_len, seq_len, bsz, n_head)
        attn_prob = ops.softmax(attn_score, axis=1)
        attn_prob = self.DropAttn(attn_prob)

        # attention output
        # v_head_h: (seq_len, bsz, n_head, d_head)
        # -> (seq_len, bsz, n_head, d_head)
        # attn_vec = ops.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
        attn_vec = ops.matmul(attn_prob.transpose([2, 3, 0, 1]),
                              v_head_h.transpose([1, 2, 0, 3])).transpose([2, 0, 1, 3])
        # attn_prob: (seq_len, seq_len, bsz, n_head)
        return (attn_vec, attn_prob)


    def rel_multihead_attn(self, h, r, r_w_bias, r_r_bias, attn_mask, adjoin_matrix):
        """Multi-head attention with relative positional encoding."""

        scale = 1 / (self.d_head ** 0.5)
        cat = h

        # content heads
        q_head_h = self.head_projection(h, 'q')
        k_head_h = self.head_projection(cat, 'k')
        v_head_h = self.head_projection(cat, 'v')

        # positional heads
        k_head_r = self.head_projection(r, 'r')

        # core attention ops
        attn_vec = self.rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r, r_w_bias, r_r_bias, attn_mask, adjoin_matrix, scale)

        attn_vec, attn_prob = attn_vec
        # post-processing
        output_h = self.post_attention(h, attn_vec)
        output_g = None
        outputs = (output_h, output_g)
        outputs = outputs + (attn_prob,)

        return outputs

    def two_stream_rel_attn(self, h, g, r, r_w_bias, r_r_bias, attn_mask_h, attn_mask_g, target_mapping,
                            adjoin_matrix):
        scale = 1 / (self.d_head ** 0.5)
        
        # content based attention score
        cat = h  # inp_k embedding: (seq_len, bsz, d_model)

        # content-based key head
        k_head_h = self.head_projection(cat, 'k')  # -> (seq_len, bsz, n_head, d_head)

        # content-based value head
        v_head_h = self.head_projection(cat, 'v')  # -> (seq_len, bsz, n_head, d_head)

        # position-based key head                   # r = pos_emb, pos_emb: (2*seq_len, bsz, d_model)
        k_head_r = self.head_projection(r, 'r')  # -> (2*seq_len, bsz, n_head, d_head)

        # h-stream
        # content-stream query head
        q_head_h = self.head_projection(h, 'q')  # -> (seq_len, bsz, n_head, d_head)

        # core attention ops
        # hˆ(m)_zt = LayerNorm(h^(m-1)_zt + RelAttn(h^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        # attn_vec_h: (seq_len, bsz, n_head, d_head)
        attn_vec_h = self.rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r, r_w_bias, r_r_bias, attn_mask_h, adjoin_matrix, scale)

        attn_vec_h, attn_prob_h = attn_vec_h
        # post-processing
        # output_h: (seq_len, bsz, d_model)
        output_h = self.post_attention(h, attn_vec_h)

        # g-stream
        # query-stream query head
        q_head_g = self.head_projection(g, 'q')

        # core attention ops
        # gˆ(m)_zt = LayerNorm(g^(m-1)_zt + RelAttn(g^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        if target_mapping is not None:
            # q_head_g = ops.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
            q_head_g = ops.matmul(q_head_g.transpose([2, 1, 3, 0]),
                                  target_mapping.transpose([2, 0, 1])).transpose([3, 1, 0, 2])
            attn_vec_g = self.rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, r_w_bias, r_r_bias, attn_mask_g, adjoin_matrix, scale)
            
            attn_vec_g, attn_prob_g = attn_vec_g
            # attn_vec_g = ops.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            attn_vec_g = ops.matmul(attn_vec_g.transpose([2, 1, 3, 0]),
                                    target_mapping.transpose([2, 1, 0])).transpose([3, 1, 0, 2])
        else:
            attn_vec_g = self.rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, r_w_bias, r_r_bias, attn_mask_g, adjoin_matrix, scale)
            attn_vec_g, attn_prob_g = attn_vec_g

        # post-processing
        output_g = self.post_attention(g, attn_vec_g)

        attn_prob = attn_prob_h, attn_prob_g
        
        outputs = (output_h, output_g)
        outputs = outputs + (attn_prob,)

        return outputs

    @staticmethod
    def _create_mask(qlen, dtype, same_length=False):
        """create causal attention mask."""
        # [[0,1,1],
        #  [0,0,1],
        #  [0,0,0]]
        attn_mask = ops.ones([qlen, qlen], type=dtype)
        mask_u = mnp.triu(attn_mask)  # Upper triangular part.
        mask_dia = mnp.tril(attn_mask) & mnp.triu(attn_mask)  # Diagonal. Figure 2(c)
        attn_mask_pad = ops.zeros([qlen, None], dtype=dtype)
        ret = mnp.concatenate([attn_mask_pad, mask_u - mask_dia], axis=1)  # [qlen, mlen]
        if same_length:
            # [[0,1,1],
            #  [1,0,1],
            #  [1,1,0]]
            mask_l = mnp.tril(attn_mask)  # Lower triangular part.
            ret = mnp.concatenate([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], axis=1)

        return ret.to(dtype=dtype)  # [qlen, qlen]

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        # sinusoid_inp: (len(pos_seq), len(inv_freq)), len(inv_freq) = d_model/2
        # sinusoid_inp = ops.einsum('i,d->id', pos_seq, inv_freq)
        sinusoid_inp = mnp.outer(pos_seq, inv_freq)
        # column concat
        # pos_emb: (len(pos_seq), d_model)
        pos_emb = mnp.concatenate([ops.sin(sinusoid_inp), ops.cos(sinusoid_inp)], axis=-1)
        # add dimension
        # pos_emb: (len(pos_seq), 1, d_model)
        pos_emb = pos_emb[:, None, :]
        if bsz is not None:
            # pos_emb: (len(pos_seq), bsz, d_model)
            pos_emb = ops.broadcast_to(pos_emb, (-1, bsz, -1))
            # temp = ms.Tensor(np.ones([pos_emb.shape[0], bsz, pos_emb.shape[2]]), dtype=ms.float32)
            # pos_emb = pos_emb.expand_as(temp)
            # del temp
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, d_model, clamp_len, attn_type,
                                     bi_data, bsz=None, dtype=None):
        """create relative positional encoding."""

        freq_seq = ops.arange(0, d_model, 2.0)  # (d_model/2, )
        if dtype is not None and dtype != ms.float32:
            freq_seq = self.cast(freq_seq, dtype)
        inv_freq = 1 / (10000 ** (freq_seq / d_model))  # (d_model/2, )

        if attn_type == 'bi':
            beg, end = klen, -qlen
        elif attn_type == 'uni':
            beg, end = klen, -1
        else:
            raise ValueError('Unknown `attn_type` {}.'.format(attn_type))

        # default: bi_data = False
        if bi_data and bsz % 2 == 0:
            fwd_pos_seq = ops.arange(beg, end, -1.0)
            bwd_pos_seq = ops.arange(-beg, -end, 1.0)

            if dtype is not None and dtype != ms.float32:
                fwd_pos_seq = self.cast(fwd_pos_seq, dtype)
                bwd_pos_seq = self.cast(bwd_pos_seq, dtype)

            if clamp_len > 0:
                fwd_pos_seq = ops.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
                bwd_pos_seq = ops.clip_by_value(bwd_pos_seq, -clamp_len, clamp_len)

            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = mnp.concatenate([fwd_pos_emb, bwd_pos_emb], axis=1)
        else:
            fwd_pos_seq = ops.arange(beg, end, -1.0)  # attn_type=='bi', (q_len+k_len, ), range(-q_len, k_len)
            if dtype is not None and dtype != ms.float32:  # attn_type=='uni', (k_len+1, ), range(-1, klen)
                fwd_pos_seq = self.cast(fwd_pos_seq, dtype)
            # default: clamp_len = -1, no clamping
            if clamp_len > 0:
                fwd_pos_seq = ops.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
            # pos_emb: (len(fwd_pos_seq), bsz, d_model)
            # default: attn_type=='bi', len(fwd_pos_seq) = q_len + k_len
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    def construct(self, inp_k, perm_mask=None, target_mapping=None, inp_q=None, input_mask=None):
        inp_k = ops.transpose(inp_k, (1, 0))                    # [seq_len, bsz]                    # [num_predict, bsz]
        if perm_mask is not None:
            perm_mask = ops.transpose(perm_mask, (1, 2, 0))           # [seq_len, seq_len, bsz]
        if target_mapping is not None:
            target_mapping = ops.transpose(target_mapping, (1, 2, 0))              # [num_predict, seq_len, bsz]
        if inp_q is not None:
            inp_q = ops.transpose(inp_q, (1, 0))                    # [seq_len, bsz]
        if input_mask is not None:
            input_mask = ops.transpose(input_mask, (1, 0))
        
        adjoin_matrix = None
        # inp_q : mask位置为1(len, bsz)
        # inp_k : token id (len, bsz)
        # target_mapping : one-hot表示，第i个预测第k个token（num_predict, len, bsz)
        # perm_mask : (len, len ,bsz)
        # input_mask : 输入的mask, (len, bsz)
        bsz = inp_k.shape[1]
        qlen = inp_k.shape[0]
        klen = qlen

        # Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self._create_mask(qlen, ms.int32, self.same_length)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

        # data mask: input mask & perm mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask  # data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:

            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]  # attn_mask = data_mask
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            # attn_mask = attn_mask > [0]
            attn_mask = attn_mask.gt(0).to(dtype=self.param_init_type)

        if attn_mask is not None:
            # 负对角矩阵，(qlen, qlen)
            non_tgt_mask = -ops.eye(qlen, qlen, t=self.param_init_type)
            # non_tgt_mask[:, :, None, None]: (qlen, qlen, 1, 1)
            # attn_mask: (qlen, qlen)
            # -> non_tgt_mask: (qlen, qlen, qlen, qlen)
            # 此处相加并调用 gt(0) 方法相当于取并，non_tgt_mask 仍是 bool 矩阵
            # 根据广播规则，此处是以 non_tgt_mask[:, :, None, None] 中的每个数值与 attn_mask 矩阵相加，
            # 最后得到矩阵中包含 qlen * qlen 个 (qlen, qlen) 矩阵
            non_tgt_mask = (attn_mask +
                            non_tgt_mask[:, :, None, None]).gt(0).to(dtype=self.param_init_type)
        else:
            non_tgt_mask = None

        # Word embedding
        lookup_table = self.embedding  # (n_token, d_model)
        word_emb_k = lookup_table(inp_k)  # (seq_len, bsz) -> (seq_len, bsz, d_model)

        if inp_q is not None:
            if target_mapping is not None:
                # mask_emb: (1, 1, d_model) -> (target_mapping.shape[0], bsz, d_model)
                word_emb_q = ops.broadcast_to(self.mask_emb,
                                              (target_mapping.shape[0], bsz, -1))
                # temp = ms.Tensor(np.ones([target_mapping.shape[0], bsz, self.mask_emb.shape[2]]), dtype=ms.float32)
                # word_emb_q = self.mask_emb.expand_as(temp)
                # del temp
            else:
                inp_q_ext = inp_q[:, :, None]
                word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
        else:
            word_emb_q = None

        # Figure 2(a), Content Stream(Original Attention), h^(0)_t = e(x_i) = e(inp_k)
        output_h = self.Dropout(word_emb_k)
        if inp_q is not None:
            # Query Stream, g^(0)_t = w
            # the first layer query stream is initialized with a trainable vector
            output_g = self.Dropout(word_emb_q)
        else:
            output_g = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(
            qlen, klen, self.d_model, self.clamp_len, self.attn_type, self.bi_data,
            bsz=bsz, dtype=self.param_init_type)
        pos_emb = self.Dropout(pos_emb)

        attentions = [] 
        hidden_states = [] 

        # Attention layers
        for i in range(self.n_layer):
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            if inp_q is not None:
                outputs = self.two_stream_rel_attn(
                    h=output_h,
                    g=output_g,
                    r=pos_emb,
                    r_w_bias=self.r_w_bias[i],
                    r_r_bias=self.r_r_bias[i],
                    attn_mask_h=non_tgt_mask,
                    attn_mask_g=attn_mask,
                    target_mapping=target_mapping,
                    adjoin_matrix=adjoin_matrix)
            else:
                outputs = self.rel_multihead_attn(
                    h=output_h,
                    r=pos_emb,
                    r_w_bias=self.r_w_bias[i],
                    r_r_bias=self.r_r_bias[i],
                    attn_mask=non_tgt_mask,
                    adjoin_matrix=adjoin_matrix)

            output_h, output_g = outputs[:2]
            if output_g is not None:
                output_g = self.ffn_layers[i](inp=output_g)

            output_h = self.ffn_layers[i](inp=output_h)
            
            attentions.append(outputs[2])
        
        hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.Dropout(output_g if output_g is not None else output_h)
        # transpose output shape [bsz, len, hidden_dim]
        output = ops.transpose(output, (1, 0, 2)) 

        
        if output_g is not None:
            temp_hs = []
            for hs in hidden_states:
                temp_hs.append(tuple(ops.transpose(h, (1, 0, 2)) for h in hs))
            hidden_states = tuple(temp_hs)
        else:
            hidden_states = tuple(ops.transpose(hs, (1, 0, 2)) for hs in hidden_states)
        
        
        if target_mapping is not None:
            temp_attn = []
            for t in attentions:
                temp_attn.append(tuple(ops.transpose(att_stream, (2, 3, 0, 1)) for att_stream in t))
            attentions = tuple(temp_attn)
        else:
            attentions = tuple(ops.transpose(t, (2, 3, 0, 1)) for t in attentions)

        # output shape [bsz, len, hidden_dim]
        return output, hidden_states, attentions
    






class FFN(nn.Cell):
    def __init__(self, d_model: int, d_inner: int, dropout: float, activation_type: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Dense(d_model, d_inner)
        self.conv2 = nn.Dense(d_inner, d_model)
        self.Dropout = nn.Dropout(keep_prob=1 - dropout)
        self.relu = nn.ReLU()
        self.activation_type = activation_type
        self.layer_norm = nn.LayerNorm((d_model,))

    @staticmethod
    def gelu(x):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
          x: float Tensor to perform activation.

        Returns:
          `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + ops.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * ops.pow(x, 3)))))
        return x * cdf

    # @ms.jit()
    def construct(self, inp: ms.Tensor) -> ms.Tensor:
        output = self.conv1(inp)
        if self.activation_type == 'relu':
            output = self.relu(output)
        elif self.activation_type == 'gelu':
            output = self.gelu(output)
        else:
            raise ValueError('Unsupported activation type {}'.format(self.activation_type))
        output = self.Dropout(output)
        output = self.conv2(output)
        output = self.Dropout(output)
        output = self.layer_norm(output + inp)
        return output


class MLP(nn.Cell):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float) -> None:
        # Mapping hidden state into vocab, that is re-building original sentence from  hidden state.
        super().__init__()
        self.d_model = d_in
        self.fc1 = dense(d_in, d_hidden, activation=nn.LeakyReLU(0.1))
        self.layer_norm = nn.LayerNorm((d_hidden,))
        self.fc2 = dense(d_hidden, d_out)
        self.dropout = nn.Dropout(keep_prob=1 - dropout)

    # @ms.jit()
    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # x: (bsz, seq_len, d_model)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.fc2(x)

        return x


class XLNetPreTrainModel(nn.Cell):
    def __init__(self, n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type):
        super(XLNetPreTrainModel, self).__init__()
        self.xlnet_layer = XLNetLayer(n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type)
        self.lm_loss = nn.Dense(d_model, n_token, has_bias=True)
    
    def construct(self, inp_k, label, perm_mask=None, target_mapping=None, target_mask=None, inp_q=None):
        output,_, _ = self.xlnet_layer(inp_k, perm_mask, target_mapping, inp_q)
        logits = self.lm_loss(output)
        logits = ops.transpose(logits, (0, 2, 1))
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits, label)
        loss = loss*target_mask
        loss = loss.sum()
        loss = loss / target_mask.sum()
        return loss
    

class XLNetForClassificationModel(nn.Cell):
    def __init__(self, n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type, num_tasks):
        super(XLNetForClassificationModel, self).__init__()
        self.xlnet_layer = XLNetLayer(n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type)
        self.logits_proj = nn.Dense(d_model, num_tasks, has_bias=True)
    
    def construct(self, inp_k, label, label_mask):
        output, _, _ = self.xlnet_layer(inp_k)
        output = output[:, 0, :]
        output = self.logits_proj(output) # (bsz, num_tasks)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        loss = loss_fn(output, label) # (bsz, num_tasks)
        loss = loss*label_mask # (bsz, num_tasks)
        loss = loss.mean()
        return loss

    

class XLNetForRegressionModel(nn.Cell):
    def __init__(self, n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type, num_tasks):
        super(XLNetForRegressionModel, self).__init__()
        self.xlnet_layer = XLNetLayer(n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type)
        self.logits_proj = nn.Dense(d_model, num_tasks, has_bias=True)
    
    def construct(self, inp_k, label, label_mask):
        output, _, _ = self.xlnet_layer(inp_k)
        output = output[:, 0, :]
        output = self.logits_proj(output)
        loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(output, label)
        loss = loss*label_mask # (bsz, num_tasks)
        loss = loss.mean()
        return loss


class XLNetPredictModel(nn.Cell):
    def __init__(self, n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type):
        super(XLNetPredictModel, self).__init__()
        self.xlnet_layer = XLNetLayer(n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type)
        self.logits_proj = nn.Dense(d_model, 1, has_bias=True)
    
    def construct(self, inp_k):
        output, hidden_states, attentions = self.xlnet_layer(inp_k)
        output = output[:, 0, :]
        output = self.logits_proj(output)
        outputs = (output, hidden_states, attentions)
        # output: (bzd, 1)
        # hidden_states[-1]: (bsz, seq_len, d_model)
        # attentions[-1]: (bsz, n_head, seq_len, seq_len)
        
        return outputs


class XLNetForEvaluateModel(nn.Cell):
    def __init__(self, n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type, task_type, num_tasks):
        super(XLNetForEvaluateModel, self).__init__()
        self.xlnet_layer = XLNetLayer(n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, param_init_type)
        self.logits_proj = nn.Dense(d_model, num_tasks, has_bias=True)
        self.task_type = task_type
        self.sigmoid = nn.Sigmoid()
    
    def construct(self, inp_k):
        output, _, _ = self.xlnet_layer(inp_k)
        output = output[:, 0, :]
        output = self.logits_proj(output) # (bsz, num_tasks)
        if self.task_type == 'classification':
            output = self.sigmoid(output)
        return output
