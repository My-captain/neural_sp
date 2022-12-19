# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-head attention (MHA) layer."""

import logging
import math
import numpy as np
import torch
import torch.nn as nn

from neural_sp.models.modules.headdrop import headdrop

logger = logging.getLogger(__name__)


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention (MHA) layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        odim: (int) dimension of output
        n_heads (int): number of heads
        dropout (float): dropout probability for attention weights
        dropout_head (float): HeadDrop probability
        atype (str): type of attention mechanism
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        xl_like: dummy argument for compatibility with relative MHA
        clamp_len: dummy

    """

    def __init__(self, kdim, qdim, adim, odim, n_heads, dropout, dropout_head=0.,
                 atype='scaled_dot', bias=True, param_init='',
                 xl_like=False, clamp_len=-1):

        super().__init__()

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.reset()

        self.dropout_attn = nn.Dropout(p=dropout)
        self.dropout_head = dropout_head

        if atype == 'scaled_dot':
            # for Transformer
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
        elif atype == 'add':
            # for LAS
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
            self.v = nn.Linear(adim, n_heads, bias=bias)
        else:
            raise NotImplementedError(atype)

        self.w_out = nn.Linear(adim, odim, bias=bias)

        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' %
                    self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_value.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.)

    def reset(self):
        self.key = None
        self.value = None
        self.mask = None

    def forward(self, key, value, query, mask, aw_prev=None, aw_lower=None,
                cache=False, mode='', trigger_points=None, eps_wait=-1, streaming=False):
        """
        多头注意力前向传播
        Args:
            key:        [B, klen, kdim]
            value:      [B, klen, kdim]
            query:      [B, qlen, qdim]
            mask:       [B, qlen, klen]
            aw_prev:    冗余接口
            aw_lower:   冗余接口
            cache:      （用于流式场景）cache key, value, and mask
            mode:       冗余接口for MoChA/MMA
            trigger_points:冗余接口for MoChA/MMA
            eps_wait:    冗余接口for MMA
            streaming:   （用于流式场景）冗余接口
        Returns:
            cv (FloatTensor):           经过attention的tokens [B, qlen, vdim]
            attn_weight (FloatTensor):  注意力权重矩阵         [B, Head, qlen, klen]
            attn_state (dict): 冗余接口
        """
        bs, klen = key.size()[: 2]
        qlen = query.size(1)
        attn_state = {}

        # 对query、key、value进行线性变换
        if self.key is None or not cache:
            self.key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen, H, d_k]`
            self.value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen, H, d_k]`
            if mask is not None:
                self.mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
                mask_size = (bs, qlen, klen, self.n_heads)
                assert self.mask.size() == mask_size, (self.mask.size(), mask_size)
            else:
                self.mask = None
        key = self.key
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)  # `[B, qlen, H, d_k]`

        # 计算energy： `[B, qlen, klen, H]`
        attn_energy = None
        if self.atype == 'scaled_dot':
            attn_energy = torch.einsum("bihd,bjhd->bijh", (query, key)) / self.scale
        elif self.atype == 'add':
            attn_energy = self.v(torch.tanh(key[:, None] + query[:, :, None]).view(bs, qlen, klen, -1))

        # 对energy应用Mask
        if self.mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=attn_energy.dtype).numpy().dtype).min)
            # `[B, qlen, klen, H]`
            attn_energy = attn_energy.masked_fill_(self.mask == 0, NEG_INF)
        # 计算注意力权重：沿着key做softmax
        attn_weight = torch.softmax(attn_energy, dim=2)

        # 对注意力权重矩阵应用Dropout
        attn_weight = self.dropout_attn(attn_weight)
        attn_weight_masked = attn_weight.clone()

        # 应用(HeadDrop)：独立地遮盖每个head
        if self.dropout_head > 0 and self.training:
            attn_weight_masked = attn_weight_masked.permute(0, 3, 1, 2)
            # `[B, H, qlen, klen]`
            attn_weight_masked = headdrop(attn_weight_masked, self.n_heads, self.dropout_head)
            attn_weight_masked = attn_weight_masked.permute(0, 2, 3, 1)

        # 最终attention得到的tokens
        cv = torch.einsum("bijh,bjhd->bihd", (attn_weight_masked, self.value))  # `[B, qlen, H, d_k]`
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)              # `[B, qlen, H * d_k]`
        # 线性变换
        cv = self.w_out(cv)
        attn_weight = attn_weight.permute(0, 3, 1, 2)           # `[B, H, qlen, klen]`

        return cv, attn_weight, attn_state
