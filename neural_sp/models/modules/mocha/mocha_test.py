# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Chunkwise attention in MoChA at test time."""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def hard_chunkwise_attention(alpha, chunkwise_energy, mask, chunk_size, n_head_chunkwise, sharpening_factor, share_chunkwise_attention):
    """Chunkwise attention in MoChA at test time.

    Args:
        alpha (FloatTensor):                单调关注的停止点概率分布 [B, H_ma, qlen, klen]
        chunkwise_energy (FloatTensor):     chunkwise energy [B, (H_ma*)H_ca, qlen, klen]
        mask (ByteTensor):                  [B, qlen, klen]
        chunk_size (int):                   chunkwise attention的窗口size
        n_head_chunkwise (int):             chunkwise注意力的head数量
        sharpening_factor (float): sharping factor for beta calculation
        share_chunkwise_attention (int): share CA heads among MA heads
    Returns:
        beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`

    """
    bs, n_head_mono_attn, qlen, klen = alpha.size()
    assert (chunkwise_energy.size(2) == qlen) and (chunkwise_energy.size(3) == klen), (chunkwise_energy.size(), alpha.size())
    alpha = alpha.unsqueeze(2)   # `[B, H_ma, 1, qlen, klen]`
    chunkwise_energy = chunkwise_energy.unsqueeze(1)  # `[B, 1, (H_ma*)n_head_chunkwise, qlen, klen]`
    # 单调关注的停止点概率分布，要重复chunkwise-attention的head数量次
    if n_head_chunkwise > 1:
        alpha = alpha.repeat([1, 1, n_head_chunkwise, 1, 1])
    if n_head_mono_attn > 1:
        if share_chunkwise_attention:
            chunkwise_energy = chunkwise_energy.repeat([1, n_head_mono_attn, 1, 1, 1])
        else:
            chunkwise_energy = chunkwise_energy.view(bs, n_head_mono_attn, n_head_chunkwise, qlen, klen)
    # 转uint8
    mask = alpha.clone().byte()  # `[B, H_ma, n_head_chunkwise, qlen, klen]`
    for b in range(bs):
        for h in range(n_head_mono_attn):
            if alpha[b, h, 0, 0].sum() > 0:
                # 取出mono_attention的关注停止点索引
                boundary = alpha[b, h, 0, 0].nonzero()[:, -1].min().item()
                if chunk_size == -1:
                    # 可以向左←看到开头
                    mask[b, h, :, 0, 0:boundary + 1] = 1
                else:
                    mask[b, h, :, 0, max(0, boundary - chunk_size + 1):boundary + 1] = 1
    # W(key) * W(value) 在窗口/整个序列上计算并没有区别，此处直接将整个序列上计算出来的energy，按chunkwise遮盖即可
    NEG_INF = float(np.finfo(torch.tensor(0, dtype=chunkwise_energy.dtype).numpy().dtype).min)
    chunkwise_energy = chunkwise_energy.masked_fill(mask == 0, NEG_INF)
    beta = torch.softmax(chunkwise_energy, dim=-1)
    return beta.view(bs, -1, qlen, klen)
