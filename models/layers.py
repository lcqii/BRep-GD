"""Common layers for defining score networks."""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch_geometric.nn as graph_nn


def get_act(config):
    """Get actiuvation functions from the config file."""

    if config.model.nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == 'swish':
        return nn.SiLU()
    elif config.model.nonlinearity.lower() == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation function does not exist!')


def conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, padding=0):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, dilation=dilation,
                     padding=padding)
    return conv


# from DDPM
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class SimpleTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, prenorm=True):
        super(SimpleTransformerDecoderLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.prenorm = prenorm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 交叉注意力
        if memory_mask is not None and memory_mask.dim() == 2:
            memory_mask = memory_mask.unsqueeze(0).repeat(tgt.size(0), 1, 1)

        if self.prenorm:
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.cross_attn(tgt2, memory, memory, attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)
        else:
            tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt2))
            tgt2 = self.norm2(tgt)
        
        # 前馈网络
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        if not self.prenorm:
            tgt = self.norm2(tgt)
        return tgt

class ModifiedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, **kwargs):
        super().__init__(d_model, nhead, **kwargs)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: (num*num, batch, dim)
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Returns:
            Modified output tensor.
        """
        # 获取形状信息
        num = int(tgt.size(0) ** 0.5)  # num*num=batch_size
        bsz = tgt.size(1)              # batch size
        embed_dim = tgt.size(2)        # dimension

        # 重塑并计算平均值来减少复杂度
        tgt_reshaped = tgt.unflatten(0, torch.Size([num, num])).mean(1)

        # 自注意力层
        if self.norm_first:
            # norm_first 模式
            tgt = tgt + self._sa_block(self.norm1(tgt_reshaped), tgt_mask, tgt_key_padding_mask, num, bsz, embed_dim)
            tgt = tgt + self._mha_block(self.norm2(tgt), memory, memory_mask, memory_key_padding_mask)
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            # 默认模式
            tgt = self.norm1(tgt + self._sa_block(tgt_reshaped, tgt_mask, tgt_key_padding_mask, num, bsz, embed_dim))
            tgt = self.norm2(tgt + self._mha_block(tgt, memory, memory_mask, memory_key_padding_mask))
            tgt = self.norm3(tgt + self._ff_block(tgt))

        return tgt

    def _sa_block(self, x, attn_mask, key_padding_mask, num, bsz, embed_dim):
        # 自注意力层计算
        satt_out = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )[0]

        # 扩展satt_out的维度并进行残差连接
        satt_out = (satt_out.unsqueeze(0).expand(num, num, bsz, embed_dim) + 
                    satt_out.unsqueeze(1).expand(num, num, bsz, embed_dim)).flatten(0, 1) / 2

        return self.dropout1(satt_out)


class CurvTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, **kwargs):
        super().__init__(d_model, nhead, **kwargs)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None,tgt_is_causal=False, memory_is_causal=False):
        """
        Args:
            tgt: (num*num, batch, dim) ,edges feature
            memory: (num, batch, dim)the sequence from the last layer of the encoder (required).,nodes feature
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Returns:
            Modified output tensor.
        """
        # 获取形状信息
        num = int(tgt.size(0) ** 0.5)  # num*num=batch_size
        bsz = tgt.size(1)              # batch size
        embed_dim = tgt.size(2)        # dimension

        # 重塑并计算平均值来减少复杂度
        tgt_reshaped = tgt.unflatten(0, torch.Size([num, num])).flatten(1, 2)

        # 自注意力层
        if self.norm_first:
            # norm_first 模式
            tgt = tgt + self._sa_block(self.norm1(tgt_reshaped), tgt_mask, tgt_key_padding_mask, num, bsz, embed_dim)
            tgt = tgt + self._mha_block(self.norm2(tgt), memory, memory_mask, memory_key_padding_mask)
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            # 默认模式
            tgt = self.norm1(tgt + self._sa_block(tgt_reshaped, tgt_mask, tgt_key_padding_mask, num, bsz, embed_dim))
            tgt = self.norm2(tgt + self._mha_block(tgt, memory, memory_mask, memory_key_padding_mask))
            tgt = self.norm3(tgt + self._ff_block(tgt))

        return tgt

    def _sa_block(self, x, attn_mask, key_padding_mask, num, bsz, embed_dim):
        # 自注意力层计算
        valid_sequences = ~(key_padding_mask.all(dim=1))
        valid_x=x[:,valid_sequences]
        x[:,valid_sequences] = self.self_attn(
            valid_x, valid_x, valid_x, attn_mask=attn_mask, key_padding_mask=key_padding_mask[valid_sequences]
        )[0].float()
        satt_out=x.unflatten(1, (num, bsz))
        satt_out=(satt_out+satt_out.transpose(0,1))/2
        satt_out=satt_out.flatten(0, 1)
        return self.dropout1(satt_out)

    def _mha_block(self, x, mem,attn_mask, key_padding_mask):
        num,batch,dim=mem.shape
        x=x.flatten(0,1).unsqueeze(0)
        mem=torch.cat([
            mem.unsqueeze(0).unsqueeze(1).expand(1,num, num, batch, dim),
            mem.unsqueeze(0).unsqueeze(2).expand(1,num, num, batch, dim)
        ], dim=0).flatten(1, 2).flatten(1, 2)
        with torch.no_grad():
            key_padding_mask=key_padding_mask.unsqueeze(1).expand(batch, num,num) |key_padding_mask.unsqueeze(2).expand(batch, num,num) 
            key_padding_mask=key_padding_mask.flatten(1,2).transpose(0,1).flatten(0,1)
            valid_sequences=~key_padding_mask
        valid_x=x[:,valid_sequences]
        valid_m=mem[:,valid_sequences]
        x[:,valid_sequences] = self.multihead_attn(valid_x, valid_m, valid_m,
                                attn_mask=attn_mask,
                                key_padding_mask=None,
                                need_weights=False)[0].float()
        x=x.squeeze(0).unflatten(0, (num*num, batch))
        return self.dropout2(x)
        


class CurvTransformerDecoderLayerFine(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, **kwargs):
        super().__init__(d_model, nhead, **kwargs)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None,tgt_is_causal=False, memory_is_causal=False):
        """
        Args:
            tgt: (num*num, batch, dim) ,edges feature
            memory: (num, batch, dim)the sequence from the last layer of the encoder (required).,nodes feature
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Returns:
            Modified output tensor.
        """
        # 获取形状信息
        num = int(tgt.size(0) ** 0.5)  # num*num=batch_size
        bsz = tgt.size(1)              # batch size
        embed_dim = tgt.size(2)        # dimension

        # 重塑并计算平均值来减少复杂度
        tgt_reshaped = tgt.unflatten(0, torch.Size([num, num])).flatten(1, 2)

        # 自注意力层
        if self.norm_first:
            # norm_first 模式
            tgt = tgt + self._sa_block(self.norm1(tgt_reshaped), tgt_mask, tgt_key_padding_mask, num, bsz, embed_dim)
            tgt = tgt + self._mha_block(self.norm2(tgt), memory, num,memory_mask, memory_key_padding_mask)
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            # 默认模式
            tgt = self.norm1(tgt + self._sa_block(tgt_reshaped, tgt_mask, tgt_key_padding_mask, num, bsz, embed_dim))
            tgt = self.norm2(tgt + self._mha_block(tgt, memory, memory_mask, memory_key_padding_mask))
            tgt = self.norm3(tgt + self._ff_block(tgt))

        return tgt

    def _sa_block(self, x, attn_mask, key_padding_mask, num, bsz, embed_dim):
        # 自注意力层计算
        valid_sequences = ~(key_padding_mask.all(dim=1))
        valid_x=x[:,valid_sequences]
        x[:,valid_sequences] = self.self_attn(
            valid_x, valid_x, valid_x, attn_mask=attn_mask, key_padding_mask=key_padding_mask[valid_sequences]
        )[0].float()
        satt_out=x.unflatten(1, (num, bsz))
        satt_out=(satt_out+satt_out.transpose(0,1))/2
        satt_out=satt_out.flatten(0, 1)
        return self.dropout1(satt_out)

    def _mha_block(self, x, mem,num,attn_mask, key_padding_mask):
        _,batch,_=x.shape
        x=x.unflatten(0, (num, num)).transpose(0,2).flatten(0,1).flatten(0,1).unsqueeze(0)
        x[:,key_padding_mask] = self.multihead_attn(x[:,key_padding_mask], mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=None,
                                need_weights=False)[0].float()
        x=x.squeeze(0).unflatten(0, (num*num, batch)).unflatten(0, (num,num))
        x=(x+x.transpose(0,1))/2
        x=x.flatten(0, 1)
        return self.dropout2(x)
        
