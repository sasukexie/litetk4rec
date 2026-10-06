# -*- coding: utf-8 -*-
"""SSMRec：纯 SSM（Mamba）序列推荐基线。

用途：验证"统一机制"路线的**前置条件**。
  该路线主张不用 Transformer，让序列建模与时间建模在同一套 SSM 动力学下完成
  （SSM 离散化步长 Δ 本身就是时间步长，天然统一）。
  但这有个前提必须先验证：SSM 在**纯序列建模**上要能打平/超过 Transformer。
  若 SSM 本身就明显弱于 TFm，则"统一"后的上限会被拉低，路线不成立。

本文件是 B-0：标准 Mamba，不含任何时间模块，用于与 TimeRec(temporal_type=none)
在完全相同的配置下对比。时间感知版本（B-1，用 Δt 驱动 Δ）待 B-0 通过后再做。

注意：mamba_ssm 缺失时只置 Mamba=None 而不抛错，避免破坏其他模型的导入。
"""
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

try:
    from mamba_ssm import Mamba
except ImportError:      # pragma: no cover - 视运行环境而定
    Mamba = None


class SSMRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SSMRec, self).__init__(config, dataset)
        if Mamba is None:
            raise ImportError(
                "SSMRec requires `mamba_ssm`. Install it, or use TimeRec instead."
            )

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.d_state = config.get("d_state", 16)
        self.d_conv = config.get("d_conv", 4)
        self.expand = config.get("expand", 2)

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # 层数与 TimeRec(n_layers=2) 对齐, 保证"SSM vs Transformer"的对比公平
        # (Mamba block 内部自带 residual, 直接堆叠即可)
        self.n_layers = config.get("n_layers", 2)
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
            )
            for _ in range(self.n_layers)
        ])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        x = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        x = self.LayerNorm(x)
        x = self.dropout(x)
        y = x
        for layer in self.mamba_layers:
            y = layer(y)                                       # [B, L, H]
        return self.gather_indexes(y, item_seq_len - 1)        # [B, H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_score = torch.sum(seq_output * self.item_embedding(pos_items), dim=-1)
            neg_score = torch.sum(seq_output * self.item_embedding(neg_items), dim=-1)
            return self.loss_fct(pos_score, neg_score)
        logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(interaction[self.ITEM_ID])
        return torch.mul(seq_output, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        return torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
