# -*- coding: utf-8 -*-
"""MoCo4SRec：动量对比学习序列推荐（ESWA 2023）。

忠实复现原论文核心机制（https://github.com/ 对应源码）：
  1. 双编码器：query encoder（item_encoder）+ key encoder（encoder_k）
  2. 动量更新：encoder_k 以 m=0.999 的 EMA 更新，参数冻结（requires_grad=False）
  3. 队列负样本：维护 dim×K 的 FIFO 队列，突破 batch 限制
  4. 投影头：整个序列展平 → 两层 MLP + BatchNorm（原文做法，非取最后位置）
  5. phi 假负样本过滤：负样本 logit > phi 时 mask 掉（避免队列里的假负样本）
  6. 序列增强：Crop / Mask / Reorder（原论文还有 Insert/Substitute 需相似度模型，
     这里实现前三种无依赖的增强）

与原论文的差异（仅为适配 RecBole 框架，不改变方法本质）：
  - 主损失用 CE（全排序），原论文用 BCE + 采样；评测协议与 TimeRec 一致
  - Insert/Substitute 增强依赖 item 相似度模型，此处省略（用 Crop/Mask/Reorder）
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class MoCo4SRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MoCo4SRec, self).__init__(config, dataset)

        # 基础配置
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # MoCo 配置
        self.moco_dim = config.get("moco_dim", 128)
        self.queue_size = config.get("queue_size", 4096)
        self.moco_momentum = config.get("moco_momentum", 0.999)
        self.temperature = config.get("ssl_temperature", 0.07)
        self.phi = config.get("moco_phi", 0.0)          # 假负样本过滤阈值，0=不过滤
        self.ssl_weight = config.get("ssl_weight", 0.1)
        self.moco_aug = config.get("moco_aug", "reorder")  # crop / mask / reorder

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # query encoder（可训练）
        self.item_encoder = TransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps,
        )
        # key encoder（动量更新，冻结梯度）
        self.encoder_k = TransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps,
        )
        self.encoder_k.load_state_dict(self.item_encoder.state_dict())
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        # 投影头：整个序列展平（L*H）→ MLP + BatchNorm（忠实原文）
        input_dim = self.max_seq_length * self.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(input_dim, self.moco_dim, bias=False),
            nn.BatchNorm1d(self.moco_dim, eps=1e-12),
            nn.ReLU(inplace=True),
            nn.Linear(self.moco_dim, self.moco_dim, bias=False),
            nn.BatchNorm1d(self.moco_dim, eps=1e-12),
        )

        # 负样本队列
        self.register_buffer(
            "queue", F.normalize(torch.randn(self.moco_dim, self.queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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

    # ============ 序列增强（原论文 Crop/Mask/Reorder）============

    def _augment(self, item_seq, item_seq_len):
        if self.moco_aug == "crop":
            return self._crop(item_seq, item_seq_len)
        if self.moco_aug == "mask":
            return self._mask(item_seq, item_seq_len)
        if self.moco_aug == "reorder":
            return self._reorder(item_seq, item_seq_len)
        raise NotImplementedError("moco_aug=%s" % self.moco_aug)

    def _crop(self, item_seq, item_seq_len, tao=0.2):
        """随机裁剪出连续子序列并左对齐。"""
        B, L = item_seq.shape
        out = torch.zeros_like(item_seq)
        new_len = torch.zeros_like(item_seq_len)
        for i in range(B):
            length = int(item_seq_len[i].item())
            sub = max(int(tao * length), 1)
            start = torch.randint(0, length - sub + 1, (1,)).item()
            out[i, :sub] = item_seq[i, start:start + sub]
            new_len[i] = sub
        return out, new_len

    def _mask(self, item_seq, item_seq_len, gamma=0.7):
        """随机掩码部分物品为 0。"""
        B, L = item_seq.shape
        out = item_seq.clone()
        for i in range(B):
            length = int(item_seq_len[i].item())
            mask_nums = max(int(gamma * length), 1)
            idx = torch.randperm(length)[:mask_nums]
            out[i, idx] = 0
        return out, item_seq_len

    def _reorder(self, item_seq, item_seq_len, beta=0.2):
        """随机打乱一段连续子序列。"""
        B, L = item_seq.shape
        out = item_seq.clone()
        for i in range(B):
            length = int(item_seq_len[i].item())
            sub = max(int(beta * length), 2)
            if sub >= length:
                continue
            start = torch.randint(0, length - sub + 1, (1,)).item()
            perm = torch.randperm(sub)
            out[i, start:start + sub] = item_seq[i, start + perm]
        return out, item_seq_len

    # ============ 编码 ============

    def _encode(self, item_seq, encoder):
        """编码并取最后有效位置，返回 [B, H]（推荐用）。"""
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        x = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        x = self.LayerNorm(x)
        x = self.dropout(x)
        mask = self.get_attention_mask(item_seq)
        out = encoder(x, mask, output_all_encoded_layers=True)[-1]
        return out

    def _encode_seq(self, item_seq, encoder):
        """编码整条序列，返回 [B, L, H]（对比学习用，忠实原文的"展平整条序列"）。"""
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        x = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        x = self.LayerNorm(x)
        x = self.dropout(x)
        mask = self.get_attention_mask(item_seq)
        return encoder(x, mask, output_all_encoded_layers=True)[-1]  # [B, L, H]

    # ============ MoCo 机制 ============

    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.item_encoder.parameters(), self.encoder_k.parameters()):
            pk.data = pk.data * self.moco_momentum + pq.data * (1.0 - self.moco_momentum)

    @torch.no_grad()
    def _enqueue(self, keys):
        B = keys.size(0)
        ptr = int(self.queue_ptr)
        if ptr + B <= self.queue_size:
            self.queue[:, ptr:ptr + B] = keys.T
            self.queue_ptr[0] = (ptr + B) % self.queue_size
        else:
            room = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:room].T
            self.queue[:, :B - room] = keys[room:].T
            self.queue_ptr[0] = B - room

    def _moco_loss(self, item_seq, item_seq_len):
        """对比损失（原论文 moco_trans_encoder 的 InfoNCE + phi 过滤）。"""
        # 两个增强视图
        seq_q, len_q = self._augment(item_seq, item_seq_len)
        seq_k, len_k = self._augment(item_seq, item_seq_len)

        # query
        q_seq = self._encode_seq(seq_q, self.item_encoder)          # [B, L, H]
        q = q_seq.reshape(q_seq.size(0), -1)                        # [B, L*H]
        q = self.projection(q)
        q = F.normalize(q, dim=1)

        # key（动量 + 无梯度）
        with torch.no_grad():
            self._momentum_update()
            k_seq = self._encode_seq(seq_k, self.encoder_k)
            k = k_seq.reshape(k_seq.size(0), -1)
            k = self.projection(k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)         # [B, 1]
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())  # [B, K]

        # phi 假负样本过滤
        if self.phi > 0:
            weights = torch.where(l_neg > self.phi, torch.tensor(0.0, device=q.device),
                                  torch.tensor(1.0, device=q.device))
            l_neg = l_neg * weights

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=q.device)

        self._enqueue(k)
        return self.loss_fct(logits, labels)

    # ============ 前向 / 损失 ============

    def forward(self, item_seq, item_seq_len):
        out = self._encode(item_seq, self.item_encoder)
        return self.gather_indexes(out, item_seq_len - 1)  # [B, H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_score = torch.sum(seq_output * self.item_embedding(pos_items), dim=-1)
            neg_score = torch.sum(seq_output * self.item_embedding(neg_items), dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # 对比损失
        if self.ssl_weight > 0:
            loss = loss + self.ssl_weight * self._moco_loss(item_seq, item_seq_len)
        return loss

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
