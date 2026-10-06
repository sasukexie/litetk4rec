# -*- coding: utf-8 -*-
"""
LiteTK4Rec: Lite Temporal Kernel for Recommendation

骨架：**标准 Transformer（SASRec 同构）**，在其注意力分数上叠加「可插拔时间偏置」。
通过 `temporal_type` 配置选择，不再分散成多个模型文件。

配置：
  temporal_type:
    none      -> 纯 SASRec（无时间项）—— 即 SASRec 本身，用于基线对齐
    interval  -> 相邻时间间隔分桶嵌入，**加在输入上**（TiSASRec 风格输入式变体）
    periodic  -> 小时/周/月正弦周期编码（加在输入上）——实测无效，保留供复现
    adaptive  -> 间隔编码 + 可靠性门控（按序列时间统计特征决定注入量）
    zpt       -> 零参数时间偏置：解析式衰减 b=-log1p(dt/tau) 作为注意力偏置，
                 整个时间模块仅 2 个可学习标量（tau 与门控 g）
    session   -> 会话感知 v1：软切分 + **强制抑制**跨会话（方向写死，已被证伪）
    session2  -> 会话感知 v2：三个"跨会话层数"标量，**方向可学习**
    session3  -> v1 主方法：每 head 间隔嵌入（TiSASRec 表达力）
                 + 会话层级先验（初始化 0 ⇒ 起点近似 TiSASRec）
                 ⚠️ 已于 2026-09-17 证实：公平配置下 ≈ TiSASRec（0.2458 vs 0.2459）
    session4  -> 连续多尺度衰减（每 head 一个 tau_h，可解释）替代离散查表
    ckernel   -> **v2 主方法（连续可学时间核）**：
                 多尺度衰减 + 对数尺度周期 + 零间隔专属 + per-head 会话层级。
                 全部系数初始化 0 ⇒ 起点 bias ≡ 0，与 none **逐位一致**
                 ⇒ SASRec 是它的特例，增益只能来自时间核本身（归因最干净）。
                 是 session3 / zpt / session4 / TiSASRec 查表的严格推广，
                 且时间模块仅 95 参数（session3 为 261，TiSASRec 查表为 32×H）。
  time_mode:
    normal    -> 原始时间戳
    shuffle   -> 用户内置换时间戳（破坏时序语义，保留边际分布）
    constant  -> 时间戳全置 0（时间编码退化为常数偏置）

置换对照语义：train/test 均按 time_mode 处理时间戳，保证分布一致；
若 shuffle 后不掉点，则说明该模块的增益不来自时间信号本身。

新任务请一律使用 `--model=LiteTK4Rec`。
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeedForward, TransformerEncoder
from recbole.model.loss import BPRLoss

class BiasMultiHeadAttention(nn.Module):
    """支持外部 attention bias 的多头自注意力。

    与 RecBole 原生 MultiHeadAttention 的唯一区别: forward 额外接受 attn_bias
    ([B, 1, L, L]), 加到 softmax 之前的注意力分数上(与 attention_mask 同级)。
    attn_bias=None 时与原实现逐位一致, 保证各对照组之间不引入骨架差异。
    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob,
                 attn_dropout_prob, layer_norm_eps):
        super(BiasMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (hidden_size, n_heads)
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        # 与 RecBole 原生实现保持一致: 只做 view, 不做 permute
        # (permute 在 forward 里做), 否则张量布局会错乱
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, attn_bias=None):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attn_bias is not None:
            attention_scores = attention_scores + attn_bias
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BiasTransformerLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, intermediate_size,
                 hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(BiasTransformerLayer, self).__init__()
        self.multi_head_attention = BiasMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(
            hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask, attn_bias=None):
        attention_output = self.multi_head_attention(
            hidden_states, attention_mask, attn_bias)
        return self.feed_forward(attention_output)

class BiasTransformerEncoder(nn.Module):
    """支持逐层透传 attn_bias 的 TransformerEncoder。"""

    def __init__(self, n_layers, n_heads, hidden_size, inner_size,
                 hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(BiasTransformerEncoder, self).__init__()
        layer = BiasTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob,
            attn_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, attn_bias=None,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, attn_bias)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class PeriodicTimeEncoder(nn.Module):
    """周期性时间编码：hour / day-of-week / week-of-month 正弦 -> 投影 -> LayerNorm。

    用于验证核心假设之一：数据实测周期强度极低（全部数据集 <= 0.064）时，
    周期性编码究竟是带来增益，还是纯噪声注入。
    """

    def __init__(self, hidden_size):
        super(PeriodicTimeEncoder, self).__init__()
        self.hidden_size = hidden_size
        period_dim = hidden_size // 4
        self.period_embeddings = nn.ModuleDict({
            "hour": nn.Linear(2, period_dim),
            "day": nn.Linear(2, period_dim),
            "week": nn.Linear(2, period_dim),
        })
        self.fusion = nn.Linear(period_dim * 3, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, timestamps):
        hour = (timestamps % 86400) / 3600
        day_of_week = (timestamps // 86400) % 7
        week_of_month = ((timestamps // 86400) % 30) / 7

        hour_feat = torch.stack(
            [torch.sin(2 * math.pi * hour / 24), torch.cos(2 * math.pi * hour / 24)], dim=-1)
        day_feat = torch.stack(
            [torch.sin(2 * math.pi * day_of_week / 7), torch.cos(2 * math.pi * day_of_week / 7)], dim=-1)
        week_feat = torch.stack(
            [torch.sin(2 * math.pi * week_of_month / 4), torch.cos(2 * math.pi * week_of_month / 4)], dim=-1)

        h = self.period_embeddings["hour"](hour_feat)
        d = self.period_embeddings["day"](day_feat)
        w = self.period_embeddings["week"](week_feat)
        return self.layer_norm(self.fusion(torch.cat([h, d, w], dim=-1)))

class LiteTK4Rec(SequentialRecommender):
    """Lite Temporal Kernel for Recommendation (LiteTK4Rec).

    标准 Transformer 骨架 + 可插拔时间偏置；所有变体由 `temporal_type` 选择。
    时间模块总参数在 session3/32桶 配置下为 261（256 间隔嵌入 + 3 会话标量 + 2 阈值）。
    """

    def __init__(self, config, dataset):
        super(LiteTK4Rec, self).__init__(config, dataset)

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

        # 时间模块配置
        self.temporal_type = config.get("temporal_type", "none")
        self.time_mode = config.get("time_mode", "normal")

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # 时间编码模块（可插拔）
        if self.temporal_type == "interval":
            self.n_interval_bins = config.get("n_interval_bins", 32)
            self.max_interval = config.get("max_interval", 30 * 24 * 3600)
            self.interval_zero_split = int(config.get("interval_zero_split", 0))
            self.interval_embedding = nn.Embedding(self.n_interval_bins, self.hidden_size)
        elif self.temporal_type == "periodic":
            self.periodic_encoder = PeriodicTimeEncoder(self.hidden_size)
        elif self.temporal_type == "adaptive":
            # 时间编码与 interval 相同，但注入量由可靠性门控决定
            self.n_interval_bins = config.get("n_interval_bins", 32)
            self.max_interval = config.get("max_interval", 30 * 24 * 3600)
            self.interval_zero_split = int(config.get("interval_zero_split", 0))
            self.interval_embedding = nn.Embedding(self.n_interval_bins, self.hidden_size)
            # 可靠性门控：由"序列级时间统计特征"预测该样本的时间信号是否可信
            # 统计特征维度=4: [非零间隔比例, log间隔均值, log间隔标准差, 序列长度]
            self.gate = nn.Sequential(
                nn.Linear(4, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, 1),
                nn.Sigmoid(),
            )
        elif self.temporal_type == "zpt":
            # 零参数时间偏置 (Zero-Param Temporal Bias):
            # 解析式衰减 -log1p(dt/tau), 不使用任何 embedding / MLP,
            # 仅引入 2 个可学习标量: 尺度 tau 与门控 g。
            # 目的: 把"参数效应"压到近乎为零, 只保留纯粹的时间信号通道;
            # 若时间无用, g -> 0, 模型自动退化为纯 Transformer(零伤害)。
            self.log_tau = nn.Parameter(torch.tensor(0.0))
            self.gate_raw = nn.Parameter(torch.tensor(0.0))
        elif self.temporal_type == "session":
            # 会话感知建模: 由时间间隔自动切分会话, 跨会话施加注意力抑制。
            # 动机: 时间信息的价值是"结构"而非"特征"。Transformer 位置编码
            # 结构性缺失"边界"概念, 这里补上, 同样只用可学习标量(3 个):
            #   tau   : 切分阈值(对数秒尺度, 初值 ~1 小时)
            #   s     : 软边界温度(保证切分可微)
            #   alpha : 跨会话抑制强度
            self.sess_log_tau = nn.Parameter(torch.tensor(8.0))
            self.sess_log_s = nn.Parameter(torch.tensor(0.0))
            self.sess_log_alpha = nn.Parameter(torch.tensor(0.0))
        elif self.temporal_type == "session2":
            # 会话感知 v2: 切分方式与 v1 相同, 但偏置方向不再强制为负。
            # v1 的问题: b = -alpha*sep (alpha>0) 硬性抑制跨会话; 而完整历史序列
            # 推荐中, 跨会话的长期偏好恰恰是重要信号, 实测 normal 反而低于 shuffle。
            # v2 给"同会话 / 跨1层 / 跨2层+"各一个独立可学标量(可正可负),
            # 只负责告诉模型"边界在哪", 怎么用交给数据决定。
            self.sess_log_tau = nn.Parameter(torch.tensor(8.0))
            self.sess_log_s = nn.Parameter(torch.tensor(0.0))
            self.sess_w = nn.Parameter(torch.zeros(3))
        elif self.temporal_type == "session3":
            # session3 = TiSASRec 的表达力 + 会话结构先验 + 方向自适应
            #
            # 动机: 实测 TiSASRec 略优于 session2(0.2158 vs 0.2149 @ep4)。
            # 它赢在"每 head 独立的间隔嵌入"(32x2=64 参数), 而 session2 只有 3 个标量,
            # 表达力不足。故此处把两者结合:
            #   bias = interval_emb(dt)  [每 head, 表达力, 同 TiSASRec]
            #        + w[会话层级]        [结构先验, 可正可负, 我们独有]
            # 关键: 当 w -> 0 时 session3 退化为 TiSASRec, 即 TiSASRec 是它的特例。
            # 因此反超的部分完全来自会话先验, 归因干净。
            self.n_interval_bins = config.get("n_interval_bins", 32)
            self.max_interval = config.get("max_interval", 30 * 24 * 3600)
            self.interval_zero_split = int(config.get("interval_zero_split", 0))
            # per_head_interval=1(默认): 每个 head 独立间隔嵌入(同 TiSASRec, 表达力强)
            # per_head_interval=0: 所有 head 共享一个间隔嵌入
            #   -> 消融用, 用于验证"per-head 独立"这一设计对性能的贡献
            self.per_head_interval = config.get("per_head_interval", 1)
            _emb_dim = self.n_heads if self.per_head_interval else 1
            self.sess3_time_emb = nn.Embedding(self.n_interval_bins, _emb_dim)
            self.sess3_w = nn.Parameter(torch.zeros(3))     # 初始化 0 -> 从 TiSASRec 出发
            self.sess_log_tau = nn.Parameter(torch.tensor(8.0))
            self.sess_log_s = nn.Parameter(torch.tensor(0.0))
        elif self.temporal_type == "session4":
            # session4: 用"连续多尺度衰减"替代 TiSASRec 的"离散间隔查表"
            #
            # TiSASRec 强的本质是"每个 head 能拟合不同的时间衰减曲线",
            # 而不是"用 32 个桶查表"。查表离散、参数多(32xH)、不可解释。
            # 这里改用连续参数化:
            #     b_ij^(h) = -g_h * log1p(|t_i-t_j| / tau_h)
            # 每个 head 仅 2 个参数(尺度 tau_h、强度 g_h), 连续可外推,
            # 且 tau_h 可直接报告为"该头关注的时间跨度", 可解释性强。
            # 再加会话层级先验(3 个标量), 总参数 2H+3+2。
            self.sess4_log_tau_h = nn.Parameter(
                torch.full((self.n_heads,), 8.0))    # 每 head 时间尺度(对数秒)
            self.sess4_gate_h = nn.Parameter(
                torch.zeros(self.n_heads))           # 每 head 强度, 0 起点
            self.sess4_w = nn.Parameter(torch.zeros(3))
            self.sess_log_tau = nn.Parameter(torch.tensor(8.0))
            self.sess_log_s = nn.Parameter(torch.tensor(0.0))
        elif self.temporal_type == "ckernel":
            # ckernel = 连续可学时间核 (Continuous learnable temporal Kernel)
            #
            # 动机（来自跨线诊断的结论）:
            #   1) 现有实现把时间注入做成"32 桶离散查表"(session3) 或"单尺度衰减"(zpt/session4)。
            #      查表不可外推、尺度全局共享; 单尺度只能表达一条衰减曲线。
            #   2) 实测这批数据的相邻间隔存在"点质量": ml-1m 中位 Δt = 0 秒、>50% 的 pair 同秒,
            #      mind 更极端(98.7% <= 1 分钟)。查表把"同秒 pair"和"1 秒 pair"混在低桶,
            #      但"同秒"其实是同一次共现(强信号), 不是"时间最小"。
            #
            # 设计: 把时间核写成四项之和
            #   u       = log1p(Δt) / log1p(norm)                      对数量纲, 跨数据集自动可比
            #   decay_k = exp(-r_k·u),  r_k = softplus(rho_k) > 0        K 个尺度, 保证单调递减
            #   per(head) b_h = Σ_k a_kh·decay_k                       多尺度衰减(近期性)
            #                 + Σ_m [c_mh·cos(w_m·u) + d_mh·sin(w_m·u)] 对数尺度上的周期
            #                 + z_h·1[Δt=0]                            零间隔(同批次共现)专属
            #                 + Σ_r e_rh·1[跨 r 层会话边界]             会话层级, per-head
            #
            # 关键性质:
            #   * 所有系数初始化为 0 => 起点 bias ≡ 0, 与 temporal_type=none 逐位一致。
            #     即"SASRec 是它的特例", 任何增益都只能来自时间核学到的东西, 归因干净。
            #   * 是 session3 / zpt / session4 / TiSASRec 查表的严格推广。
            #   * r_k / w_m / 系数均可读 => 可画出每个 head 的时间核曲线(可解释性证据)。
            self.ck_K = int(config.get("ckernel_K", 3))
            self.ck_M = int(config.get("ckernel_M", 2))
            self.ck_norm = float(config.get("ckernel_norm", 30 * 24 * 3600))
            self.ck_log_norm = math.log1p(self.ck_norm)
            # 基函数的尺度/频率全 head 共享(降参), 系数 per-head(保表达力)
            self.ck_rho = nn.Parameter(torch.linspace(-1.0, 3.0, self.ck_K))
            self.ck_omega = nn.Parameter(torch.linspace(1.0, 4.0, self.ck_M))
            self.ck_a = nn.Parameter(torch.zeros(self.n_heads, self.ck_K))
            self.ck_c = nn.Parameter(torch.zeros(self.n_heads, self.ck_M))
            self.ck_d = nn.Parameter(torch.zeros(self.n_heads, self.ck_M))
            self.ck_z = nn.Parameter(torch.zeros(self.n_heads))
            self.ck_w = nn.Parameter(torch.zeros(self.n_heads, 3))
            self.sess_log_tau = nn.Parameter(torch.tensor(8.0))
            self.sess_log_s = nn.Parameter(torch.tensor(0.0))

        # 统一用支持 bias 的 encoder: attn_bias=None 时与原 TransformerEncoder
        # 完全等价, 避免"换骨架"成为对照组之间的混淆变量
        self.trm_encoder = BiasTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # ==================== SSL（自监督辅助损失，全配置驱动）====================
        # 设计原则：时间模块(temporal_type)与 SSL(ssl_method)完全正交，
        # 任意组合只需改 yaml，不碰代码，避免"改来改去有残渣/误跑"。
        #   ssl_method : none / infonce / moco / byol / barlow
        #   ssl_aug    : shuffle(时间置换) / dropout / crop / session
        self.ssl_method = config.get("ssl_method", "none")
        self.ssl_aug = config.get("ssl_aug", "shuffle")
        self.ssl_weight = config.get("ssl_weight", 0.0)
        self.ssl_temperature = config.get("ssl_temperature", 0.07)
        self.ssl_proj_dim = config.get("ssl_proj_dim", None)
        self.moco_queue_size = config.get("moco_queue_size", 4096)
        self.moco_momentum = config.get("moco_momentum", 0.999)
        # 保时增强(item_mask)的 item 替换比例, 仅 ssl_aug=item_mask 时生效
        self.ssl_mask_ratio = config.get("ssl_mask_ratio", 0.2)

        # 可选投影头：把表示映射到对比空间
        if self.ssl_proj_dim:
            self.ssl_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.ssl_proj_dim),
                nn.ReLU(),
                nn.Linear(self.ssl_proj_dim, self.ssl_proj_dim),
            )
        else:
            self.ssl_proj = None

        # MoCo 负样本队列（仅 ssl_method=moco 使用，惰性注册）
        if self.ssl_method == "moco":
            proj_dim = self.ssl_proj_dim or self.hidden_size
            self.register_buffer(
                "ssl_queue",
                F.normalize(torch.randn(proj_dim, self.moco_queue_size), dim=0),
            )
            self.register_buffer("ssl_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _process_timestamps(self, timestamps, item_seq_len):
        """按 time_mode 处理时间戳（训练与评估一致）。

        shuffle 用向量化实现：给有效位置随机噪声排序得到置换，
        padding 位置赋递增大值使其稳定排在末尾（相对顺序不变），
        避免逐 batch 的 Python 循环（原实现慢 3~4 倍）。
        """
        if self.time_mode == "shuffle":
            B, L = timestamps.shape
            device = timestamps.device
            noise = torch.rand(B, L, device=device)
            arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
            valid = arange < item_seq_len.unsqueeze(1)          # [B, L]
            pad_noise = 2.0 + arange.float() / max(L, 1)        # 递增，保证稳定排在末尾
            noise = torch.where(valid, noise, pad_noise)
            perm = noise.argsort(dim=1)                          # [B, L]
            return torch.gather(timestamps, 1, perm)
        elif self.time_mode == "constant":
            return torch.zeros_like(timestamps)
        return timestamps

    def _interval_bins(self, dt):
        """把时间间隔映射为桶号（两种口径，由 interval_zero_split 选择）。

        interval_zero_split=0（默认，历史口径）:
            bins = clamp(log1p(dt)/log1p(max_interval)·(B-1), 0, B-1)
            —— 与既有全部结果逐位一致，默认不改变任何现存运行。

        interval_zero_split=1（零间隔专桶）:
            dt==0 固定进 0 号桶，非零间隔线性铺满 1..B-1。
            动机来自本文件长期记录的诊断：ml-1m 中位 Δt=0 秒、>50% 的相邻对同秒，
            mind 更是 98.7% <= 1 分钟。默认口径下"同秒共现"（同一次交互批次，强信号）
            与"隔 1 秒"（时间上最弱）落在相邻低桶，被查表混为一谈；给 Δt=0 单独一个桶
            即可把两者分开，而且**不需要任何按数据集的超参**。这是用"模型内的桶标定"
            修问题，而不是去改数据——改数据会连基线一起改动，比较就不公平了。
        """
        dt = dt.clamp(min=0.0)
        b = int(self.n_interval_bins)
        u = torch.log1p(dt) / math.log1p(float(self.max_interval))
        if getattr(self, "interval_zero_split", 0):
            idx = 1 + (u * (b - 2)).long()
            idx = idx.clamp(1, b - 1)
            return torch.where(dt > 0, idx, torch.zeros_like(idx))
        return (u * (b - 1)).long().clamp(0, b - 1)

    def _get_interval_emb(self, timestamps):
        """相邻时间间隔的 log 分桶嵌入，加在输入序列上。"""
        dt = timestamps[:, 1:] - timestamps[:, :-1]  # [B, L-1]
        bins = self._interval_bins(dt)
        first = torch.zeros_like(bins[:, :1])
        bins = torch.cat([first, bins], dim=1)  # [B, L]
        return self.interval_embedding(bins)  # [B, L, H]

    def _temporal_stats(self, timestamps, item_seq_len):
        """序列级时间可靠性统计特征 [B, 4]。

        设计依据：诊断实验发现时间模块的收益高度依赖数据的时间结构，
        门控应据此判断"这条序列的时间信号是否可信"，从而自适应决定注入量。
        特征：
          1) 非零间隔比例 —— 比例低说明时间戳被离散化（如天对齐），时间信息稀薄
          2) log 间隔均值 —— 序列的整体节奏
          3) log 间隔标准差 —— 间隔的动态范围，越大说明时间结构越丰富
          4) 归一化序列长度 —— 短序列难以支撑时间建模
        """
        B, L = timestamps.shape
        dt = torch.clamp(timestamps[:, 1:] - timestamps[:, :-1], min=0.0)  # [B, L-1]
        arange = torch.arange(L - 1, device=dt.device).unsqueeze(0).expand(B, L - 1)
        valid = arange < (item_seq_len.unsqueeze(1) - 1).clamp(min=0)
        vf = valid.float()
        dt = dt * vf
        cnt = vf.sum(1).clamp(min=1.0)

        nonzero = ((dt > 0).float() * vf).sum(1) / cnt
        logdt = torch.log1p(dt)
        mean_log = (logdt * vf).sum(1) / cnt
        var_log = ((logdt - mean_log.unsqueeze(1)) ** 2 * vf).sum(1) / cnt
        std_log = torch.sqrt(var_log.clamp(min=0.0))
        seqlen = item_seq_len.float() / max(L, 1)

        return torch.stack(
            [nonzero, mean_log / 10.0, std_log / 5.0, seqlen], dim=-1
        )  # [B, 4]

    def _zpt_bias(self, timestamps):
        """零参数时间偏置: b_ij = -g * log1p(|t_i - t_j| / tau), 返回 [B, 1, L, L]。

        整个模块只有 2 个可学习标量:
          tau = softplus(log_tau) > 0  衰减尺度, 自适应数据的时间粒度
          g   = sigmoid(gate_raw) ∈ (0,1)  时间通道强度, 时间无用则自动趋于 0

        与朴素的 exp(-sqrt(dt)) 时间偏置相比的关键区别:
          - 量纲正确: dt 与 tau 同为秒, 相除后无量纲, 不受时间戳绝对大小影响
          - 不会塌成常数: dt=0 时 bias=0, dt 增大时单调下降, 行内方差始终存在,
            因此不会像常数偏置那样被 softmax 平移不变性吃掉
        """
        dt = torch.abs(timestamps.unsqueeze(2) - timestamps.unsqueeze(1))  # [B, L, L]
        tau = F.softplus(self.log_tau) + 1e-6
        bias = -torch.log1p(dt / tau)          # [B, L, L], 取值 <= 0
        g = torch.sigmoid(self.gate_raw)
        self.last_gate = g.detach()            # 供分析: 门控是否随数据集自动关闭
        self._maybe_report("zpt")
        return (g * bias).unsqueeze(1)         # [B, 1, L, L], 广播到各 head

    def _session_bias(self, timestamps):
        """会话感知注意力偏置: b_ij = -alpha * (i,j 之间跨越的会话边界数)。

        切分用软边界以保持可微:
            gap_k = sigmoid((log1p(dt_k) - tau) / s)   位置 k,k+1 之间是边界的概率
            sep_ij = |cum_i - cum_j|                    i,j 之间跨越的边界数
        用 log1p(dt) 与阈值比较, 因为间隔常跨多个数量级, 对数尺度更稳定。

        退化保护: 若数据无会话结构(如每次交互都相隔很久), gap 全趋近同一常数,
        sep 近似线性于 |i-j|, 偏置退化为与位置相关的常数项, 模型几乎不受干扰。
        """
        dt = torch.clamp(timestamps[:, 1:] - timestamps[:, :-1], min=0.0)  # [B, L-1]
        tau = F.softplus(self.sess_log_tau)
        s = F.softplus(self.sess_log_s) + 0.1
        gap = torch.sigmoid((torch.log1p(dt) - tau) / s)           # [B, L-1]
        zero = torch.zeros_like(gap[:, :1])
        cum = torch.cat([zero, torch.cumsum(gap, dim=1)], dim=1)   # [B, L]
        sep = torch.abs(cum.unsqueeze(2) - cum.unsqueeze(1))        # [B, L, L]
        alpha = F.softplus(self.sess_log_alpha)
        self.last_sep = sep.detach()           # 供分析: 切出的会话数是否合理
        self._maybe_report("session(v1)", extra={"alpha": alpha.item()})
        return (-alpha * sep).unsqueeze(1)     # [B, 1, L, L]

    def _maybe_report(self, tag, extra=None):        # noqa: C901
        """周期性把时间模块学到的参数打到日志里。

        用途: 可解释性分析。session2 的三个标量 [同会话, 跨1层, 跨2层+] 直接反映
        模型对"跨会话信息"的态度——有会话结构的数据应学到明显的区分,
        无会话结构的数据应学到接近 0(即自动退化回普通 Transformer)。
        这是"自适应行为"最直观的证据, 成本极低, 故默认开启。
        """
        self._report_count = getattr(self, "_report_count", 0) + 1
        if self._report_count % 500 != 1:
            return
        parts = ["[TimeReport] %s step=%d" % (tag, self._report_count)]
        for attr in ("sess_w", "sess3_w", "sess4_w"):
            if hasattr(self, attr):
                w = getattr(self, attr).detach().float().cpu().tolist()
                parts.append("%s=[%s]" % (attr, ", ".join("%+.4f" % v for v in w)))
        if hasattr(self, "sess4_log_tau_h"):
            # 每 head 的时间尺度 tau_h 是核心可解释性证据:
            # 可直接读作"该头关注多长的时间跨度"
            tau_h = F.softplus(self.sess4_log_tau_h).detach().float().cpu().tolist()
            g_h = torch.sigmoid(self.sess4_gate_h).detach().float().cpu().tolist()
            parts.append("tau_h=[%s]" % ", ".join("%.2f" % v for v in tau_h))
            parts.append("g_h=[%s]" % ", ".join("%.4f" % v for v in g_h))
        if hasattr(self, "sess_log_tau"):
            parts.append("tau=%.2f" % F.softplus(self.sess_log_tau).item())
        if hasattr(self, "gate_raw"):
            parts.append("gate=%.4f" % torch.sigmoid(self.gate_raw).item())
        if hasattr(self, "ck_a"):
            # ckernel 可解释性: 报告学到的尺度 r_k、频率 w_m 与各 head 系数强度。
            # r_k 可直接读作"第 k 个衰减模态的时间常数"(per log-unit)。
            r = F.softplus(self.ck_rho).detach().float().cpu().tolist()
            wm = F.softplus(self.ck_omega).detach().float().cpu().tolist()
            parts.append("r=[%s]" % ", ".join("%.3f" % v for v in r))
            parts.append("w=[%s]" % ", ".join("%.3f" % v for v in wm))
            parts.append("|a_h|=[%s]" % ", ".join(
                "%.3f" % v for v in self.ck_a.detach().norm(dim=1).float().cpu().tolist()))
            parts.append("|z_h|=[%s]" % ", ".join(
                "%.3f" % v for v in self.ck_z.detach().abs().float().cpu().tolist()))
            parts.append("|sess_h|=[%s]" % ", ".join(
                "%.3f" % v for v in self.ck_w.detach().norm(dim=1).float().cpu().tolist()))
        if extra:
            parts.append(" ".join("%s=%.4f" % (k, v) for k, v in extra.items()))
        print(" ".join(parts), flush=True)

    def _session2_bias(self, timestamps):
        """会话感知 v2: 按"跨越了几层会话边界"分配三个独立可学偏置。

        与 v1 的唯一区别: 三个标量可正可负, 不强制抑制跨会话。
        用三角核做软分配以保持可微:
            w0 ~ 同会话(sep≈0),  w1 ~ 跨1层(sep≈1),  w2 ~ 跨2层+(sep>=2)
        """
        dt = torch.clamp(timestamps[:, 1:] - timestamps[:, :-1], min=0.0)
        tau = F.softplus(self.sess_log_tau)
        s = F.softplus(self.sess_log_s) + 0.1
        gap = torch.sigmoid((torch.log1p(dt) - tau) / s)
        zero = torch.zeros_like(gap[:, :1])
        cum = torch.cat([zero, torch.cumsum(gap, dim=1)], dim=1)
        sep = torch.abs(cum.unsqueeze(2) - cum.unsqueeze(1))        # [B, L, L]

        w0 = torch.clamp(1.0 - sep, min=0.0, max=1.0)
        w1 = torch.clamp(1.0 - (sep - 1.0).abs(), min=0.0, max=1.0)
        w2 = torch.clamp(sep - 1.0, min=0.0, max=1.0)
        bias = (w0 * self.sess_w[0] + w1 * self.sess_w[1] + w2 * self.sess_w[2])
        self.last_sep = sep.detach()
        self.last_sess_w = self.sess_w.detach()   # 供分析: 学到的是增强还是抑制
        self._maybe_report("session2")
        return bias.unsqueeze(1)                   # [B, 1, L, L]

    def _session3_bias(self, timestamps):
        """session3: 每 head 间隔嵌入(TiSASRec 式) + 会话层级先验(我们独有)。

        返回 [B, n_heads, L, L]。
        注意 sess3_w 初始化为 0, 故训练起点即 TiSASRec; 若会话先验确有信息,
        模型会自行把 w 推离 0。这让"反超部分来自会话先验"的归因变得干净。
        """
        # 1) 每 head 独立的间隔嵌入
        dt = torch.abs(timestamps.unsqueeze(2) - timestamps.unsqueeze(1))  # [B, L, L]
        bins = self._interval_bins(dt)
        _ib = self.sess3_time_emb(bins)                                    # [B,L,L,H] 或 [B,L,L,1]
        if _ib.size(-1) == 1:
            _ib = _ib.expand(-1, -1, -1, self.n_heads)                     # 共享嵌入 -> 广播到各 head
        interval_bias = _ib.permute(0, 3, 1, 2)                            # [B, H, L, L]

        # 2) 会话层级先验（与 session2 相同的软切分）
        dt_adj = torch.clamp(timestamps[:, 1:] - timestamps[:, :-1], min=0.0)
        tau = F.softplus(self.sess_log_tau)
        s = F.softplus(self.sess_log_s) + 0.1
        gap = torch.sigmoid((torch.log1p(dt_adj) - tau) / s)
        zero = torch.zeros_like(gap[:, :1])
        cum = torch.cat([zero, torch.cumsum(gap, dim=1)], dim=1)
        sep = torch.abs(cum.unsqueeze(2) - cum.unsqueeze(1))               # [B, L, L]
        w0 = torch.clamp(1.0 - sep, min=0.0, max=1.0)
        w1 = torch.clamp(1.0 - (sep - 1.0).abs(), min=0.0, max=1.0)
        w2 = torch.clamp(sep - 1.0, min=0.0, max=1.0)
        sess_bias = (w0 * self.sess3_w[0] + w1 * self.sess3_w[1]
                     + w2 * self.sess3_w[2])                               # [B, L, L]

        self.last_sep = sep.detach()
        self._maybe_report("session3")
        return interval_bias + sess_bias.unsqueeze(1)                      # [B, H, L, L]

    def _session4_bias(self, timestamps):
        """session4: 连续多尺度衰减(每 head 一个时间尺度) + 会话层级先验。

        b_ij^(h) = -g_h * log1p(|t_i - t_j| / tau_h)

        相比 TiSASRec 的 32 桶查表: 连续可外推、参数少一个量级, 且 tau_h 可直接
        报告为"该注意力头关注的时间跨度", 是可解释性证据。
        gate_h 初始化为 0 -> 训练起点无时间偏置(等价普通 Transformer)。
        """
        # 1) 连续多尺度衰减
        #
        # 重要: 比较必须在"对数秒"空间进行。
        # 早期版本写成 log1p(dt / tau), 其中 dt 是真实秒(可达 1e5), 而 tau 初始化
        # 在 ~8 量级, 导致 dt/tau 上万、初始偏置压到 -6 以上, 直接把注意力打死
        # (实测 ml-1m 仅 0.1945, 且 normal≈constant≈shuffle, 时间信号完全没进去)。
        # 改为 log1p(dt) - tau, 此时 tau=8 对应约 1 小时, 量纲与语义都合理。
        dt = torch.abs(timestamps.unsqueeze(2) - timestamps.unsqueeze(1))  # [B, L, L]
        dt_log = torch.log1p(dt)                                           # [B, L, L]
        tau_h = F.softplus(self.sess4_log_tau_h).view(1, -1, 1, 1)         # [1, H, 1, 1]
        # 强度用 sigmoid 限幅到 (0,1), 避免 softplus 无界导致偏置过大
        g_h = torch.sigmoid(self.sess4_gate_h).view(1, -1, 1, 1)           # [1, H, 1, 1]
        decay = -g_h * F.softplus(dt_log.unsqueeze(1) - tau_h)             # [B, H, L, L]

        # 2) 会话层级先验（与 session2/3 相同的软切分）
        dt_adj = torch.clamp(timestamps[:, 1:] - timestamps[:, :-1], min=0.0)
        tau = F.softplus(self.sess_log_tau)
        s = F.softplus(self.sess_log_s) + 0.1
        gap = torch.sigmoid((torch.log1p(dt_adj) - tau) / s)
        zero = torch.zeros_like(gap[:, :1])
        cum = torch.cat([zero, torch.cumsum(gap, dim=1)], dim=1)
        sep = torch.abs(cum.unsqueeze(2) - cum.unsqueeze(1))               # [B, L, L]
        w0 = torch.clamp(1.0 - sep, min=0.0, max=1.0)
        w1 = torch.clamp(1.0 - (sep - 1.0).abs(), min=0.0, max=1.0)
        w2 = torch.clamp(sep - 1.0, min=0.0, max=1.0)
        sess_bias = (w0 * self.sess4_w[0] + w1 * self.sess4_w[1]
                     + w2 * self.sess4_w[2])                               # [B, L, L]

        self.last_sep = sep.detach()
        self._maybe_report("session4")
        return decay + sess_bias.unsqueeze(1)                              # [B, H, L, L]

    # ---------- ckernel: 连续可学时间核 ----------

    def _session_levels(self, timestamps):
        """软切分出的"跨越了几层会话边界"的三档权重 [B, L, L] -> (w0, w1, w2)。

        三档语义: w0≈同会话, w1≈跨 1 层, w2≈跨 2 层以上。
        与 session2/3/4 使用完全相同的切分逻辑, 保证与历史变体可比。
        """
        dt_adj = torch.clamp(timestamps[:, 1:] - timestamps[:, :-1], min=0.0)
        tau = F.softplus(self.sess_log_tau)
        s = F.softplus(self.sess_log_s) + 0.1
        gap = torch.sigmoid((torch.log1p(dt_adj) - tau) / s)               # [B, L-1]
        zero_pad = torch.zeros_like(gap[:, :1])
        cum = torch.cat([zero_pad, torch.cumsum(gap, dim=1)], dim=1)        # [B, L]
        sep = torch.abs(cum.unsqueeze(2) - cum.unsqueeze(1))               # [B, L, L]
        w0 = torch.clamp(1.0 - sep, min=0.0, max=1.0)
        w1 = torch.clamp(1.0 - (sep - 1.0).abs(), min=0.0, max=1.0)
        w2 = torch.clamp(sep - 1.0, min=0.0, max=1.0)
        self.last_sep = sep.detach()
        return w0, w1, w2

    def ckernel_basis(self, u):
        """在给定的归一化对数间隔 u 上求基函数值, 供前向与可解释性导出共用。

        u 可为任意形状张量; 返回 (decay, cos_t, sin_t), 形状均为 [..., K/M, ...] 前置一维。
        """
        r = F.softplus(self.ck_rho).view(1, -1, 1, 1)
        decay = torch.exp(-r * u.unsqueeze(1))
        w = F.softplus(self.ck_omega).view(1, -1, 1, 1)
        ang = w * u.unsqueeze(1)
        return decay, torch.cos(ang), torch.sin(ang)

    @torch.no_grad()
    def ckernel_curve(self, n_grid=48, dt_max_s=None):
        """导出每个 head 的时间核曲线, 用于可解释性分析（不进反传）。

        返回 dict: dt(秒) / u / bias[H, n_grid]。
        可直接回答"第 h 个头关注多长的时间跨度"——这是本模块的可解释性证据。
        """
        dt_max_s = dt_max_s or self.ck_norm
        dt = torch.linspace(0.0, float(dt_max_s), n_grid)
        u = torch.log1p(dt) / self.ck_log_norm                            # [G]
        r = F.softplus(self.ck_rho).view(-1, 1)
        decay = torch.exp(-r * u.view(1, -1))                             # [K, G]
        w = F.softplus(self.ck_omega).view(-1, 1)
        ang = w * u.view(1, -1)
        bias = (self.ck_a.detach() @ decay
                + self.ck_c.detach() @ torch.cos(ang)
                + self.ck_d.detach() @ torch.sin(ang))                    # [H, G]
        bias = bias + self.ck_z.detach().view(-1, 1) * (dt.view(1, -1) < 0.5).float()
        return {"dt": dt.numpy(), "u": u.numpy(), "bias": bias.numpy()}

    def _ckernel_bias(self, timestamps):
        """连续可学时间核 -> [B, n_heads, L, L]。系数初始为 0 时恒等于 0。"""
        dt = torch.abs(timestamps.unsqueeze(2) - timestamps.unsqueeze(1))  # [B,L,L]
        u = torch.log1p(dt) / self.ck_log_norm                            # 无上界 => 可外推

        # 1) 多尺度衰减基
        r = F.softplus(self.ck_rho).view(1, -1, 1, 1)
        decay = torch.exp(-r * u.unsqueeze(1))                            # [B,K,L,L]
        bias = torch.einsum("hk,bkij->bhij", self.ck_a, decay)

        # 2) 对数尺度上的周期基（让"周期性"第一次可被表达）
        w = F.softplus(self.ck_omega).view(1, -1, 1, 1)
        ang = w * u.unsqueeze(1)                                          # [B,M,L,L]
        bias = bias + torch.einsum("hm,bmij->bhij", self.ck_c, torch.cos(ang)) \
                    + torch.einsum("hm,bmij->bhij", self.ck_d, torch.sin(ang))

        # 3) 零间隔(同秒/同批次)专属项：这批数据里它是"共现"而非"时间最小"
        zero = (dt < 0.5).to(bias.dtype).unsqueeze(1)                      # [B,1,L,L]
        bias = bias + self.ck_z.view(1, -1, 1, 1) * zero

        # 4) 会话层级先验（per-head）
        w0, w1, w2 = self._session_levels(timestamps)
        cw = self.ck_w.view(1, self.n_heads, 3, 1, 1)
        sess = (w0.unsqueeze(1) * cw[:, :, 0]
                + w1.unsqueeze(1) * cw[:, :, 1]
                + w2.unsqueeze(1) * cw[:, :, 2])                          # [B,H,L,L]
        bias = bias + sess

        self._maybe_report("ckernel")
        return bias

    def forward(self, item_seq, item_seq_len, timestamps=None):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding

        attn_bias = None
        if self.temporal_type != "none" and timestamps is not None:
            timestamps = self._process_timestamps(timestamps, item_seq_len)
            if self.temporal_type == "interval":
                input_emb = input_emb + self._get_interval_emb(timestamps)
            elif self.temporal_type == "periodic":
                input_emb = input_emb + self.periodic_encoder(timestamps)
            elif self.temporal_type == "adaptive":
                # 时间编码按"可靠性门控"加权注入：门控由序列自身的时间统计特征决定
                time_emb = self._get_interval_emb(timestamps)
                stats = self._temporal_stats(timestamps, item_seq_len)
                gate = self.gate(stats)                       # [B, 1]，值域 (0,1)
                input_emb = input_emb + gate.unsqueeze(1) * time_emb
                self.last_gate = gate.detach()                # 供分析：门控均值是否随数据集变化
            elif self.temporal_type == "zpt":
                attn_bias = self._zpt_bias(timestamps)        # [B, 1, L, L]
            elif self.temporal_type == "session":
                attn_bias = self._session_bias(timestamps)    # [B, 1, L, L]
            elif self.temporal_type == "session2":
                attn_bias = self._session2_bias(timestamps)   # [B, 1, L, L]
            elif self.temporal_type == "session3":
                attn_bias = self._session3_bias(timestamps)   # [B, n_heads, L, L]
            elif self.temporal_type == "session4":
                attn_bias = self._session4_bias(timestamps)   # [B, n_heads, L, L]
            elif self.temporal_type == "ckernel":
                attn_bias = self._ckernel_bias(timestamps)    # [B, n_heads, L, L]

        # 统一 dtype：当 timestamp_fp64=1 时时间戳是 float64，上面各条时间通道会产生
        # float64 中间量；这里一次性转回骨架 dtype，避免隐式提升带来的 dtype 不一致。
        _dt = self.item_embedding.weight.dtype
        input_emb = input_emb.to(_dt)
        if attn_bias is not None:
            attn_bias = attn_bias.to(_dt)

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, attn_bias, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    # ==================== SSL（自监督）实现 ====================

    def _shuffle_ts(self, timestamps, item_seq_len):
        """用户内置换时间戳（向量化），作为"时间破坏"增强视图。"""
        B, L = timestamps.shape
        device = timestamps.device
        noise = torch.rand(B, L, device=device)
        arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        valid = arange < item_seq_len.unsqueeze(1)
        pad_noise = 2.0 + arange.float() / max(L, 1)
        noise = torch.where(valid, noise, pad_noise)
        perm = noise.argsort(dim=1)
        return torch.gather(timestamps, 1, perm)

    def _mask_items(self, item_seq, item_seq_len):
        """保时增强（time-preserving）：随机替换部分 item，时间戳完全不变。

        与 shuffle 的本质区别：
          shuffle    -> 置换时间戳，破坏时间结构，等价于迫使模型忽略时间信号，
                        与时间建模目标直接冲突（实测掉点）。
          item_mask  -> 时间戳原样保留，仅扰动 item 内容，
                        让模型学到对 item 噪声的鲁棒性，而不损害时间建模。
        """
        B, L = item_seq.shape
        device = item_seq.device
        arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        valid = arange < item_seq_len.unsqueeze(1)
        prob = torch.rand(B, L, device=device) < self.ssl_mask_ratio
        mask = prob & valid
        rand_items = torch.randint(1, self.n_items, (B, L), device=device)
        return torch.where(mask, rand_items, item_seq)

    def _ssl_views(self, item_seq, item_seq_len, timestamps):
        """生成两个增强视图的表示 (h1, h2)，增强方式由 ssl_aug 决定。

        - shuffle   : 视图2 置换时间戳（时间破坏，与时间建模冲突，已证实有害）
        - dropout   : 同一序列两次前向，靠 dropout 随机性产生差异（DuoRec 式）
        - item_mask : 保时增强——时间戳不变，仅随机替换部分 item（本方法提出）
        """
        if self.ssl_aug == "shuffle":
            h1 = self.forward(item_seq, item_seq_len, timestamps)
            ts2 = self._shuffle_ts(timestamps, item_seq_len)
            h2 = self.forward(item_seq, item_seq_len, ts2)
        elif self.ssl_aug == "dropout":
            h1 = self.forward(item_seq, item_seq_len, timestamps)
            h2 = self.forward(item_seq, item_seq_len, timestamps)
        elif self.ssl_aug == "item_mask":
            h1 = self.forward(item_seq, item_seq_len, timestamps)
            seq2 = self._mask_items(item_seq, item_seq_len)
            h2 = self.forward(seq2, item_seq_len, timestamps)
        else:
            raise NotImplementedError("ssl_aug=%s not implemented" % self.ssl_aug)
        if self.ssl_proj is not None:
            h1 = self.ssl_proj(h1)
            h2 = self.ssl_proj(h2)
        return h1, h2

    def _ssl_loss(self, item_seq, item_seq_len, timestamps):
        """按 ssl_method 分发到具体 SSL 损失。"""
        if self.ssl_method == "none":
            return torch.zeros((), device=item_seq.device)
        h1, h2 = self._ssl_views(item_seq, item_seq_len, timestamps)
        if self.ssl_method == "infonce":
            return self._infonce(h1, h2)
        if self.ssl_method == "moco":
            return self._moco_loss(h1, h2)
        if self.ssl_method == "byol":
            return self._byol_loss(h1, h2)
        if self.ssl_method == "barlow":
            return self._barlow_loss(h1, h2)
        raise NotImplementedError("ssl_method=%s not implemented" % self.ssl_method)

    def _infonce(self, h1, h2):
        """对称 InfoNCE（SimCLR 式），负样本为 batch 内其他序列。"""
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)
        logits = torch.matmul(h1, h2.T) / self.ssl_temperature
        labels = torch.arange(h1.size(0), device=h1.device)
        return (self.loss_fct(logits, labels) + self.loss_fct(logits.T, labels)) * 0.5

    def _moco_loss(self, h1, h2):
        """MoCo 式队列对比（先无动量，后续加 EMA 动量编码器）。"""
        q = F.normalize(h1, dim=-1)
        k = F.normalize(h2, dim=-1).detach()   # 停止 key 梯度
        B = k.size(0)
        # FIFO 入队
        ptr = int(self.ssl_queue_ptr)
        room = self.moco_queue_size - ptr
        if B <= room:
            self.ssl_queue[:, ptr:ptr + B] = k.T
            self.ssl_queue_ptr[0] = (ptr + B) % self.moco_queue_size
        else:
            self.ssl_queue[:, ptr:] = k[:room].T
            self.ssl_queue[:, :B - room] = k[room:].T
            self.ssl_queue_ptr[0] = B - room
        # query 与 [当前 key + 队列负样本] 对比
        logits_pos = (q * k).sum(-1, keepdim=True) / self.ssl_temperature
        logits_neg = torch.matmul(q, self.ssl_queue) / self.ssl_temperature
        logits = torch.cat([logits_pos, logits_neg], dim=-1)
        labels = torch.zeros(B, dtype=torch.long, device=q.device)
        return self.loss_fct(logits, labels)

    def _byol_loss(self, h1, h2):
        raise NotImplementedError("byol 待实现（无负样本：动量编码器 + stop-gradient）")

    def _barlow_loss(self, h1, h2):
        raise NotImplementedError("barlow 待实现（时间/序列通道去冗余）")

    def _get_timestamps(self, interaction):
        if "timestamp_list" in interaction:
            return interaction["timestamp_list"]
        return None

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        timestamps = self._get_timestamps(interaction)
        seq_output = self.forward(item_seq, item_seq_len, timestamps)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # SSL 辅助损失（ssl_weight=0 时不生效，零额外开销）
        if self.ssl_weight > 0 and self.ssl_method != "none":
            loss = loss + self.ssl_weight * self._ssl_loss(
                item_seq, item_seq_len, timestamps)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        timestamps = self._get_timestamps(interaction)
        seq_output = self.forward(item_seq, item_seq_len, timestamps)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        timestamps = self._get_timestamps(interaction)
        seq_output = self.forward(item_seq, item_seq_len, timestamps)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

# ---------------------------------------------------------------------------
# 兼容别名（deprecated）
# ---------------------------------------------------------------------------
# 变体已全部收敛到 LiteTK4Rec，因此这里只做名字映射，**不是另一个模型**。
# 新任务请使用 `--model=LiteTK4Rec`。

# ---------------------------------------------------------------------------
