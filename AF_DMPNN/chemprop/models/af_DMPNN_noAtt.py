import torch
import torch.nn.functional as F
from chemprop.args import TrainArgs
from chemprop.features import get_atom_fdim, get_bond_fdim, BatchMolGraph, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
from torch import nn
from typing import List
import numpy as np


class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力桥接模块
    连接原子级和碎片级表征
    """

    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        # 查询、键、值的投影
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # 跨尺度门控机制
        self.scale_gate = nn.Linear(hidden_dim * 2, 1)
        # 非线性变换
        self.nonlinear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, molecular_context, atom_repr, fragment_repr):
        """
        参数:
            molecular_context: [batch_size, hidden_dim]
            atom_repr: [batch_size, num_atoms, hidden_dim]
            fragment_repr: [batch_size, num_fragments, hidden_dim]
        """
        batch_size, num_atoms, _ = atom_repr.shape
        _, num_fragments, _ = fragment_repr.shape

        contextual_features = torch.cat([
            molecular_context.unsqueeze(1).expand(-1, num_fragments, -1),
            fragment_repr
        ], dim=-1)

        # 非线性变换得到碎片权重
        frag_importance = torch.sigmoid(
            self.nonlinear(contextual_features).mean(dim=-1)
        )  # [batch_size, num_fragments]
        frag_importance = F.softmax(frag_importance, dim=-1)
        # 投影
        query = self.query_proj(atom_repr)  # [batch_size, num_atoms, hidden_dim]
        key = self.key_proj(fragment_repr)  # [batch_size, num_fragments, hidden_dim]
        value = self.value_proj(fragment_repr)  # [batch_size, num_fragments, hidden_dim]

        # 计算注意力分数
        attention_scores = torch.bmm(query,
                                     key.transpose(1, 2)) / self.temperature  # [batch_size, num_atoms, num_fragments]

        # 应用碎片权重
        expanded_importance = frag_importance.unsqueeze(1).expand(-1, num_atoms,
                                                                  -1)  # [batch_size, num_atoms, num_fragments]
        weighted_scores = attention_scores + torch.log(expanded_importance + 1e-8)

        # softmax归一化
        attention_weights = F.softmax(weighted_scores, dim=-1)

        # 应用注意力权重到值上
        attended_value = torch.bmm(attention_weights, value)  # [batch_size, num_atoms, hidden_dim]

        # 门控融合
        concatenated = torch.cat([atom_repr, attended_value], dim=-1)
        gate_values = torch.sigmoid(self.scale_gate(concatenated))
        # 残差连接
        fused_repr = gate_values * atom_repr + (1 - gate_values) * attended_value

        return fused_repr


class DMPNN(nn.Module):

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(DMPNN, self).__init__()
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim(atom_messages=args.atom_messages)
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.device = args.device
        self.fragment_type = args.fragment_type
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # Activation
        self.act_func = get_activation_function(args.activation)
        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        w_h_input_size = self.hidden_size
        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.gate_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self, batch: BatchMolGraph):
        """
        Encodes a batch of molecular graphs.

        :param batch: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        """
        fragList, frag2a, mol2Frags, frag2frag, f_frags, a2frag, a2neighbor, frag2mol = batch.get_frags(self.fragment_type)
        fragBatch = mol2graph(fragList)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = batch.get_components()
        # Input
        input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size
        f_frags_atoms, f_frags_bonds, frags_a2b, frags_b2a, frags_b2revb, frags_a_scope, frags_b_scope = fragBatch.get_components()
        with torch.no_grad():
            frags_input = self.W_i(f_frags_bonds)  # num_bonds x hidden_size
            frags_b_message = self.act_func(frags_input)  # num_bonds x hidden_size
        # Message passing
        for depth in range(self.depth - 1):
            # 获取碎片信息
            with torch.no_grad():
                frags_b_message = self.message_aggregation(frags_input, frags_b_message, frags_a2b, frags_b2a,
                                                           frags_b2revb)
                frags_feature, _ = self.molReadout(f_frags_atoms, frags_b_message, frags_a2b, frags_a_scope,
                                                   padding=True)
            # 碎片->原子信息交互残差连接
            frags_a_message = frags_feature[a2frag]
            frags_a_message_expend = frags_a_message[b2a]
            combined_message = torch.cat([message, frags_a_message_expend], dim=1)
            gate_values = self.gate_layer(combined_message)
            message = gate_values * message + (1 - gate_values) * frags_a_message_expend
            message = self.message_aggregation(input, message, a2b, b2a, b2revb)
        frags_feature, _ = self.molReadout(f_frags_atoms, frags_b_message, frags_a2b, frags_a_scope, padding=True)
        mol_vecs, atom_hiddens = self.molReadout(f_atoms, message, a2b, a_scope)
        return mol_vecs, atom_hiddens, frags_feature, mol2Frags  # num_molecules x hidden

    def message_aggregation(self, input, message, x2y, y2x, y2revb):
        nei_a_message = index_select_ND(message, x2y)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        rev_message = message[y2revb]  # num_bonds x hidden
        message = a_message[y2x] - rev_message  # num_bonds x hidden
        message = self.W_h(message)
        message = self.act_func(input + message)  # num_bonds x hidden_size
        message = self.dropout_layer(message)  # num_bonds x hidden
        return message

    def molReadout(self, f_x, message, x2y, x_scope, padding=False):
        nei_a_message = index_select_ND(message, x2y)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_x, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        if padding:
            mol_vecs.append(self.cached_zero_vector)
        for i, (a_start, a_size) in enumerate(x_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)
        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs, atom_hiddens


class af_DMPNN(nn.Module):
    """
    完整的增强版DMPNN：层级融合 + 自适应重要性评估
    """

    def __init__(self, args: TrainArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout
        # 原子级DMPNN编码器 (原有主干)
        self.atom_dmpnn = DMPNN(args)
        # 跨尺度注意力桥接
        self.cross_scale_attention = CrossScaleAttention(self.hidden_size)
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.fragment_type = args.fragment_type

    def forward(self, batch: BatchMolGraph, features_batch: List[np.ndarray] = None):
        """
        参数:
            batch: 分子图批次数据
        """
        # # 通过DMPNN获取原子级表征
        molecular_global, atom_repr, f_frag, mol2Frags = self.atom_dmpnn(batch)
        fragment_repr = index_select_ND(f_frag, mol2Frags)
        fragment_repr = fragment_repr[1:, :]
        # === 跨尺度融合 ===
        mol2a = batch.get_mol2a()
        mols_atoms = index_select_ND(atom_repr, mol2a)
        mols_atoms = mols_atoms[1:, :]
        molPerfrags = (mol2Frags > 0).float().sum(dim=1).unsqueeze(-1)[1:, :]
        # === 分层读取 ===
        mean_pool = mols_atoms.sum(dim=1)
        max_pool = mols_atoms.max(dim=1)[0]
        mean_fragment = torch.sum(fragment_repr, dim=1) / molPerfrags
        molFeature = torch.cat([molecular_global, mean_pool, max_pool, mean_fragment], dim=-1)
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)
            features_batch = features_batch.to(molFeature)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
            molFeature = torch.cat([molFeature, features_batch], dim=1)  # (num_molecules, hidden_size)
        return molFeature
