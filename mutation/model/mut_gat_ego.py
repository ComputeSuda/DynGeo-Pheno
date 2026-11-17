import tensorflow as tf
from keras import backend as K
from keras import models, layers
from keras.layers import Input, Dropout, Dense, Concatenate, LayerNormalization, Add, GlobalAveragePooling1D, Multiply

class InverseLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(InverseLayer, self).__init__(**kwargs)
    def call(self, inputs):
        return 1.0 - inputs
    def get_config(self):
        return super(InverseLayer, self).get_config()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module: mut_gat_ego.py
Description:
Local 3D microenvironment Graph Attention Network (MutEgoGATBatch) for missense mutation prediction:
1. Sequence projection layer (seq_proj): aligns 128-d ESM embeddings to arbitrary hidden_dim.
2. Adaptive frequency-band gating (FrequencyBandGating: low vs. high frequency modes).
3. Sequence-dynamics residual gating fusion.
4. Edge-biased GATv2 convolutions incorporating geometric distance, cross-correlation, and stiffness.
5. Dual-stream readout combining central mutated residue (h_center) and microenvironment pooling (h_env).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax


class FrequencyBandGating(nn.Module):
    """Adaptive gating unit for low- and high-frequency dynamic features."""
    def __init__(self, in_dim_low, in_dim_high, out_dim):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim_low + in_dim_high, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.proj_low = nn.Linear(in_dim_low, out_dim)
        self.proj_high = nn.Linear(in_dim_high, out_dim)
        self.out_ln = nn.LayerNorm(out_dim)

    def forward(self, x_low, x_high):
        cat_feat = torch.cat([x_low, x_high], dim=-1)
        gate = self.gate_mlp(cat_feat)
        h_low = self.proj_low(x_low)
        h_high = self.proj_high(x_high)
        fused = gate * h_low + (1.0 - gate) * h_high
        return self.out_ln(fused), gate


class MutEgoGATBatch(nn.Module):
    def __init__(
        self,
        dyn_dim=0,
        in_dim_low=0,
        in_dim_high=0,
        edge_dim=2,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_dual_band = (in_dim_low > 0 and in_dim_high > 0)
        self.use_dyn = (dyn_dim > 0 or self.is_dual_band)

        # 0. Sequence projection layer aligning 128-d ESM features to hidden_dim 
        if hidden_dim == 128:
            self.seq_proj = nn.Identity()
        else:
            self.seq_proj = nn.Sequential(
                nn.Linear(128, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )

        # 1. Dynamic feature adaptor and gating modules 
        if self.is_dual_band:
            self.band_gate = FrequencyBandGating(in_dim_low, in_dim_high, hidden_dim)
            self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        elif dyn_dim > 0:
            self.dyn_adaptor = nn.Sequential(
                nn.Linear(dyn_dim, 64),
                nn.GELU(),
                nn.Linear(64, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)

        self.node_ln = nn.LayerNorm(hidden_dim)

        # 2. Continuous edge-guided GATv2 backbone 
        self.conv_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=4,
                concat=False,
                edge_dim=edge_dim,
                dropout=dropout
            )
            self.conv_layers.append(conv)
            self.ln_layers.append(nn.LayerNorm(hidden_dim))

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 3. Microenvironment attention pooling 
        self.env_attn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # 4. Dual-stream classification head: [h_center || h_env] 
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, batch_data, return_gate=False):
        x_seq = self.seq_proj(batch_data.x_seq)
        edge_index = batch_data.edge_index
        edge_attr = batch_data.edge_attr
        batch = batch_data.batch
        center_mask = batch_data.center_mask.bool()

        gate_band = None

        # Step 1: Node-level dynamic gating fusion 
        if self.is_dual_band and hasattr(batch_data, 'x_dyn_low') and hasattr(batch_data, 'x_dyn_high'):
            h_dyn, gate_band = self.band_gate(batch_data.x_dyn_low, batch_data.x_dyn_high)
            gate_seq = torch.sigmoid(self.gate_fc(torch.cat([x_seq, h_dyn], dim=-1)))
            h = self.node_ln(x_seq + gate_seq * h_dyn)
        elif self.use_dyn and hasattr(batch_data, 'x_dyn') and batch_data.x_dyn is not None and batch_data.x_dyn.size(-1) > 0 and not torch.all(batch_data.x_dyn == 0):
            h_dyn = self.dyn_adaptor(batch_data.x_dyn)
            gate_seq = torch.sigmoid(self.gate_fc(torch.cat([x_seq, h_dyn], dim=-1)))
            h = self.node_ln(x_seq + gate_seq * h_dyn)
        else:
            h = self.node_ln(x_seq)

        # Step 2: Spatial graph attention message passing 
        for conv, ln in zip(self.conv_layers, self.ln_layers):
            h_res = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = ln(h)
            h = self.act(h)
            h = self.dropout(h)
            h = h + h_res

        # Step 3: Extract central mutated residue representation 
        h_center = h[center_mask]

        # Step 4: Attention-weighted microenvironment pooling 
        attn_logits = self.env_attn(h)
        attn_w = pyg_softmax(attn_logits, batch)
        h_env = global_add_pool(attn_w * h, batch)

        # Step 5: Dual-stream classification 
        h_pair = torch.cat([h_center, h_env], dim=-1)
        logits = self.classifier(h_pair).squeeze(-1)

        if return_gate:
            return logits, gate_band
        return logits
class GraphConvolutionLayer(layers.Layer):
    def __init__(self, output_dim, lambda_param=1.3, alpha=0.7, l=1,
                 activation=None, use_bias=True, variant=True, **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.l = l
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.variant = variant

    def build(self, input_shape):
        features_shape, adjacency_matrix_shape, h0_shape = input_shape
        input_dim = features_shape[-1]
        self.input_dim = 2 * input_dim if self.variant else input_dim
        self.weight = self.add_weight(shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform', trainable=True)
        self.residual_proj = layers.Dense(self.output_dim, use_bias=False)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,), initializer='zeros', trainable=True)
        super(GraphConvolutionLayer, self).build(input_shape)

    def call(self, inputs):
        features, adjacency_matrix, h0 = inputs
        l = tf.cast(self.l, float)
        theta = tf.minimum(1.0, tf.math.log(self.lambda_param / l + 1.0))
        hi = tf.matmul(adjacency_matrix, features)
        if self.variant:
            support = tf.concat([hi, h0], axis=-1)
            r = (1 - self.alpha) * hi + self.alpha * h0
        else:
            support = (1 - self.alpha) * hi + self.alpha * h0
            r = support
        conv_out = tf.matmul(support, self.weight)
        r_proj = self.residual_proj(r)
        output = theta * conv_out + (1 - theta) * r_proj
        if self.use_bias:
            output += self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim, 'lambda_param': self.lambda_param,
            'alpha': self.alpha, 'l': self.l,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias, 'variant': self.variant
        }
        return dict(list(super().get_config().items()) + list(config.items()))


class GraphAttentionLayer(layers.Layer):
    def __init__(self, in_features, out_features, dropout, activation='relu', concat=True, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.in_features, self.out_features),
                                 initializer='glorot_uniform', trainable=True, name='W')
        self.a = self.add_weight(shape=(2 * self.out_features, 1),
                                 initializer='glorot_uniform', trainable=True, name='a')
        self.leakyrelu = layers.LeakyReLU()
        super().build(input_shape)

    def call(self, inputs, training=False):
        h, adj = inputs
        Wh = tf.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * tf.ones_like(e)
        attention = tf.where(adj > 0, e, zero_vec)
        attention = tf.nn.softmax(attention, axis=-1)
        attention = tf.nn.dropout(attention, rate=self.dropout)
        h_prime = tf.matmul(attention, Wh)
        return tf.nn.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = tf.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = tf.matmul(Wh, self.a[self.out_features:, :])
        return self.leakyrelu(Wh1 + tf.transpose(Wh2, perm=[0, 2, 1]))

    def get_config(self):
        config = super().get_config()
        config.update({'in_features': self.in_features, 'out_features': self.out_features,
                       'dropout': self.dropout, 'concat': self.concat})
        return config


class GAT(layers.Layer):
    def __init__(self, in_feature, out_feature, dropout, nheads, **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.dropout = dropout
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.nheads = nheads

    def build(self, input_shape):
        self.attentions = [GraphAttentionLayer(self.in_feature, self.out_feature, self.dropout, concat=True)
                          for _ in range(self.nheads)]
        self.out_att = GraphAttentionLayer(self.out_feature * self.nheads, self.out_feature, self.dropout, concat=False)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x, adj = inputs
        x = tf.nn.dropout(x, rate=self.dropout)
        x = tf.concat([att([x, adj], training=training) for att in self.attentions], axis=-1)
        x = tf.nn.dropout(x, rate=self.dropout)
        x = self.out_att([x, adj], training=training)
        return tf.nn.elu(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_feature)

    def get_config(self):
        config = super().get_config()
        config.update({'dropout': self.dropout, 'in_feature': self.in_feature,
                       'out_feature': self.out_feature, 'nheads': self.nheads})
        return config


class GCN_GATLayer(layers.Layer):
    def __init__(self, in_feature, out_feature, dropout, nheads, lamda, alpha, l=5, **kwargs):
        kwargs.pop('l', None)
        super(GCN_GATLayer, self).__init__(**kwargs)
        self.nheads = nheads
        self.dropout = dropout
        self.lamda = lamda
        self.alpha = alpha
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.l = l

    def build(self, input_shape):
        self.gat_layers = GAT(self.in_feature, self.out_feature, self.dropout, self.nheads)
        self.gcn_layer = GraphConvolutionLayer(self.out_feature, activation='relu', use_bias=True, variant=False)
        self.cat_concat = layers.Concatenate(axis=-1)
        self.cat_lin = layers.Dense(self.out_feature)
        self.res_add = layers.Add()
        self.residual_proj = layers.Dense(self.out_feature) if self.in_feature != self.out_feature else None
        super().build(input_shape)

    def call(self, inputs):
        layer_inner, input_cmap, _ = inputs
        gat_out = self.gat_layers([layer_inner, input_cmap])
        gcn_out = self.gcn_layer([layer_inner, input_cmap, layer_inner])
        residual = self.residual_proj(layer_inner) if self.residual_proj is not None else layer_inner
        return self.res_add([self.cat_lin(self.cat_concat([gat_out, gcn_out])), residual])

    def get_config(self):
        config = super().get_config().copy()
        config.update({'nheads': self.nheads, 'dropout': self.dropout, 'lamda': self.lamda,
                       'alpha': self.alpha, 'in_feature': self.in_feature,
                       'out_feature': self.out_feature, 'l': self.l})
        return config


def build_seq_pssm_feature2_model(dim1):
    input_x = Input(shape=(dim1, 128), name='knn_G')
    x_emb = Dropout(0.1)(input_x)

    input_KF_front = Input(shape=(dim1, 90), name='KF_front')
    input_KF_last = Input(shape=(dim1, 90), name='KF_last')
    x_KF_front = Dropout(0.1)(input_KF_front)
    x_KF_last = Dropout(0.1)(input_KF_last)

    feat_front = Dense(128, activation='relu', name='feat_front_proj')(x_KF_front)
    feat_last = Dense(128, activation='relu', name='feat_last_proj')(x_KF_last)

    concat_feats = Concatenate(name='gate_concat')([feat_front, feat_last])
    gate = Dense(128, activation='sigmoid', name='fusion_gate')(concat_feats)

    gated_front = Multiply(name='gate_front_mult')([gate, feat_front])
    gated_last = Multiply(name='gate_last_mult')([InverseLayer(name='gate_inverse')(gate), feat_last])
    x_emb_KF = Add(name='gate_add')([gated_front, gated_last])

    node_feature = Concatenate(name='knn_feature')([x_emb, x_emb_KF])
    input_cmap = Input(shape=(None, None), name='input_knn_cmap')

    num_layers = 3
    input_dim = 256
    hidden_dim = 218
    heads, dropout, lamda, alpha, l = 4, 0.3, 1.3, 0.1, 5

    gcn_gat_layers = []
    for i in range(num_layers):
        in_feat = input_dim if i == 0 else hidden_dim
        gcn_gat_layers.append(GCN_GATLayer(in_feat, hidden_dim, dropout, heads, lamda, alpha, l))

    layer_inner = node_feature
    for gcn_gat_layer in gcn_gat_layers:
        layer_inner = Dropout(dropout)(layer_inner)
        gcn_gat_out = gcn_gat_layer([layer_inner, input_cmap, layer_inner])
        gcn_gat_out = LayerNormalization()(gcn_gat_out)
        layer_inner = gcn_gat_out

    extracted_feature = Dropout(0.5)(layer_inner)
    extracted_feature = Dense(32, activation="relu")(extracted_feature)
    knn49_pooling_extracted_feature = GlobalAveragePooling1D()(extracted_feature)
    out = Dropout(0.5)(knn49_pooling_extracted_feature)
    output = Dense(1, activation='sigmoid', name='output_dense')(out)

    model = models.Model(inputs=[input_x, input_KF_front, input_KF_last, input_cmap], outputs=output)
    return model
