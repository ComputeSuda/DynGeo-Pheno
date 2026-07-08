import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import (Layer, Dense, Dropout, Concatenate,
                                     LayerNormalization, GlobalAveragePooling1D,
                                     Multiply, Add, Input)

class InverseLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return 1.0 - inputs
    def get_config(self):
        return super().get_config()

class GraphConvolutionLayer(Layer):
    def __init__(self, output_dim, activation=None, use_bias=True, variant=True, name_prefix='', **kwargs):
        if 'name' not in kwargs: kwargs['name'] = name_prefix
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.variant = variant
        self.name_prefix = name_prefix

    def build(self, input_shape):
        feat_dim = input_shape[0][-1]
        h0_dim = input_shape[2][-1]
        self.feat_dim = feat_dim
        self.h0_dim = h0_dim
        if self.variant:
            self.input_dim = feat_dim + h0_dim
            self.weight = self.add_weight(shape=(self.input_dim, self.output_dim), initializer='glorot_uniform', trainable=True, name=f'{self.name_prefix}_weight')
            self.r_weight = self.add_weight(shape=(feat_dim, self.output_dim), initializer='glorot_uniform', trainable=True, name=f'{self.name_prefix}_r_weight')
        else:
            self.input_dim = feat_dim
            self.weight = self.add_weight(shape=(self.input_dim, self.output_dim), initializer='glorot_uniform', trainable=True, name=f'{self.name_prefix}_weight')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,), initializer='zeros', trainable=True, name=f'{self.name_prefix}_bias')
        super().build(input_shape)

    def call(self, inputs):
        features, adj, h0, lamda, alpha, l = inputs
        l = tf.cast(l, tf.float32)
        theta = tf.minimum(1.0, tf.math.log(lamda / l + 1.0))
        hi = tf.matmul(adj, features)
        if self.variant:
            support = tf.concat([hi, h0], axis=-1)
            r = (1 - alpha) * hi + alpha * h0
            r_proj = tf.matmul(r, self.r_weight)
            output = theta * tf.matmul(support, self.weight) + (1-theta) * r_proj
        else:
            support = (1 - alpha) * hi + alpha * h0
            output = theta * tf.matmul(support, self.weight) + (1-theta) * support
        if self.use_bias: output += self.bias
        if self.activation: output = self.activation(output)
        return output

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(output_dim=self.output_dim, activation=tf.keras.activations.serialize(self.activation), use_bias=self.use_bias, variant=self.variant, name_prefix=self.name_prefix))
        return cfg

class GraphAttentionLayer(Layer):
    def __init__(self, in_features, out_features, dropout, activation='relu', name_prefix='', concat=True, **kwargs):
        if 'name' not in kwargs: kwargs['name'] = name_prefix
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout
        self.activation_name = activation
        self.concat = concat
        self.name_prefix = name_prefix
        self.leakyrelu = tf.keras.layers.LeakyReLU()
        self.attn_dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.in_features, self.out_features), initializer='glorot_uniform', trainable=True, name=f'{self.name_prefix}_W')
        self.a = self.add_weight(shape=(2 * self.out_features, 1), initializer='glorot_uniform', trainable=True, name=f'{self.name_prefix}_a')
        super().build(input_shape)

    def call(self, inputs, training=False):
        h, adj = inputs
        Wh = tf.matmul(h, self.W)
        Wh1 = tf.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = tf.matmul(Wh, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + tf.transpose(Wh2, perm=[0, 2, 1]))
        attention = tf.where(adj > 0, e, -9e15 * tf.ones_like(e))
        attention = tf.nn.softmax(attention, axis=-1)
        attention = self.attn_dropout(attention, training=training)
        h_prime = tf.matmul(attention, Wh)
        return tf.nn.elu(h_prime) if self.concat else h_prime

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(in_features=self.in_features, out_features=self.out_features, dropout=self.dropout_rate, activation=self.activation_name, concat=self.concat, name_prefix=self.name_prefix))
        return cfg

class GAT(Layer):
    def __init__(self, in_feature, out_feature, dropout, nheads, name_prefix='', **kwargs):
        if 'name' not in kwargs: kwargs['name'] = name_prefix
        super().__init__(**kwargs)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dropout_rate = dropout
        self.nheads = nheads
        self.name_prefix = name_prefix
        self.attentions = [GraphAttentionLayer(in_feature, out_feature, dropout, name_prefix=f'{name_prefix}_att_{i}') for i in range(nheads)]
        self.out_att = GraphAttentionLayer(out_feature * nheads, out_feature, dropout, concat=False, name_prefix=f'{name_prefix}_out_att')
        self.feat_drop1 = layers.Dropout(dropout)
        self.feat_drop2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x, adj = inputs
        x = self.feat_drop1(x, training=training)
        x = tf.concat([att([x, adj], training=training) for att in self.attentions], axis=-1)
        x = self.feat_drop2(x, training=training)
        return tf.nn.elu(self.out_att([x, adj], training=training))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(in_feature=self.in_feature, out_feature=self.out_feature, dropout=self.dropout_rate, nheads=self.nheads, name_prefix=self.name_prefix))
        return cfg

class GCN_GATLayer(Layer):
    def __init__(self, in_feature, out_feature, dropout, nheads, lamda, alpha, name_prefix='', **kwargs):
        if 'name' not in kwargs: kwargs['name'] = name_prefix
        super().__init__(**kwargs)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dropout_rate = dropout
        self.nheads = nheads
        self.lamda = lamda
        self.alpha = alpha
        self.name_prefix = name_prefix
        self.gat_layers = GAT(in_feature, out_feature, dropout, nheads, name_prefix=f'{name_prefix}_gat')
        self.gcn_layer = GraphConvolutionLayer(out_feature, activation='relu', name_prefix=f'{name_prefix}_gcn')
        self.cat_lin = layers.Dense(out_feature, name=f'{name_prefix}_cat_lin')
        self.residual_proj = layers.Dense(out_feature, name=f'{name_prefix}_res_proj') if in_feature != out_feature else None
        self.res_add = layers.Add(name=f'{name_prefix}_res_add')

    def call(self, inputs, training=False):
        x, adj, h0, lamda, alpha, l = inputs
        identity = x
        gat_out = self.gat_layers([x, adj], training=training)
        gcn_out = self.gcn_layer([x, adj, h0, lamda, alpha, l])
        merged = self.cat_lin(tf.concat([gat_out, gcn_out], axis=-1))
        if self.residual_proj is not None:
            output = self.res_add([merged, self.residual_proj(identity)])
        else:
            output = self.res_add([merged, identity])
        return output

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(in_feature=self.in_feature, out_feature=self.out_feature, dropout=self.dropout_rate, nheads=self.nheads, lamda=self.lamda, alpha=self.alpha, name_prefix=self.name_prefix))
        return cfg

def gated_fusion(input_front, input_last, target_dim=128, prefix=''):
    x_front = Dropout(0.1, name=f'{prefix}front_drop')(input_front)
    x_last  = Dropout(0.1, name=f'{prefix}last_drop')(input_last)
    feat_front = Dense(target_dim, activation='relu', name=f'{prefix}front_proj')(x_front)
    feat_last  = Dense(target_dim, activation='relu', name=f'{prefix}last_proj')(x_last)
    concat_feats = Concatenate(name=f'{prefix}gate_concat')([feat_front, feat_last])
    gate = Dense(target_dim, activation='sigmoid', name=f'{prefix}gate')(concat_feats)
    gated_front = Multiply(name=f'{prefix}gated_front')([gate, feat_front])
    gated_last  = Multiply(name=f'{prefix}gated_last')([InverseLayer(name=f'{prefix}inverse')(gate), feat_last])
    fused = Add(name=f'{prefix}fused')([gated_front, gated_last])
    return fused

def build_graph_branch(esm_input, kf_front_input, kf_last_input, cmap_input, cfg, prefix):
    nh = cfg['nh']; heads = cfg['heads']; dp = cfg['dp']
    lamda = cfg['lamda']; alpha = cfg['alpha']; l = cfg['l']; nlayers = cfg['nlayers']
    l2_reg = tf.keras.regularizers.l2(1e-4)

    x_esm = Dropout(0.1, name=f'{prefix}esm_drop')(esm_input)
    x_kf = gated_fusion(kf_front_input, kf_last_input, target_dim=128, prefix=f'{prefix}kf_')
    node_feat = Concatenate(name=f'{prefix}node')([x_esm, x_kf])
    
    h0 = Dense(nh, activation='relu', kernel_regularizer=l2_reg, name=f'{prefix}h0_proj')(node_feat)
    h = Dropout(0.1, name=f'{prefix}h0_drop')(h0)

    for i in range(nlayers):
        h = Dropout(dp, name=f'{prefix}dp_{i}')(h)
        gcl = GCN_GATLayer(nh, nh, dp, heads, lamda, alpha, name_prefix=f'{prefix}gcl_{i}')
        h = LayerNormalization(name=f'{prefix}ln_{i}')(
                tf.keras.layers.Activation('relu', name=f'{prefix}relu_{i}')(
                    gcl([h, cmap_input, h0, lamda, alpha, l])))

    pooled = GlobalAveragePooling1D(name=f'{prefix}avg_pool')(h)
    branch_out = Dense(32, activation='relu', kernel_regularizer=l2_reg, name=f'{prefix}fc')(
        Dropout(0.5, name=f'{prefix}dp_fin')(pooled))
    return branch_out

def build_gated_gcngat_model(dim1, dim2, kf_dim, p1_kf_dim, fold, cfg):
    P = f'f{fold}_'
    l2_reg = tf.keras.regularizers.l2(1e-4)

    inp_knn_esm  = Input(shape=(dim1, 128), name=f'{P}knn_esm')
    inp_knn_kf_f = Input(shape=(dim1, kf_dim), name=f'{P}knn_KF_front')
    inp_knn_kf_l = Input(shape=(dim1, kf_dim), name=f'{P}knn_KF_last')
    inp_knn_cmap = Input(shape=(None, None), name=f'{P}knn_cmap')

    inp_p1_esm  = Input(shape=(dim2, 128), name=f'{P}p1_esm')
    inp_p1_kf_f = Input(shape=(dim2, p1_kf_dim), name=f'{P}p1_KF_front')
    inp_p1_kf_l = Input(shape=(dim2, p1_kf_dim), name=f'{P}p1_KF_last')
    inp_p1_cmap = Input(shape=(None, None), name=f'{P}p1_cmap')

    knn_out = build_graph_branch(inp_knn_esm, inp_knn_kf_f, inp_knn_kf_l, inp_knn_cmap, cfg, prefix=f'{P}knn_')
    p1_out  = build_graph_branch(inp_p1_esm, inp_p1_kf_f, inp_p1_kf_l, inp_p1_cmap, cfg, prefix=f'{P}p1_')

    merged = Dropout(0.1, name=f'{P}merge_dp')(Concatenate(name=f'{P}merge_cat')([knn_out, p1_out]))
    output = Dense(1, activation='sigmoid', kernel_regularizer=l2_reg, name=f'{P}output')(merged)

    return models.Model(inputs=[inp_knn_esm, inp_knn_kf_f, inp_knn_kf_l, inp_knn_cmap,
                                inp_p1_esm, inp_p1_kf_f, inp_p1_kf_l, inp_p1_cmap],
                        outputs=output, name=f'{P}GatedGCNGAT')

def get_custom_objects():
    from utils.loss import FocalLoss
    return {
        'InverseLayer': InverseLayer,
        'GraphConvolutionLayer': GraphConvolutionLayer,
        'GraphAttentionLayer': GraphAttentionLayer,
        'GAT': GAT,
        'GCN_GATLayer': GCN_GATLayer,
        'FocalLoss': FocalLoss
    }
