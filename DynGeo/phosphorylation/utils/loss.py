import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        bce = -(y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(focal_weight * bce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(gamma=self.gamma, alpha=self.alpha))
        return cfg


def create_optimizer(cfg):
    lr = cfg['lr']
    wd = cfg.get('weight_decay', 0)
    cn = cfg.get('clipnorm', 1.0)
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=wd, clipnorm=cn), 'AdamW'
    except AttributeError:
        pass
    try:
        return tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr, weight_decay=wd, clipnorm=cn), 'AdamW(exp)'
    except AttributeError:
        pass
    return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=cn), 'Adam'
