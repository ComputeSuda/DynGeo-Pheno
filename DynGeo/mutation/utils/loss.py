import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.55):
    """Focal Loss for binary classification."""
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.math.pow(1.0 - p_t, gamma)
        loss = alpha_factor * modulating_factor * cross_entropy
        return tf.math.reduce_mean(loss)
    return focal_loss_fn
