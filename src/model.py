
import tensorflow as tf

def make_atomic_nn(
    desc_dim,
    hidden=(64, 64),
    activation="tanh",
    name="atomic_nn",
    l2_reg=1e-4,          
    dropout_rate=0.05      
):
    reg = tf.keras.regularizers.l2(l2_reg) if (l2_reg and l2_reg > 0) else None

    inp = tf.keras.layers.Input(shape=(desc_dim,))
    x = inp

    for h in hidden:
        x = tf.keras.layers.Dense(
            h,
            activation=activation,
            kernel_regularizer=reg,
            bias_regularizer=None
        )(x)

        if dropout_rate and dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    out = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_regularizer=reg
    )(x)

    return tf.keras.Model(inp, out, name=name)

class BPHDNNP(tf.keras.Model):
    def __init__(self, n_elements, desc_dim, hidden=(64, 64), l2_reg=1e-6, dropout_rate=0.0):
        super().__init__()
        self.n_elements = n_elements
        self.desc_dim = desc_dim
        self.atomic_nns = [
            make_atomic_nn(
                desc_dim,
                hidden=hidden,
                name=f"nn_elem_{e}",
                l2_reg=l2_reg,
                dropout_rate=dropout_rate
            )
            for e in range(n_elements)
        ]

    def call(self, inputs):
        X, Z = inputs
        B = tf.shape(X)[0]
        N = tf.shape(X)[1]

        X_flat = tf.reshape(X, (-1, self.desc_dim))
        Z_flat = tf.reshape(Z, (-1,))

        E_flat = tf.zeros((tf.shape(X_flat)[0], 1), dtype=tf.float32)

        for elem_id, nn in enumerate(self.atomic_nns):
            idx = tf.where(tf.equal(Z_flat, elem_id))
            X_sel = tf.gather_nd(X_flat, idx)
            E_sel = nn(X_sel)
            E_flat = tf.tensor_scatter_nd_update(E_flat, idx, E_sel)

        E_atoms = tf.reshape(E_flat, (B, N, 1))
        E_total = tf.reduce_sum(E_atoms, axis=1)
        return tf.squeeze(E_total, axis=-1)