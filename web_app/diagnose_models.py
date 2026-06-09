import os, re, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class QuantumSuperposition(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.W_real = self.add_weight('W_real', (input_shape[-1], self.units), initializer='glorot_uniform')
        self.W_imag = self.add_weight('W_imag', (input_shape[-1], self.units), initializer='glorot_uniform')
        self.bias = self.add_weight('bias', (self.units,), initializer='zeros')
    def call(self, x):
        real = tf.matmul(x, self.W_real)
        imag = tf.matmul(x, self.W_imag)
        magnitude = tf.sqrt(tf.square(real) + tf.square(imag) + 1e-8)
        return tf.nn.relu(magnitude + self.bias)
    def get_config(self):
        return {**super().get_config(), 'units': self.units}

class QuantumInterference(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.phase = self.add_weight('phase', (input_shape[-1],), initializer='zeros', trainable=True)
        self.gamma = self.add_weight('gamma', (input_shape[-1],), initializer='ones', trainable=True)
    def call(self, x):
        return x * tf.cos(self.phase) + self.gamma * tf.sin(self.phase)

class QuantumMeasurement(tf.keras.layers.Layer):
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
    def build(self, input_shape):
        self.basis = self.add_weight('basis', (input_shape[-1], self.n_classes), initializer='glorot_uniform')
    def call(self, x):
        return tf.nn.softmax(tf.matmul(x, self.basis))
    def get_config(self):
        return {**super().get_config(), 'n_classes': self.n_classes}

class QuantumEntanglement(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        n = input_shape[-1]
        self.W_q = self.add_weight('W_q', (n, n//4), initializer='glorot_uniform')
        self.W_k = self.add_weight('W_k', (n, n//4), initializer='glorot_uniform')
        self.W_v = self.add_weight('W_v', (n, n), initializer='glorot_uniform')
    def call(self, x):
        q = tf.matmul(x, self.W_q)
        k = tf.matmul(x, self.W_k)
        v = tf.matmul(x, self.W_v)
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(q.shape[-1])))
        scale = tf.reduce_mean(attn, axis=-1, keepdims=True)
        return x + scale * v

custom_objects = {
    'QuantumSuperposition': QuantumSuperposition,
    'QuantumInterference': QuantumInterference,
    'QuantumMeasurement': QuantumMeasurement,
    'QuantumEntanglement': QuantumEntanglement
}

def patch_h5(src, dst):
    import h5py
    shutil.copy2(src, dst)
    pattern = re.compile(r',?\s*"time_major":\s*(true|false)')
    with h5py.File(dst, 'r+') as f:
        raw = f.attrs.get('model_config')
        if raw:
            cfg = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
            patched = pattern.sub('', cfg)
            f.attrs['model_config'] = patched.encode('utf-8')
    return dst

def test_model(path, custom_objects=None):
    try:
        if custom_objects:
            m = tf.keras.models.load_model(path, custom_objects=custom_objects)
        else:
            m = tf.keras.models.load_model(path)
        shape = m.input_shape[1:]
        p1 = m.predict(np.random.randn(1, *shape).astype(np.float32), verbose=0)[0]
        p2 = m.predict(np.random.randn(1, *shape).astype(np.float32), verbose=0)[0]
        diff = float(abs(p1-p2).max())
        status = "DISCRIMINATING" if diff > 0.05 else "DEAD"
        print(f"  OK  [{status}]  input={m.input_shape}  max_diff={diff:.4f}")
        print(f"       Pred1: {p1.round(3)}")
        print(f"       Pred2: {p2.round(3)}")
        return m, status
    except Exception as e:
        print(f"  FAIL: {e}")
        return None, "FAIL"

models_to_test = [
    ('model_file/epilepsy_quantum_model.h5', custom_objects),
    ('model_file/best_model.h5', custom_objects),
    ('trainmodel_file/epilepsy_quantum_model.h5', custom_objects),
    ('trainmodel_file/best_model.h5', custom_objects),
    ('train3saved_model/qinn_v3_final.keras', None),
]

print("=" * 70)
print("MODEL DIAGNOSTIC")
print("=" * 70)
for rel_path, co in models_to_test:
    full_path = os.path.join('..', rel_path)
    print(f"\nModel: {rel_path}")
    if not os.path.exists(full_path):
        print("  NOT FOUND")
        continue
    # Try direct load first
    m, status = test_model(full_path, co)
    if status == "FAIL" and rel_path.endswith('.h5'):
        # Try patching for time_major
        print("  Retrying with time_major patch...")
        try:
            patched = patch_h5(full_path, full_path + '.patched.h5')
            m, status = test_model(patched, co)
        except Exception as e:
            print(f"  Patch failed: {e}")

print("\n" + "=" * 70)
print("DONE")
