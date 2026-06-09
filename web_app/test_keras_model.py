"""Test the train3saved qinn_v3_final.keras model with proper registration."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers

SIGNAL_LENGTH = 500
NUM_CHANNELS = 16
CLASSES = ['NORMAL', 'ICTAL', 'PREICTAL', 'POSTICTAL']

@tf.keras.utils.register_keras_serializable()
class QuantumSuperposition(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        d = input_shape[-1]
        self.W_real = self.add_weight(name="W_real", shape=(d, self.units), initializer='glorot_uniform')
        self.W_imag = self.add_weight(name="W_imag", shape=(d, self.units), initializer='glorot_uniform')
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer='zeros')
    def call(self, x):
        real = tf.matmul(x, self.W_real)
        imag = tf.matmul(x, self.W_imag)
        magnitude = tf.sqrt(tf.square(real) + tf.square(imag) + 1e-8)
        return tf.nn.relu(magnitude + self.bias)
    def get_config(self):
        return {**super().get_config(), "units": self.units}

@tf.keras.utils.register_keras_serializable()
class QuantumInterference(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        d = input_shape[-1]
        self.phase = self.add_weight(name="phase", shape=(d,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name="gamma", shape=(d,), initializer='ones', trainable=True)
    def call(self, x):
        return x * tf.cos(self.phase) + self.gamma * tf.sin(self.phase)

@tf.keras.utils.register_keras_serializable()
class QuantumEntanglement(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        n = input_shape[-1]
        k = max(n // 4, 4)
        self.W_q = self.add_weight(name="W_q", shape=(n, k), initializer='glorot_uniform')
        self.W_k = self.add_weight(name="W_k", shape=(n, k), initializer='glorot_uniform')
        self.W_v = self.add_weight(name="W_v", shape=(n, n), initializer='glorot_uniform')
        self.alpha = self.add_weight(name="alpha", shape=(), initializer='zeros', trainable=True)
    def call(self, x):
        q = tf.matmul(x, self.W_q)
        k = tf.matmul(x, self.W_k)
        v = tf.matmul(x, self.W_v)
        dk = tf.cast(tf.shape(q)[-1], tf.float32)
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(dk))
        scale = tf.reduce_mean(attn, axis=-1, keepdims=True)
        gate = tf.nn.sigmoid(self.alpha)
        return x + gate * scale * v

@tf.keras.utils.register_keras_serializable()
class QuantumMeasurement(layers.Layer):
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
    def build(self, input_shape):
        self.basis = self.add_weight(name="basis", shape=(input_shape[-1], self.n_classes), initializer='glorot_uniform')
    def call(self, x):
        return tf.nn.softmax(tf.matmul(x, self.basis))
    def get_config(self):
        return {**super().get_config(), "n_classes": self.n_classes}


print("Loading train3saved_model/qinn_v3_final.keras ...")
try:
    model = tf.keras.models.load_model('../train3saved_model/qinn_v3_final.keras')
    print("Loaded OK")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    
    # Test on random noise
    shape = model.input_shape[1:]
    for i in range(5):
        t = np.random.randn(1, *shape).astype(np.float32)
        p = model.predict(t, verbose=0)[0]
        print(f"  Noise test {i+1}: {p.round(3)}  -> {CLASSES[np.argmax(p)]} ({np.max(p)*100:.1f}%)")
    
    print()
    # Now test on actual dataset files
    import pandas as pd
    from pathlib import Path
    import scipy.signal as scipy_signal
    
    SAMPLING_RATE = 256
    
    def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=SAMPLING_RATE, order=4):
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, data, axis=0).astype(np.float32)
    
    def notch_filter(data, freq=50.0, fs=SAMPLING_RATE, Q=30):
        nyq = 0.5 * fs
        if freq >= nyq:
            return data
        b, a = scipy_signal.iirnotch(freq / nyq, Q)
        return scipy_signal.filtfilt(b, a, data, axis=0).astype(np.float32)
    
    def preprocess(data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n, c = data.shape
        c_use = min(c, NUM_CHANNELS)
        if c_use < NUM_CHANNELS:
            data = np.pad(data[:, :c_use], ((0, 0), (0, NUM_CHANNELS - c_use)), mode='constant')
        else:
            data = data[:, :NUM_CHANNELS]
        try:
            data = bandpass_filter(data)
            data = notch_filter(data)
        except:
            pass
        n = data.shape[0]
        if n > SIGNAL_LENGTH:
            start = (n - SIGNAL_LENGTH) // 2
            data = data[start:start + SIGNAL_LENGTH]
        elif n < SIGNAL_LENGTH:
            data = np.pad(data, ((0, SIGNAL_LENGTH - n), (0, 0)), mode='edge')
        mu = data.mean(axis=0, keepdims=True)
        sigma = data.std(axis=0, keepdims=True) + 1e-8
        data = (data - mu) / sigma
        return data.astype(np.float32)
    
    dataset_dir = Path('../dataset')
    results = []
    for patient_dir in sorted(dataset_dir.iterdir())[:3]:
        if not patient_dir.is_dir():
            continue
        for class_name in CLASSES:
            class_dir = patient_dir / class_name
            if not class_dir.exists():
                continue
            for f in list(class_dir.iterdir())[:2]:
                try:
                    df = pd.read_csv(str(f), header=None)
                    try:
                        float(df.iloc[0, 0])
                    except:
                        df = df.iloc[1:]
                    data = df.values.astype(np.float32)
                    data = preprocess(data)
                    inp = np.expand_dims(data, 0)
                    pred = model.predict(inp, verbose=0)[0]
                    predicted = CLASSES[np.argmax(pred)]
                    confidence = float(np.max(pred)) * 100
                    results.append((f.name[:28], class_name, predicted, confidence, pred))
                except Exception as e:
                    print('  Error:', f.name, '->', e)
    
    print("%-30s %-12s %-12s %6s" % ("File", "True", "Pred", "Conf"))
    print("-" * 75)
    correct = 0
    for fname, true_cls, pred_cls, conf, probs in results:
        match = "OK" if true_cls == pred_cls else "XX"
        if true_cls == pred_cls:
            correct += 1
        print("%-30s %-12s %-12s %5.1f%%  %s  [N=%.2f I=%.2f PRE=%.2f POST=%.2f]" % (
            fname, true_cls, pred_cls, conf, match, probs[0], probs[1], probs[2], probs[3]))
    
    print()
    print("Accuracy: %d/%d = %.1f%%" % (correct, len(results), 100 * correct / max(len(results), 1)))

except Exception as e:
    import traceback
    print("FAILED:", e)
    traceback.print_exc()
