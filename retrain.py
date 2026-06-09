"""
================================================================================
QUICK RETRAIN — QINN for Epilepsy Prediction
================================================================================
Trains the final QINN model on all available data with proper augmentation.
Saves as both .keras (new format) and .h5 (legacy format) for the web app.
Expected time: 5-15 minutes depending on CPU.
================================================================================
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '42'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import scipy.signal as scipy_signal
import numpy as np

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Config ───────────────────────────────────────────────────────────────────
CLASSES       = ["NORMAL", "ICTAL", "PREICTAL", "POSTICTAL"]
NUM_CLASSES   = 4
SIGNAL_LENGTH = 1000   # samples fed to model
NUM_CHANNELS  = 23     # EEG channels used
BATCH_SIZE    = 32
EPOCHS        = 80
SAMPLING_RATE = 256
SAVE_DIR      = "model_file"

print("=" * 60)
print("  QINN — Quick Retrain")
print(f"  TensorFlow {tf.__version__}")
print(f"  SIGNAL_LENGTH={SIGNAL_LENGTH}  NUM_CHANNELS={NUM_CHANNELS}")
print("=" * 60)

# ─── EEG Preprocessing ────────────────────────────────────────────────────────

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

def preprocess_eeg(data):
    """Full preprocessing pipeline matching the web app."""
    if data is None or data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n, c = data.shape
    # Channel fix
    if c > NUM_CHANNELS:
        data = data[:, :NUM_CHANNELS]
    elif c < NUM_CHANNELS:
        data = np.pad(data, ((0, 0), (0, NUM_CHANNELS - c)), mode='constant')

    # Filter
    try:
        data = bandpass_filter(data)
        data = notch_filter(data)
    except Exception:
        pass  # skip if too short

    # Length fix: center-crop / edge-pad
    n = data.shape[0]
    if n > SIGNAL_LENGTH:
        start = (n - SIGNAL_LENGTH) // 2
        data = data[start:start + SIGNAL_LENGTH]
    elif n < SIGNAL_LENGTH:
        data = np.pad(data, ((0, SIGNAL_LENGTH - n), (0, 0)), mode='edge')

    # Z-score normalize per channel
    mu = data.mean(axis=0, keepdims=True)
    sigma = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - mu) / sigma

    # Clip extreme values
    data = np.clip(data, -10, 10)
    return data.astype(np.float32)

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_file(fp):
    try:
        df = pd.read_csv(fp, header=None)
        try:
            float(df.iloc[0, 0])
        except (ValueError, TypeError):
            df = df.iloc[1:]
        return df.values.astype(np.float32)
    except Exception:
        return None

def load_dataset(dataset_dir="dataset"):
    print("\n[1/4] Loading and preprocessing dataset...")
    X_list, y_list = [], []
    for patient_dir in sorted(Path(dataset_dir).iterdir()):
        if not patient_dir.is_dir():
            continue
        for ci, cn in enumerate(CLASSES):
            class_dir = patient_dir / cn
            if not class_dir.exists():
                continue
            for f in class_dir.iterdir():
                if f.suffix.lower() not in ['.xls', '.xlsx', '.csv']:
                    continue
                raw = load_file(str(f))
                if raw is None:
                    continue
                processed = preprocess_eeg(raw)
                if processed is not None and processed.shape == (SIGNAL_LENGTH, NUM_CHANNELS):
                    X_list.append(processed)
                    y_list.append(ci)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"  Loaded: {len(X)} samples  Distribution: {dict(Counter(y_list))}")
    print(f"  Shape: {X.shape}  Label dist: {[CLASSES[i] for i in range(4)]}")
    return X, y

# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment(X, y, factor=8):
    Xa, ya = [X], [y]
    for i in range(factor - 1):
        a = X.copy()
        choice = i % 5
        if choice == 0:
            a += np.random.normal(0, 0.05, a.shape).astype(np.float32)
        elif choice == 1:
            a *= np.random.uniform(0.9, 1.1)
        elif choice == 2:
            a = -a
        elif choice == 3:
            a = np.roll(a, np.random.randint(-30, 30), axis=1)
        else:
            # Channel dropout
            drop_ch = np.random.choice(NUM_CHANNELS, size=NUM_CHANNELS // 4, replace=False)
            a[:, :, drop_ch] = 0.0
        Xa.append(a.astype(np.float32))
        ya.append(y)
    Xout = np.concatenate(Xa)
    yout = np.concatenate(ya)
    idx = np.random.permutation(len(Xout))
    return Xout[idx], yout[idx]

# ─── Quantum-Inspired Layers ──────────────────────────────────────────────────

class QuantumSuperposition(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        d = input_shape[-1]
        self.W_real = self.add_weight(name="W_real", shape=(d, self.units), initializer='glorot_uniform')
        self.W_imag = self.add_weight(name="W_imag", shape=(d, self.units), initializer='glorot_uniform')
        self.bias   = self.add_weight(name="bias",   shape=(self.units,),    initializer='zeros')
    def call(self, x):
        real = tf.matmul(x, self.W_real)
        imag = tf.matmul(x, self.W_imag)
        mag  = tf.sqrt(tf.square(real) + tf.square(imag) + 1e-8)
        return tf.nn.relu(mag + self.bias)
    def get_config(self):
        return {**super().get_config(), "units": self.units}

class QuantumInterference(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        d = input_shape[-1]
        self.phase = self.add_weight(name="phase", shape=(d,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name="gamma", shape=(d,), initializer='ones',  trainable=True)
    def call(self, x):
        return x * tf.cos(self.phase) + self.gamma * tf.sin(self.phase)

class QuantumEntanglement(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        n = input_shape[-1]
        k = max(n // 4, 4)
        self.W_q   = self.add_weight(name="W_q",   shape=(n, k), initializer='glorot_uniform')
        self.W_k   = self.add_weight(name="W_k",   shape=(n, k), initializer='glorot_uniform')
        self.W_v   = self.add_weight(name="W_v",   shape=(n, n), initializer='glorot_uniform')
        self.alpha = self.add_weight(name="alpha",  shape=(),     initializer='zeros', trainable=True)
    def call(self, x):
        q = tf.matmul(x, self.W_q)
        k = tf.matmul(x, self.W_k)
        v = tf.matmul(x, self.W_v)
        dk   = tf.cast(tf.shape(q)[-1], tf.float32)
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(dk))
        scale = tf.reduce_mean(attn, axis=-1, keepdims=True)
        gate  = tf.nn.sigmoid(self.alpha)
        return x + gate * scale * v

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

# ─── Model ────────────────────────────────────────────────────────────────────

def build_model():
    inp = layers.Input((SIGNAL_LENGTH, NUM_CHANNELS))
    # CNN backbone
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Quantum layers
    x = QuantumSuperposition(128, name='superposition')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = QuantumInterference(name='interference')(x)
    x = QuantumEntanglement(name='entanglement')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = QuantumMeasurement(NUM_CLASSES, name='measurement')(x)

    # Dense head for stability
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inp, out, name='QINN_Retrained')

# ─── Train ────────────────────────────────────────────────────────────────────

X, y = load_dataset()

if len(X) == 0:
    print("ERROR: No data loaded!")
    sys.exit(1)

print("\n[2/4] Augmenting data...")
X_aug, y_aug = augment(X, y, factor=8)
print(f"  After augmentation: {len(X_aug)} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, stratify=y_train, random_state=SEED)

print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# Class weights for imbalance handling
cw = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
print(f"  Class weights: {cw}")

y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_oh   = keras.utils.to_categorical(y_val,   NUM_CLASSES)

print("\n[3/4] Training model...")
model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=1),
]

t0 = time.time()
history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=cw,
    callbacks=callbacks,
    verbose=1
)
elapsed = time.time() - t0
print(f"\n  Training done in {elapsed:.1f}s")

# ─── Evaluate ─────────────────────────────────────────────────────────────────

print("\n[4/4] Evaluating...")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\n  Test Accuracy: {acc*100:.2f}%")
print()
print(classification_report(y_test, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_test, y_pred, labels=range(NUM_CLASSES))
print("Confusion Matrix:")
header = "%-12s" % "" + "".join(["%-12s" % c for c in CLASSES])
print(header)
for i, row in enumerate(cm):
    print("%-12s" % CLASSES[i] + "".join(["%-12d" % v for v in row]))

# ─── Save ─────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

# Save as H5 (for web app compatibility)
h5_path = os.path.join(SAVE_DIR, "epilepsy_quantum_model.h5")
custom_objects = {
    'QuantumSuperposition': QuantumSuperposition,
    'QuantumInterference': QuantumInterference,
    'QuantumMeasurement': QuantumMeasurement,
    'QuantumEntanglement': QuantumEntanglement,
}
model.save(h5_path)
print(f"\n  Saved H5: {h5_path}")

# Save label mapping
label_map = {str(i): n for i, n in enumerate(CLASSES)}
with open(os.path.join(SAVE_DIR, "label_mapping.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# Save results
results = {
    "test_accuracy": float(acc),
    "training_samples": int(len(X_train)),
    "val_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "signal_length": SIGNAL_LENGTH,
    "num_channels": NUM_CHANNELS,
    "classes": CLASSES,
}
with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"  DONE — Test Accuracy: {acc*100:.2f}%")
print(f"  Model: {h5_path}")
print(f"{'='*60}")
