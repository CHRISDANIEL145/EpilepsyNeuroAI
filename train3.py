"""
================================================================================
QINN v3 — Quantum-Inspired Neural Network for Epileptic Seizure Prediction
================================================================================
Upgrades over v2:
  - Proper EEG preprocessing (bandpass, notch, artifact rejection)
  - Leave-One-Patient-Out Cross Validation (LOPO-CV)
  - PennyLane hybrid quantum-classical feature projection layer
  - Ablation study (4 model variants)
  - Medical metrics (ROC, AUC, Sensitivity, Specificity)
  - Grad-CAM explainability
  - Statistical validation (5-run, t-test, 95% CI)
  - Real-time feasibility benchmarking
  - Full reproducibility (seeds, hardware, versions)
================================================================================
"""

import os, sys, json, warnings, time, tracemalloc, platform
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_fscore_support,
                             accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
import scipy
from scipy import signal as scipy_signal
from scipy import stats

# Optional PennyLane
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("[WARN] PennyLane not installed. Hybrid quantum layer disabled.")

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
GLOBAL_SEED = 42

def set_seeds(seed=GLOBAL_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

# ============================================================================
# CONFIG
# ============================================================================
CLASSES = ["NORMAL", "ICTAL", "PREICTAL", "POSTICTAL"]
NUM_CLASSES = 4
SIGNAL_LENGTH = 500
NUM_CHANNELS = 16
BATCH_SIZE = 256
TARGET_ACC = 0.96
SAVE_DIR = "train3saved_model_v3"
RESULTS_DIR = "results_v3"
SAMPLING_RATE = 256  # CHB-MIT native Hz
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def log_environment():
    """Log hardware and software environment for reproducibility."""
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__ if hasattr(scipy, '__version__') else "installed",
        "pennylane": qml.__version__ if HAS_PENNYLANE else "N/A",
        "gpu_devices": [d.name for d in tf.config.list_physical_devices('GPU')],
        "cpu_count": os.cpu_count(),
        "global_seed": GLOBAL_SEED,
    }
    with open(f"{RESULTS_DIR}/environment.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  GPUs: {info['gpu_devices'] or 'None (CPU only)'}")
    print(f"  PennyLane: {info['pennylane']}")
    return info

# ============================================================================
# EEG PREPROCESSING (Section 8)
# ============================================================================

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=SAMPLING_RATE, order=4):
    """Apply 0.5-40 Hz Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clamp to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, data, axis=0).astype(np.float32)

def notch_filter(data, freq=50.0, fs=SAMPLING_RATE, Q=30):
    """Apply 50 Hz notch filter to remove power-line interference."""
    nyq = 0.5 * fs
    if freq >= nyq:
        return data
    b, a = scipy_signal.iirnotch(freq / nyq, Q)
    return scipy_signal.filtfilt(b, a, data, axis=0).astype(np.float32)

def reject_artifacts(data, threshold_uv=500.0):
    """Simple amplitude-threshold artifact rejection.
    Returns None if > 30% of samples exceed threshold."""
    if np.mean(np.abs(data) > threshold_uv) > 0.3:
        return None
    # Clip extreme values
    return np.clip(data, -threshold_uv, threshold_uv).astype(np.float32)

def segment_to_fixed_length(data, target_len=SIGNAL_LENGTH):
    """Window segmentation to fixed length."""
    n = data.shape[0]
    if n > target_len:
        # Take center segment
        start = (n - target_len) // 2
        data = data[start:start + target_len]
    elif n < target_len:
        data = np.pad(data, ((0, target_len - n), (0, 0)), mode='edge')
    return data

def normalize_channels(data):
    """Channel-wise z-score normalization."""
    mu = data.mean(axis=0, keepdims=True)
    sigma = data.std(axis=0, keepdims=True) + 1e-8
    return ((data - mu) / sigma).astype(np.float32)

def preprocess_eeg(data, fs=SAMPLING_RATE):
    """Complete EEG preprocessing pipeline."""
    if data is None or len(data) == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    # Standardize channels
    n, c = data.shape
    c = min(c, NUM_CHANNELS)
    if c < NUM_CHANNELS:
        data = np.pad(data, ((0, 0), (0, NUM_CHANNELS - c)), mode='constant')
    else:
        data = data[:, :NUM_CHANNELS]
    # 1. Bandpass filter 0.5-40 Hz
    try:
        data = bandpass_filter(data, fs=fs)
    except Exception:
        pass  # If filter fails (too short), skip
    # 2. Notch filter 50 Hz
    try:
        data = notch_filter(data, fs=fs)
    except Exception:
        pass
    # 3. Artifact rejection
    data = reject_artifacts(data)
    if data is None:
        return None
    # 4. Segment to fixed length
    data = segment_to_fixed_length(data)
    # 5. Normalize
    data = normalize_channels(data)
    return data

# ============================================================================
# QUANTUM-INSPIRED LAYERS
# ============================================================================

class QuantumSuperposition(layers.Layer):
    """Complex-valued dual-weight representation (amplitude-phase encoding)."""
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


class QuantumInterference(layers.Layer):
    """Phase-based constructive/destructive feature gating."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]
        self.phase = self.add_weight(name="phase", shape=(d,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name="gamma", shape=(d,), initializer='ones', trainable=True)

    def call(self, x):
        return x * tf.cos(self.phase) + self.gamma * tf.sin(self.phase)


class QuantumEntanglement(layers.Layer):
    """Non-local cross-dimensional feature correlation via scaled self-attention."""
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


class QuantumMeasurement(layers.Layer):
    """Born rule probability projection to class labels."""
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes

    def build(self, input_shape):
        self.basis = self.add_weight(name="basis", shape=(input_shape[-1], self.n_classes),
                                     initializer='glorot_uniform')

    def call(self, x):
        projection = tf.matmul(x, self.basis)
        return tf.nn.softmax(projection)

    def get_config(self):
        return {**super().get_config(), "n_classes": self.n_classes}


# ============================================================================
# PENNYLANE HYBRID QUANTUM LAYER (Section 9)
# ============================================================================

if HAS_PENNYLANE:
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface=None)
    def quantum_circuit(inputs, weights):
        """4-qubit variational circuit for nonlinear feature projection.
        Uses AngleEmbedding for data encoding and StronglyEntanglingLayers
        for parameterized unitary evolution with CNOT entanglement."""
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class QuantumFeatureProjection(layers.Layer):
        """Hybrid quantum-classical feature projection layer.
        Classical CNN features -> linear projection -> 4-qubit variational circuit
        -> Pauli-Z measurements -> linear projection back.
        The quantum circuit uses:
          - AngleEmbedding: encodes classical features as qubit rotation angles
          - StronglyEntanglingLayers: parameterized rotations + CNOT entanglement gates
          - PauliZ measurements: Born rule probability extraction
        This is a nonlinear feature mapping simulated on classical hardware.
        No quantum speedup is claimed."""
        def __init__(self, n_qubits=4, n_layers=2, **kwargs):
            super().__init__(**kwargs)
            self.n_qubits = n_qubits
            self.n_qlayers = n_layers

        def build(self, input_shape):
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=self.n_qlayers, n_wires=self.n_qubits)
            self.q_weights = self.add_weight(
                name="q_weights", shape=weight_shape,
                initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
                trainable=False)
            d = input_shape[-1]
            self.proj_down = self.add_weight(name="proj_down", shape=(d, self.n_qubits),
                                             initializer='glorot_uniform')
            self.proj_up = self.add_weight(name="proj_up", shape=(self.n_qubits, d),
                                           initializer='glorot_uniform')

        def _run_batch(self, x_np, w_np):
            """Execute quantum circuit on each sample (numpy domain)."""
            results = []
            for xi in x_np:
                out = quantum_circuit(
                    xi.astype(np.float64), w_np.astype(np.float64))
                results.append(np.array(out, dtype=np.float32))
            return np.stack(results).astype(np.float32)

        def call(self, x):
            x_proj = tf.matmul(x, self.proj_down)
            x_proj = tf.nn.tanh(x_proj) * np.pi
            q_out = tf.numpy_function(
                self._run_batch, [x_proj, self.q_weights], tf.float32)
            q_out = tf.ensure_shape(q_out, [None, self.n_qubits])
            return x + tf.matmul(q_out, self.proj_up)

        def get_config(self):
            return {**super().get_config(),
                    "n_qubits": self.n_qubits, "n_layers": self.n_qlayers}
else:
    class QuantumFeatureProjection(layers.Layer):
        """Fallback: identity layer when PennyLane is not available."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def call(self, x):
            return x


# ============================================================================
# DATA LOADING
# ============================================================================

def load_file(fp):
    try:
        df = pd.read_csv(fp, header=None)
        try:
            float(df.iloc[0, 0])
            return df.values.astype(np.float32)
        except (ValueError, IndexError):
            return df.iloc[1:].values.astype(np.float32)
    except Exception:
        return None

def load_data_by_patient(dataset_dir="dataset"):
    """Load data organized by patient for LOPO-CV.
    Returns dict: {patient_id: (X_list, y_list)}"""
    print("\n  Loading dataset by patient...")
    patient_data = {}
    for d in sorted(Path(dataset_dir).iterdir()):
        if not d.is_dir() or "chb" not in d.name.lower():
            continue
        pid = d.name.lower().replace(" data", "").replace(" ", "")
        X_p, y_p = [], []
        for ci, cn in enumerate(CLASSES):
            cd = d / cn
            if not cd.exists():
                continue
            for f in cd.iterdir():
                if f.suffix.lower() in ['.xls', '.xlsx', '.csv']:
                    data = load_file(str(f))
                    if data is not None:
                        p = preprocess_eeg(data)
                        if p is not None:
                            X_p.append(p)
                            y_p.append(ci)
        if X_p:
            patient_data[pid] = (np.array(X_p), np.array(y_p))
            print(f"    {pid}: {len(X_p)} samples {dict(Counter(y_p))}")
    print(f"  Total: {sum(len(v[0]) for v in patient_data.values())} samples from {len(patient_data)} patients")
    return patient_data

def augment(X, y, factor=15):
    """Multi-strategy data augmentation."""
    Xa, ya = [X], [y]
    for i in range(factor - 1):
        a = X.copy()
        if i % 4 == 0:
            a += np.random.normal(0, 0.05, a.shape).astype(np.float32)
        elif i % 4 == 1:
            a *= np.random.uniform(0.9, 1.1)
        elif i % 4 == 2:
            a = -a
        else:
            a = np.roll(a, np.random.randint(-20, 20), axis=1)
        Xa.append(a.astype(np.float32))
        ya.append(y)
    Xo, yo = np.concatenate(Xa), np.concatenate(ya)
    idx = np.random.permutation(len(Xo))
    return Xo[idx], yo[idx]

# ============================================================================
# MODEL BUILDERS (Ablation Study — Section 3)
# ============================================================================

def _cnn_backbone(inp):
    """Shared 3-block 1D CNN backbone."""
    x = layers.Conv1D(48, 5, padding='same', activation='relu', name='conv1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(4, name='pool1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(96, 3, padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(4, name='pool2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(96, 3, padding='same', activation='relu', name='conv3')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    return x

def build_baseline_cnn():
    """Model A: Baseline CNN only (no quantum layers)."""
    inp = layers.Input((SIGNAL_LENGTH, NUM_CHANNELS))
    x = _cnn_backbone(inp)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inp, out, name='Baseline_CNN')

def build_cnn_superposition():
    """Model B: CNN + Superposition layer."""
    inp = layers.Input((SIGNAL_LENGTH, NUM_CHANNELS))
    x = _cnn_backbone(inp)
    x = QuantumSuperposition(64, name='superposition')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inp, out, name='CNN_Superposition')

def build_cnn_interference():
    """Model C: CNN + Superposition + Interference."""
    inp = layers.Input((SIGNAL_LENGTH, NUM_CHANNELS))
    x = _cnn_backbone(inp)
    x = QuantumSuperposition(64, name='superposition')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = QuantumInterference(name='interference')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inp, out, name='CNN_Interference')

def build_full_qinn(use_pennylane=False):
    """Model D: Full QINN (all quantum-inspired layers + PennyLane circuit)."""
    inp = layers.Input((SIGNAL_LENGTH, NUM_CHANNELS))
    x = _cnn_backbone(inp)
    x = QuantumSuperposition(64, name='superposition')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = QuantumInterference(name='interference')(x)
    x = QuantumEntanglement(name='entanglement')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    if use_pennylane and HAS_PENNYLANE:
        x = QuantumFeatureProjection(n_qubits=4, n_layers=2, name='q_projection')(x)
    x = QuantumMeasurement(NUM_CLASSES, name='measurement')(x)
    # Stabilize output with Dense head after quantum measurement
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inp, out, name='QINN_Full')

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def compile_model(model, lr=0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, verbose=0):
    """Train with early stopping and class weights."""
    ytr_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    yv_oh = keras.utils.to_categorical(y_val, NUM_CLASSES)
    cw = dict(enumerate(compute_class_weight('balanced',
                                              classes=np.unique(y_train), y=y_train)))
    es = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                        restore_best_weights=True)
    history = model.fit(X_train, ytr_oh, validation_data=(X_val, yv_oh),
                        epochs=epochs, batch_size=BATCH_SIZE, class_weight=cw,
                        callbacks=[es], verbose=verbose)
    return history

# ============================================================================
# MEDICAL METRICS (Section 4)
# ============================================================================

def compute_medical_metrics(model, X_test, y_test):
    """Compute healthcare-grade evaluation metrics."""
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_oh = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Per-class metrics
    prec, rec, f1, supp = precision_recall_fscore_support(y_test, y_pred, average=None,
                                                           labels=range(NUM_CLASSES))
    acc = accuracy_score(y_test, y_pred)

    # Specificity per class (one-vs-rest)
    cm = confusion_matrix(y_test, y_pred, labels=range(NUM_CLASSES))
    specificity = []
    for i in range(NUM_CLASSES):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp + 1e-8))

    # ROC/AUC per class
    auc_scores = []
    for i in range(NUM_CLASSES):
        try:
            fpr, tpr, _ = roc_curve(y_test_oh[:, i], y_pred_proba[:, i])
            auc_scores.append(auc(fpr, tpr))
        except Exception:
            auc_scores.append(0.0)

    metrics = {
        "accuracy": float(acc),
        "macro_auc": float(np.mean(auc_scores)),
        "per_class": {}
    }
    for i, cn in enumerate(CLASSES):
        metrics["per_class"][cn] = {
            "precision": float(prec[i]),
            "sensitivity": float(rec[i]),  # recall = sensitivity
            "specificity": float(specificity[i]),
            "f1_score": float(f1[i]),
            "auc": float(auc_scores[i]),
            "support": int(supp[i])
        }
    return metrics, cm

# ============================================================================
# GRAD-CAM EXPLAINABILITY (Section 5)
# ============================================================================

def grad_cam_1d(model, X_sample, class_idx, last_conv_name='conv3'):
    """Compute Grad-CAM heatmap for 1D CNN."""
    try:
        last_conv = model.get_layer(last_conv_name)
        grad_model = Model(model.inputs, [last_conv.output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(X_sample[np.newaxis])
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        weights = tf.reduce_mean(grads, axis=1)  # (1, filters)
        cam = tf.reduce_sum(conv_out[0] * weights[0], axis=-1)  # (time,)
        cam = tf.nn.relu(cam).numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    except Exception as e:
        print(f"  [Grad-CAM error: {e}]")
        return None

def save_gradcam_visualization(model, X_test, y_test, save_dir=RESULTS_DIR):
    """Generate and save Grad-CAM heatmaps for each class."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('Grad-CAM: EEG Temporal Attention by Seizure State', fontsize=14)
        for i, (cn, ax) in enumerate(zip(CLASSES, axes.flat)):
            idxs = np.where(y_test == i)[0]
            if len(idxs) == 0:
                continue
            sample = X_test[idxs[0]]
            cam = grad_cam_1d(model, sample, i)
            if cam is not None:
                ax.plot(sample[:, 0], alpha=0.5, label='EEG Ch-1', color='steelblue')
                ax_t = ax.twinx()
                cam_interp = np.interp(np.linspace(0, len(cam)-1, SIGNAL_LENGTH),
                                       np.arange(len(cam)), cam)
                ax_t.fill_between(range(SIGNAL_LENGTH), cam_interp, alpha=0.3, color='red')
                ax_t.set_ylabel('Attention', color='red')
            ax.set_title(f'{cn}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gradcam_attention.png', dpi=150)
        plt.close()
        print(f"  Grad-CAM saved to {save_dir}/gradcam_attention.png")
    except Exception as e:
        print(f"  [Grad-CAM visualization error: {e}]")

# ============================================================================
# REAL-TIME FEASIBILITY (Section 7)
# ============================================================================

def benchmark_inference(model, n_samples=100):
    """Measure inference latency and memory usage."""
    dummy = np.random.randn(1, SIGNAL_LENGTH, NUM_CHANNELS).astype(np.float32)
    # Warm up
    for _ in range(5):
        model.predict(dummy, verbose=0)

    # Latency (single sample)
    times = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append(time.perf_counter() - t0)

    # Memory
    tracemalloc.start()
    model.predict(dummy, verbose=0)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Batch throughput
    batch = np.random.randn(64, SIGNAL_LENGTH, NUM_CHANNELS).astype(np.float32)
    t0 = time.perf_counter()
    model.predict(batch, verbose=0)
    batch_time = time.perf_counter() - t0

    result = {
        "mean_latency_ms": float(np.mean(times) * 1000),
        "std_latency_ms": float(np.std(times) * 1000),
        "min_latency_ms": float(np.min(times) * 1000),
        "max_latency_ms": float(np.max(times) * 1000),
        "peak_memory_MB": float(peak_mem / 1024 / 1024),
        "throughput_samples_per_sec": float(64 / batch_time),
        "device": "GPU" if tf.config.list_physical_devices('GPU') else "CPU",
    }
    return result

# ============================================================================
# STATISTICAL VALIDATION (Section 6)
# ============================================================================

def run_statistical_validation(patient_data, n_runs=5):
    """Run repeated experiments for statistical significance."""
    print("\n" + "=" * 60)
    print("  STATISTICAL VALIDATION (5-run repeated experiments)")
    print("=" * 60)

    baseline_accs = []
    qinn_accs = []

    for run in range(n_runs):
        seed = GLOBAL_SEED + run
        set_seeds(seed)
        print(f"\n  Run {run + 1}/{n_runs} (seed={seed})")

        # Combine all patients for quick validation run
        all_X = np.concatenate([v[0] for v in patient_data.values()])
        all_y = np.concatenate([v[1] for v in patient_data.values()])
        X_aug, y_aug = augment(all_X, all_y, factor=10)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=seed)

        # Baseline CNN
        m_base = compile_model(build_baseline_cnn())
        train_model(m_base, X_tr, y_tr, X_te, y_te, epochs=50, verbose=0)
        p_base = np.argmax(m_base.predict(X_te, verbose=0), 1)
        acc_base = accuracy_score(y_te, p_base)
        baseline_accs.append(acc_base)

        # Full QINN
        m_qinn = compile_model(build_full_qinn())
        train_model(m_qinn, X_tr, y_tr, X_te, y_te, epochs=50, verbose=0)
        p_qinn = np.argmax(m_qinn.predict(X_te, verbose=0), 1)
        acc_qinn = accuracy_score(y_te, p_qinn)
        qinn_accs.append(acc_qinn)

        print(f"    Baseline: {acc_base*100:.2f}%  |  QINN: {acc_qinn*100:.2f}%")
        del m_base, m_qinn
        keras.backend.clear_session()

    # Statistics
    base_arr = np.array(baseline_accs)
    qinn_arr = np.array(qinn_accs)
    t_stat, p_value = stats.ttest_rel(qinn_arr, base_arr)
    diff = qinn_arr - base_arr
    ci_95 = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))

    results = {
        "baseline": {"mean": float(base_arr.mean()), "std": float(base_arr.std()),
                     "scores": [float(x) for x in baseline_accs]},
        "qinn": {"mean": float(qinn_arr.mean()), "std": float(qinn_arr.std()),
                 "scores": [float(x) for x in qinn_accs]},
        "paired_ttest": {"t_statistic": float(t_stat), "p_value": float(p_value)},
        "95_ci_improvement": [float(ci_95[0]), float(ci_95[1])],
    }
    with open(f"{RESULTS_DIR}/statistical_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Baseline: {base_arr.mean()*100:.2f} ± {base_arr.std()*100:.2f}%")
    print(f"  QINN:     {qinn_arr.mean()*100:.2f} ± {qinn_arr.std()*100:.2f}%")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"  95% CI of improvement: [{ci_95[0]*100:.2f}%, {ci_95[1]*100:.2f}%]")
    return results

# ============================================================================
# LOPO-CV (Section 2)
# ============================================================================

def run_lopo_cv(patient_data, builder_fn=build_full_qinn, aug_factor=15):
    """Leave-One-Patient-Out Cross Validation."""
    print("\n" + "=" * 60)
    print("  LEAVE-ONE-PATIENT-OUT CROSS VALIDATION")
    print("=" * 60)

    patients = sorted(patient_data.keys())
    fold_results = []

    for fold_idx, test_pid in enumerate(patients):
        print(f"\n  Fold {fold_idx+1}/{len(patients)}: test patient = {test_pid}")
        X_te, y_te = patient_data[test_pid]

        # Train on all other patients
        train_Xs = [patient_data[p][0] for p in patients if p != test_pid]
        train_ys = [patient_data[p][1] for p in patients if p != test_pid]
        if not train_Xs:
            continue
        X_tr = np.concatenate(train_Xs)
        y_tr = np.concatenate(train_ys)

        # Augment training data only
        X_tr_aug, y_tr_aug = augment(X_tr, y_tr, factor=aug_factor)

        # Build and train
        set_seeds(GLOBAL_SEED + fold_idx)
        model = compile_model(builder_fn())
        # Use 10% of training for validation during training
        X_trn, X_val, y_trn, y_val = train_test_split(
            X_tr_aug, y_tr_aug, test_size=0.1, stratify=y_tr_aug,
            random_state=GLOBAL_SEED)
        train_model(model, X_trn, y_trn, X_val, y_val, epochs=80, verbose=0)

        # Evaluate on held-out patient
        metrics, cm = compute_medical_metrics(model, X_te, y_te)
        fold_results.append({
            "patient": test_pid,
            "accuracy": metrics["accuracy"],
            "macro_auc": metrics["macro_auc"],
            "per_class": metrics["per_class"],
            "n_test_samples": len(y_te)
        })
        print(f"    Acc: {metrics['accuracy']*100:.2f}%  AUC: {metrics['macro_auc']:.4f}  "
              f"(n={len(y_te)})")

        del model
        keras.backend.clear_session()

    # Aggregate
    accs = [r["accuracy"] for r in fold_results]
    aucs = [r["macro_auc"] for r in fold_results]
    agg = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "per_fold": fold_results
    }
    with open(f"{RESULTS_DIR}/lopo_cv_results.json", "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n  LOPO-CV Aggregate:")
    print(f"    Accuracy: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%")
    print(f"    AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return agg

# ============================================================================
# ABLATION STUDY (Section 3)
# ============================================================================

def run_ablation_study(patient_data):
    """Run ablation study with 4 model variants."""
    print("\n" + "=" * 60)
    print("  ABLATION STUDY")
    print("=" * 60)

    all_X = np.concatenate([v[0] for v in patient_data.values()])
    all_y = np.concatenate([v[1] for v in patient_data.values()])
    X_aug, y_aug = augment(all_X, all_y, factor=15)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=GLOBAL_SEED)

    models = [
        ("A: Baseline CNN", build_baseline_cnn),
        ("B: CNN+Superposition", build_cnn_superposition),
        ("C: CNN+Super+Interf", build_cnn_interference),
        ("D: Full QINN", build_full_qinn),
    ]

    results = []
    for name, builder in models:
        print(f"\n  Training {name}...")
        set_seeds(GLOBAL_SEED)
        model = compile_model(builder())
        train_model(model, X_tr, y_tr, X_te, y_te, epochs=80, verbose=0)
        metrics, cm = compute_medical_metrics(model, X_te, y_te)
        bench = benchmark_inference(model)
        result = {
            "model": name,
            "params": model.count_params(),
            "accuracy": metrics["accuracy"],
            "macro_auc": metrics["macro_auc"],
            "f1_weighted": float(np.mean([v["f1_score"] for v in metrics["per_class"].values()])),
            "latency_ms": bench["mean_latency_ms"],
        }
        results.append(result)
        print(f"    Acc: {metrics['accuracy']*100:.2f}%  AUC: {metrics['macro_auc']:.4f}  "
              f"Params: {model.count_params():,}  Latency: {bench['mean_latency_ms']:.1f}ms")
        del model
        keras.backend.clear_session()

    with open(f"{RESULTS_DIR}/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("  QINN v3 — Quantum-Inspired Neural Network")
    print("  Epileptic Seizure Prediction (CHB-MIT)")
    print("=" * 60)

    # 1. Environment
    print("\n[1/8] Environment")
    env = log_environment()

    # 2. Load data
    print("\n[2/8] Loading Data (per-patient)")
    patient_data = load_data_by_patient()
    if not patient_data:
        print("  No data found!")
        return

    # 3. Ablation Study
    print("\n[3/8] Ablation Study")
    ablation = run_ablation_study(patient_data)

    # 4. LOPO-CV (main evaluation)
    print("\n[4/8] LOPO Cross-Validation")
    lopo = run_lopo_cv(patient_data, builder_fn=build_full_qinn)

    # 5. Statistical Validation
    print("\n[5/8] Statistical Validation")
    stat_val = run_statistical_validation(patient_data, n_runs=5)

    # 6. Final model for explainability + benchmarking
    print("\n[6/8] Training Final Model")
    all_X = np.concatenate([v[0] for v in patient_data.values()])
    all_y = np.concatenate([v[1] for v in patient_data.values()])
    X_aug, y_aug = augment(all_X, all_y, factor=15)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=GLOBAL_SEED)

    set_seeds(GLOBAL_SEED)
    final_model = compile_model(build_full_qinn(use_pennylane=HAS_PENNYLANE))
    train_model(final_model, X_tr, y_tr, X_te, y_te, epochs=100, verbose=1)

    # 7. Medical metrics + Grad-CAM
    print("\n[7/8] Medical Metrics & Explainability")
    metrics, cm = compute_medical_metrics(final_model, X_te, y_te)
    with open(f"{RESULTS_DIR}/medical_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
    for cn in CLASSES:
        m = metrics['per_class'][cn]
        print(f"    {cn}: Sens={m['sensitivity']:.3f} Spec={m['specificity']:.3f} "
              f"F1={m['f1_score']:.3f} AUC={m['auc']:.3f}")

    save_gradcam_visualization(final_model, X_te, y_te)

    # 8. Benchmarking
    print("\n[8/8] Real-Time Feasibility")
    bench = benchmark_inference(final_model)
    with open(f"{RESULTS_DIR}/benchmark.json", "w") as f:
        json.dump(bench, f, indent=2)
    print(f"  Latency: {bench['mean_latency_ms']:.2f} ± {bench['std_latency_ms']:.2f} ms")
    print(f"  Peak Memory: {bench['peak_memory_MB']:.2f} MB")
    print(f"  Throughput: {bench['throughput_samples_per_sec']:.1f} samples/sec")

    # Save final model
    final_model.save(f'{SAVE_DIR}/qinn_v3_final.keras')
    json.dump({str(i): n for i, n in enumerate(CLASSES)},
              open(f'{SAVE_DIR}/label_mapping.json', 'w'))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"  Model saved to:   {SAVE_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
