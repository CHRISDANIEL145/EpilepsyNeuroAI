"""
================================================================================
NEUROGUARD AI - CLINICAL EPILEPSY PREDICTION SYSTEM
================================================================================
Production-Ready Flask Backend with Bulletproof EEG File Handling
Supports: CSV, XLS, XLSX, mislabeled files, corrupted data
================================================================================
"""

import os
import io
import json
import uuid
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import scipy.signal as scipy_signal
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'epilepsy_prediction_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'txt'}
MODEL_PATH = '../model_file/epilepsy_quantum_model.h5'
SIGNAL_LENGTH = 1000
NUM_CHANNELS = 23
CLASSES = ["NORMAL", "ICTAL", "PREICTAL", "POSTICTAL"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = None
prediction_history = []


# ============================================================================
# QUANTUM LAYERS (Required for model loading)
# ============================================================================

class QuantumSuperposition(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.W_real = self.add_weight(name="W_real", shape=(input_shape[-1], self.units), initializer='glorot_uniform')
        self.W_imag = self.add_weight(name="W_imag", shape=(input_shape[-1], self.units), initializer='glorot_uniform')
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer='zeros')
    def call(self, x):
        real = tf.matmul(x, self.W_real)
        imag = tf.matmul(x, self.W_imag)
        magnitude = tf.sqrt(tf.square(real) + tf.square(imag) + 1e-8)
        return tf.nn.relu(magnitude + self.bias)
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

class QuantumInterference(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.phase = self.add_weight(name="phase", shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name="gamma", shape=(input_shape[-1],), initializer='ones', trainable=True)
    def call(self, x):
        return x * tf.cos(self.phase) + self.gamma * tf.sin(self.phase)

class QuantumMeasurement(tf.keras.layers.Layer):
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
    def build(self, input_shape):
        self.basis = self.add_weight(name="basis", shape=(input_shape[-1], self.n_classes), initializer='glorot_uniform')
    def call(self, x):
        projection = tf.matmul(x, self.basis)
        return tf.nn.softmax(projection)
    def get_config(self):
        config = super().get_config()
        config.update({"n_classes": self.n_classes})
        return config

class QuantumEntanglement(tf.keras.layers.Layer):
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

# ============================================================================
# MODEL LOADING
# ============================================================================

def _patch_h5_remove_time_major(filepath):
    """Patch H5 model file to remove deprecated 'time_major' LSTM argument."""
    try:
        import h5py, json, re
        with h5py.File(filepath, 'r+') as f:
            raw = f.attrs.get('model_config')
            if raw is None:
                return False
            if isinstance(raw, bytes):
                config_str = raw.decode('utf-8')
            else:
                config_str = str(raw)
            # Remove time_major occurrences (both true and false)
            patched = re.sub(r',?\s*"time_major":\s*(true|false)', '', config_str)
            f.attrs['model_config'] = patched.encode('utf-8')
        logger.info("✅ H5 model patched: removed 'time_major' from LSTM configs")
        return True
    except Exception as e:
        logger.error(f"❌ Patch error: {e}")
        return False


def load_model():
    """Load the trained Keras model with custom Quantum layers.
    Automatically patches the H5 file if it contains the deprecated
    'time_major' LSTM argument (saved with older TensorFlow versions)."""
    global model
    custom_objects = {
        'QuantumSuperposition': QuantumSuperposition,
        'QuantumInterference': QuantumInterference,
        'QuantumMeasurement': QuantumMeasurement,
        'QuantumEntanglement': QuantumEntanglement
    }
    # First attempt — load as-is
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        logger.info(f"✅ Model loaded: {MODEL_PATH}")
        return True
    except Exception as e:
        err = str(e)
        logger.warning(f"⚠️ First load attempt failed: {err}")

    # If the error is the known 'time_major' incompatibility, patch and retry
    if 'time_major' in err:
        logger.info("🔧 Detected 'time_major' incompatibility — patching H5 file...")
        if _patch_h5_remove_time_major(MODEL_PATH):
            try:
                model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
                logger.info(f"✅ Model loaded after patch: {MODEL_PATH}")
                return True
            except Exception as e2:
                logger.error(f"❌ Model load error after patch: {e2}")
        else:
            logger.error("❌ Patching failed")
    else:
        logger.error(f"❌ Model load error: {err}")
    return False

# ============================================================================
# SMART FILE TYPE DETECTION
# ============================================================================

def detect_file_type(filepath: str) -> str:
    """
    Detect actual file type by reading magic bytes.
    Returns: 'xls', 'xlsx', 'csv', 'text', or 'unknown'
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)
        
        # XLS (BIFF) magic bytes: D0 CF 11 E0 A1 B1 1A E1
        if header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
            return 'xls'
        
        # XLSX (ZIP) magic bytes: 50 4B 03 04
        if header[:4] == b'\x50\x4b\x03\x04':
            return 'xlsx'
        
        # Check if it's text-based (CSV or mislabeled)
        with open(filepath, 'rb') as f:
            sample = f.read(4096)
        
        # Try to decode as text
        try:
            text = sample.decode('utf-8')
            if ',' in text or '\t' in text or ';' in text:
                return 'csv'
            return 'text'
        except:
            try:
                text = sample.decode('latin-1')
                if ',' in text or '\t' in text or ';' in text:
                    return 'csv'
                return 'text'
            except:
                pass
        
        return 'unknown'
    except Exception as e:
        logger.error(f"File type detection error: {e}")
        return 'unknown'


# ============================================================================
# BULLETPROOF EEG FILE LOADER
# ============================================================================

def load_eeg_file(filepath: str) -> tuple:
    """
    Production-ready EEG file loader with multiple fallback strategies.
    
    Handles:
    - Real XLS (Excel 97-2003 binary)
    - Real XLSX (Excel 2007+ XML)
    - CSV files
    - Mislabeled files (.xls that are actually CSV)
    - Files with BOM encoding
    - Files with metadata headers
    - Tab/semicolon delimited files
    
    Returns:
        tuple: (numpy_array, error_message)
    """
    logger.info(f"📂 Loading: {filepath}")
    
    if not os.path.exists(filepath):
        return None, "File not found"
    
    # Detect actual file type (not just extension)
    actual_type = detect_file_type(filepath)
    file_ext = Path(filepath).suffix.lower()
    
    logger.info(f"📊 Extension: {file_ext}, Detected type: {actual_type}")
    
    df = None
    load_method = None
    
    # ========================================
    # STRATEGY 1: Real Excel Binary (XLS)
    # ========================================
    if actual_type == 'xls':
        try:
            df = pd.read_excel(filepath, header=None, engine='xlrd')
            load_method = "xlrd (real XLS)"
            logger.info(f"✅ Loaded with {load_method}")
        except Exception as e:
            logger.warning(f"xlrd failed: {e}")
    
    # ========================================
    # STRATEGY 2: Real Excel XML (XLSX)
    # ========================================
    if df is None and actual_type == 'xlsx':
        try:
            df = pd.read_excel(filepath, header=None, engine='openpyxl')
            load_method = "openpyxl (real XLSX)"
            logger.info(f"✅ Loaded with {load_method}")
        except Exception as e:
            logger.warning(f"openpyxl failed: {e}")
    
    # ========================================
    # STRATEGY 3: CSV (comma-separated)
    # ========================================
    if df is None and actual_type in ['csv', 'text']:
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
            for sep in [',', '\t', ';', ' ']:
                try:
                    df = pd.read_csv(filepath, header=None, encoding=encoding, sep=sep)
                    if df.shape[1] > 1:  # Valid if more than 1 column
                        load_method = f"CSV ({encoding}, sep='{sep}')"
                        logger.info(f"✅ Loaded with {load_method}")
                        break
                except:
                    continue
            if df is not None and df.shape[1] > 1:
                break
    
    # ========================================
    # STRATEGY 4: Mislabeled XLS (actually text)
    # ========================================
    if df is None and file_ext == '.xls' and actual_type in ['csv', 'text', 'unknown']:
        logger.info("🔄 XLS appears to be mislabeled text file, trying CSV parsers...")
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            for sep in [',', '\t', ';', ' ']:
                try:
                    df = pd.read_csv(filepath, header=None, encoding=encoding, sep=sep)
                    if df.shape[1] > 1:
                        load_method = f"Mislabeled XLS as CSV ({encoding}, sep='{sep}')"
                        logger.info(f"✅ Loaded with {load_method}")
                        break
                except:
                    continue
            if df is not None and df.shape[1] > 1:
                break

    # ========================================
    # STRATEGY 5: Force Excel engines (last resort)
    # ========================================
    if df is None:
        engines = ['xlrd', 'openpyxl', None]
        for engine in engines:
            try:
                if engine:
                    df = pd.read_excel(filepath, header=None, engine=engine)
                else:
                    df = pd.read_excel(filepath, header=None)
                load_method = f"Excel (engine={engine})"
                logger.info(f"✅ Loaded with {load_method}")
                break
            except Exception as e:
                logger.warning(f"Engine {engine} failed: {e}")
                continue
    
    # ========================================
    # STRATEGY 6: Raw binary read and parse
    # ========================================
    if df is None:
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            
            # Try different decodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text = content.decode(encoding)
                    lines = text.strip().split('\n')
                    
                    # Parse as numeric matrix
                    data_rows = []
                    for line in lines:
                        # Try different separators
                        for sep in [',', '\t', ';', ' ', '  ']:
                            parts = [p.strip() for p in line.split(sep) if p.strip()]
                            if len(parts) > 1:
                                try:
                                    row = [float(p) for p in parts]
                                    data_rows.append(row)
                                    break
                                except:
                                    continue
                    
                    if len(data_rows) > 10:
                        # Normalize row lengths
                        max_cols = max(len(r) for r in data_rows)
                        normalized = [r + [0]*(max_cols-len(r)) for r in data_rows]
                        df = pd.DataFrame(normalized)
                        load_method = f"Raw binary parse ({encoding})"
                        logger.info(f"✅ Loaded with {load_method}")
                        break
                except:
                    continue
        except Exception as e:
            logger.error(f"Raw parse failed: {e}")
    
    # ========================================
    # CHECK IF LOADING SUCCEEDED
    # ========================================
    if df is None or df.empty:
        return None, "Could not read file. Tried: xlrd, openpyxl, CSV, raw parse. File may be corrupted."
    
    logger.info(f"📊 Raw shape: {df.shape}, Method: {load_method}")
    
    # ========================================
    # CLEAN AND CONVERT DATA
    # ========================================
    return clean_dataframe(df)


def clean_dataframe(df: pd.DataFrame) -> tuple:
    """
    Clean and convert DataFrame to numeric numpy array.
    Handles headers, NaN, mixed types, irregular rows.
    """
    try:
        # Remove completely empty rows/columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        if df.empty:
            return None, "File contains no data"
        
        logger.info(f"📊 After empty removal: {df.shape}")
        
        # Find first row that is mostly numeric
        first_data_row = 0
        for idx in range(min(20, len(df))):
            row = df.iloc[idx]
            numeric_count = 0
            total_valid = 0
            
            for val in row:
                if pd.notna(val):
                    total_valid += 1
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
            
            # If >60% numeric, this is data
            if total_valid > 0 and numeric_count / total_valid > 0.6:
                first_data_row = idx
                break
        
        if first_data_row > 0:
            logger.info(f"📊 Skipping {first_data_row} header rows")
            df = df.iloc[first_data_row:].reset_index(drop=True)
        
        # Convert all to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate NaN percentage
        nan_pct = df.isna().sum().sum() / df.size * 100
        logger.info(f"📊 NaN percentage: {nan_pct:.1f}%")
        
        if nan_pct > 80:
            return None, f"File has {nan_pct:.0f}% non-numeric data. Not valid EEG."
        
        # Fill NaN: first with column mean, then with 0
        df = df.fillna(df.mean())
        df = df.fillna(0)
        
        # Remove all-zero rows
        df = df.loc[~(df == 0).all(axis=1)]
        
        if df.empty or len(df) < 10:
            return None, "Insufficient numeric data in file"
        
        data = df.values.astype(np.float32)
        logger.info(f"✅ Clean data: {data.shape}, range: [{data.min():.2f}, {data.max():.2f}]")
        
        return data, None
        
    except Exception as e:
        logger.error(f"Clean error: {e}")
        return None, f"Data cleaning failed: {str(e)}"


# ============================================================================
# ROBUST SIGNAL PREPROCESSING
# ============================================================================

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=256, order=4):
    """Apply 0.5-40 Hz Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(low, 0.001)
    high = min(high, 0.999)
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, data, axis=0).astype(np.float32)

def notch_filter(data, freq=50.0, fs=256, Q=30):
    """Apply 50 Hz notch filter to remove power-line interference."""
    nyq = 0.5 * fs
    if freq >= nyq:
        return data
    b, a = scipy_signal.iirnotch(freq / nyq, Q)
    return scipy_signal.filtfilt(b, a, data, axis=0).astype(np.float32)

def preprocess_signal(data: np.ndarray) -> tuple:
    """
    Preprocess EEG to model input shape: (1000, 23)
    
    Handles:
    - 1D arrays
    - Wrong channel count (pad/truncate)
    - Filters (Bandpass & Notch)
    - Wrong length (center segment/pad)
    - NaN/Inf values
    - Per-channel normalization
    """
    logger.info(f"🔧 Preprocessing: {data.shape}")
    
    if data is None or data.size == 0:
        return None, "Empty data"
    
    try:
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if data.ndim != 2:
            return None, f"Invalid dimensions: {data.ndim}D"
        
        n_samples, n_channels = data.shape
        logger.info(f"📊 Input: {n_samples} samples × {n_channels} channels")
        
        # Fix channels
        if n_channels > NUM_CHANNELS:
            data = data[:, :NUM_CHANNELS]
            logger.info(f"📊 Truncated channels → {NUM_CHANNELS}")
        elif n_channels < NUM_CHANNELS:
            pad = np.zeros((n_samples, NUM_CHANNELS - n_channels), dtype=np.float32)
            data = np.hstack([data, pad])
            logger.info(f"📊 Padded channels → {NUM_CHANNELS}")
            
        # Apply Filters
        try:
            data = bandpass_filter(data)
            data = notch_filter(data)
            logger.info("📊 Applied Bandpass and Notch filters")
        except Exception as e:
            logger.warning(f"⚠️ Filtering failed, continuing with raw data: {e}")
        
        # Fix length (Center Segment)
        n_samples = data.shape[0]
        if n_samples > SIGNAL_LENGTH:
            start = (n_samples - SIGNAL_LENGTH) // 2
            data = data[start:start + SIGNAL_LENGTH]
            logger.info(f"📊 Center segment → {SIGNAL_LENGTH}")
        elif n_samples < SIGNAL_LENGTH:
            pad_total = SIGNAL_LENGTH - n_samples
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            data = np.pad(data, ((pad_before, pad_after), (0, 0)), mode='reflect')
            logger.info(f"📊 Reflect-padded → {SIGNAL_LENGTH}")
        
        # Normalize per channel
        mu = data.mean(axis=0, keepdims=True)
        sigma = data.std(axis=0, keepdims=True) + 1e-8
        data = (data - mu) / sigma
        
        # Clean NaN/Inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate shape
        if data.shape != (SIGNAL_LENGTH, NUM_CHANNELS):
            return None, f"Shape error: {data.shape}"
        
        logger.info(f"✅ Final: {data.shape}, range: [{data.min():.2f}, {data.max():.2f}]")
        return data.astype(np.float32), None
        
    except Exception as e:
        logger.error(f"Preprocess error: {e}")
        return None, f"Preprocessing failed: {str(e)}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_class_color(class_name: str) -> dict:
    colors = {
        "NORMAL": {"primary": "#3b82f6", "bg": "rgba(59, 130, 246, 0.2)"},
        "PREICTAL": {"primary": "#eab308", "bg": "rgba(234, 179, 8, 0.2)"},
        "ICTAL": {"primary": "#ef4444", "bg": "rgba(239, 68, 68, 0.2)"},
        "POSTICTAL": {"primary": "#22c55e", "bg": "rgba(34, 197, 94, 0.2)"}
    }
    return colors.get(class_name, colors["NORMAL"])

def get_class_description(class_name: str) -> str:
    descriptions = {
        "NORMAL": "Brain activity is within normal parameters. No seizure activity detected.",
        "PREICTAL": "Warning: Pre-seizure activity detected. Seizure may occur soon.",
        "ICTAL": "ALERT: Active seizure detected. Immediate medical attention recommended.",
        "POSTICTAL": "Post-seizure recovery phase. Patient may experience confusion."
    }
    return descriptions.get(class_name, "Unknown state")


# ============================================================================
# FLASK ROUTES - PAGES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/visualization')
def visualization_page():
    return render_template('visualization.html')


# ============================================================================
# FLASK API ROUTES
# ============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle EEG file upload"""
    logger.info("📤 Upload request")
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Use CSV, XLS, or XLSX.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)
        
        session['uploaded_file'] = filepath
        session['original_filename'] = filename
        
        logger.info(f"✅ Saved: {filepath}")
        return jsonify({'success': True, 'filename': filename, 'file_id': unique_name})
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run seizure prediction on uploaded EEG file"""
    global model, prediction_history
    
    logger.info("🔮 Prediction request")
    
    # Load model if needed
    if model is None and not load_model():
        return jsonify({'success': False, 'error': 'Model not available'}), 500
    
    # Get file path
    filepath = session.get('uploaded_file')
    original_filename = session.get('original_filename', 'Unknown')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'No file uploaded. Please upload first.'}), 400
    
    # Load EEG data
    data, load_error = load_eeg_file(filepath)
    if load_error:
        try: os.remove(filepath)
        except: pass
        return jsonify({'success': False, 'error': load_error}), 400
    
    original_shape = data.shape
    
    # Preprocess
    processed, prep_error = preprocess_signal(data)
    if prep_error:
        try: os.remove(filepath)
        except: pass
        return jsonify({'success': False, 'error': prep_error}), 400
    
    # Predict
    try:
        input_data = np.expand_dims(processed, axis=0)
        predictions = model.predict(input_data, verbose=0)[0]
        
        pred_idx = int(np.argmax(predictions))
        pred_class = CLASSES[pred_idx]
        confidence = float(predictions[pred_idx]) * 100
        
        all_probs = {CLASSES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASSES))}
        
        logger.info(f"✅ Prediction: {pred_class} ({confidence:.1f}%)")
        
        result = {
            'success': True,
            'prediction': {
                'class': pred_class,
                'confidence': round(confidence, 2),
                'color': get_class_color(pred_class),
                'description': get_class_description(pred_class),
                'all_probabilities': all_probs
            },
            'file_info': {
                'filename': original_filename,
                'samples': original_shape[0],
                'channels': original_shape[1] if len(original_shape) > 1 else 1
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        prediction_history.insert(0, result)
        prediction_history = prediction_history[:50]
        
        try: os.remove(filepath)
        except: pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        try: os.remove(filepath)
        except: pass
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({'success': True, 'history': prediction_history})

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    global prediction_history
    prediction_history = []
    return jsonify({'success': True, 'message': 'History cleared'})

@app.route('/api/system-status', methods=['GET'])
def system_status():
    return jsonify({
        'success': True,
        'status': {
            'model_loaded': model is not None,
            'classes': CLASSES,
            'signal_length': SIGNAL_LENGTH,
            'num_channels': NUM_CHANNELS,
            'predictions_made': len(prediction_history)
        }
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Max 50MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🧠 NEUROGUARD AI - CLINICAL EPILEPSY PREDICTION")
    print("=" * 60)
    load_model()
    print(f"📊 Config: {SIGNAL_LENGTH} samples × {NUM_CHANNELS} channels")
    print(f"📊 Classes: {', '.join(CLASSES)}")
    print("🚀 Server: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
