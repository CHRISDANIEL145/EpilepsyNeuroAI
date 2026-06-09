# NeuroGuard AI - Clinical-Grade Epilepsy Prediction System

A modern, futuristic web interface for real-time seizure detection using deep learning.

![NeuroGuard AI](https://img.shields.io/badge/NeuroGuard-AI-0afff2?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-ff6f00?style=for-the-badge&logo=tensorflow)

## 🧠 Features

- **Real-Time Prediction**: Upload EEG files and get instant seizure predictions
- **4-Class Classification**: NORMAL, PREICTAL, ICTAL, POSTICTAL
- **Modern UI**: Futuristic medical-grade interface with neon aesthetics
- **Interactive Visualizations**: Chart.js powered probability distributions
- **EEG Visualization**: Simulated brain wave monitoring
- **Prediction History**: Track all previous analyses
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Trained model file (`epilepsy_quantum_model.h5`)

### Installation

1. **Navigate to the web app directory:**
   ```bash
   cd web_app
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model file exists:**
   The app expects the trained model at `../model_file/epilepsy_quantum_model.h5`

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Open your browser:**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
web_app/
├── app.py                 # Flask backend application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── uploads/              # Temporary upload directory
├── static/
│   ├── css/
│   │   └── custom.css    # Additional custom styles
│   └── js/
│       └── main.js       # Global JavaScript utilities
└── templates/
    ├── base.html         # Base template with navigation
    ├── index.html        # Landing page / Dashboard
    ├── upload.html       # File upload page
    ├── results.html      # Prediction results page
    ├── history.html      # Prediction history page
    └── visualization.html # EEG visualization page
```

## 🎨 UI Features

### Color Palette
- **Neon Aqua**: `#0afff2` - Primary accent
- **Neon Purple**: `#a855f7` - Secondary accent
- **Neon Indigo**: `#4f46e5` - Tertiary accent
- **Dark Zinc**: `#18181b` - Background
- **Dark Slate**: `#0f172a` - Deep background

### Class Colors
- 🔵 **NORMAL**: Blue (`#3b82f6`)
- 🟡 **PREICTAL**: Yellow (`#eab308`)
- 🔴 **ICTAL**: Red (`#ef4444`)
- 🟢 **POSTICTAL**: Green (`#22c55e`)

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/upload` | GET | File upload page |
| `/results` | GET | Results page |
| `/history` | GET | History page |
| `/visualization` | GET | EEG visualization |
| `/api/upload` | POST | Upload EEG file |
| `/api/predict` | POST | Run prediction |
| `/api/history` | GET | Get prediction history |
| `/api/clear-history` | POST | Clear history |
| `/api/system-status` | GET | System status |

## 📊 Supported File Formats

- CSV (`.csv`)
- Excel 97-2003 (`.xls`)
- Excel 2007+ (`.xlsx`)

### File Requirements
- Numerical EEG data
- Rows = time samples
- Columns = EEG channels (up to 23)
- No headers required (auto-detected)

## 🛠️ Technologies Used

### Frontend
- **TailwindCSS** - Utility-first CSS framework
- **Material Tailwind** - Material Design components
- **Chart.js** - Interactive charts
- **Anime.js** - Smooth animations

### Backend
- **Flask** - Python web framework
- **TensorFlow/Keras** - Deep learning model
- **Pandas** - Data processing
- **NumPy** - Numerical operations

## 🔧 Configuration

Edit `app.py` to modify:

```python
# Model path
MODEL_PATH = '../model_file/epilepsy_quantum_model.h5'

# Signal parameters
SIGNAL_LENGTH = 1000
NUM_CHANNELS = 23

# Server settings
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📱 Responsive Breakpoints

- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

## 🔒 Security Notes

- File uploads are sanitized using `secure_filename`
- Uploaded files are deleted after prediction
- Session-based file tracking
- No external API calls

## 🐛 Troubleshooting

### Model not loading
- Ensure `epilepsy_quantum_model.h5` exists in `../model_file/`
- Check TensorFlow version compatibility

### File upload fails
- Verify file format (CSV, XLS, XLSX)
- Check file contains numerical data
- Ensure file is not corrupted

### Charts not displaying
- Clear browser cache
- Check browser console for errors
- Ensure Chart.js CDN is accessible

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

**Built with ❤️ for Clinical Epilepsy Research**
