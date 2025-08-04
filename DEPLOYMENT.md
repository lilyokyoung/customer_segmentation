# Deployment Guide for Customer Segmentation AI Agent

## 🚀 Quick Deployment Options

### Option 1: Use Basic Requirements (Recommended for strict platforms)
```bash
pip install -r requirements-basic.txt
python app.py
```

### Option 2: Use Minimal Requirements (Ultra-conservative)
```bash
pip install -r requirements-minimal.txt
python src/main.py
```

### Option 3: Use Flexible Requirements (No upper bounds)
```bash
pip install -r requirements-flexible.txt
python src/main.py
```

### Option 4: Standard Requirements
```bash
pip install -r requirements.txt
python src/main.py
```

## 🔧 Platform-Specific Instructions

### Heroku
- Uses `runtime.txt` (Python 3.10)
- Uses `requirements.txt`
- Uses `Procfile`

### Railway/Render
- Uses `requirements.txt`
- Entry point: `python app.py`

### Vercel/Netlify
- Uses `pyproject.toml`
- Entry point: `app.py`

### Docker
- Uses `.dockerignore`
- Base image: `python:3.10-slim`

## 🛠️ Troubleshooting Installer Errors

If you get "installer returned a non-zero exit code":

1. **Try minimal requirements first:**
   ```bash
   pip install -r requirements-basic.txt
   ```

2. **Use app.py instead of src/main.py:**
   ```bash
   python app.py
   ```

3. **Check Python version compatibility:**
   - Platform supports Python 3.8-3.11
   - Uses `runtime.txt` with python-3.10

4. **Environment variables (if needed):**
   ```bash
   export PYTHONPATH=src
   export PYTHONUNBUFFERED=1
   ```

## 📦 Package Installation Order (if individual installation needed)

```bash
pip install numpy>=1.21.0
pip install pandas>=1.5.0
pip install scikit-learn>=1.2.0
pip install matplotlib>=3.5.0
pip install pyyaml>=6.0.0
```

## ✅ Verification

Test your deployment:
```bash
python -c "import numpy, pandas, sklearn, matplotlib, yaml; print('All dependencies OK')"
python app.py
```

The application should run and show:
- Warning messages for missing optional components (normal)
- Sample data creation
- Workflow completion with 1000 customers analyzed
- 3 exported result files

## 🎯 Key Features

- **Graceful degradation**: Works with missing dependencies
- **Mock components**: Provides functionality even with minimal packages
- **Multiple entry points**: `app.py`, `src/main.py`
- **Flexible requirements**: Multiple requirement files for different scenarios
- **Platform compatibility**: Supports major deployment platforms

## 📝 Notes

- The application uses mock components when full dependencies aren't available
- This is intentional to ensure deployment success
- Full functionality requires all packages from requirements.txt
- Basic functionality works with just requirements-basic.txt
