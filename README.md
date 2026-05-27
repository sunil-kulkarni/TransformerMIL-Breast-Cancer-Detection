# TransformerMIL Breast Cancer Detection

Simple FastAPI app that runs a preloaded Swin + Phikon feature extractor with a 5-fold Transformer MIL ensemble to classify breast histopathology images.

**Files:**
- [app.py](app.py#L1-L200): FastAPI server and model inference pipeline.
- [capstone_ensemble](capstone_ensemble): folder containing `brain_0.pth`..`brain_4.pth` ensemble weights.
- [models](models): additional model weights used by the project.

**Requirements:**
Install the Python dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Note: GPU with CUDA and a matching `torch` build is recommended for reasonable inference speed.

**Model files:**
- The app expects the 5 ensemble weights in `capstone_ensemble/brain_0.pth` ... `brain_4.pth`.
- The script also downloads or loads locally the CTransPath and Phikon weights (configured to use local cache / offline files).

**Run (development):**

```bash
# run directly
python app.py

# or with uvicorn for production-like server
uvicorn app:app --host 0.0.0.0 --port 8000
```

**API:**
- `GET /` serves the UI from `static/index.html`.
- `POST /analyze` accepts a multipart `file` (image). Response JSON contains `prediction` and `all_scores` breakdown.

Example curl usage:

```bash
curl -F "file=@/path/to/image.png" http://127.0.0.1:8000/analyze
```

**Notes & troubleshooting:**
- If you want to run without GPU, the app will fall back to CPU but will be much slower.
- Ensure the model weight files are present and readable by the process.

If you want, I can pin versions in `requirements.txt` or create a `venv` setup script.
