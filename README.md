# signature-verification
A Flask-based signature verification app that uses a trained deep learning model to classify uploaded signatures as genuine or forged. It includes training data handling, model inference, and a web interface for users to upload signature images and view verification results.
# Signature Verification (Flask + Deep Learning)

A Flask web app that verifies whether an uploaded signature is **verified** or **not verified** using a trained Keras model.

## Project Structure
- `app.py` - Flask server and inference logic
- `train.py` - Training script
- `model/signature_model.h5` - Trained model
- `templates/` - UI templates
- `static/uploads/` - Uploaded images (ignored by git)

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, install at minimum:

```bash
pip install flask tensorflow werkzeug pillow numpy
```

## Run the App
```bash
python app.py
```
Then open:

- http://127.0.0.1:5000

## Note about `reference.png`
The app expects the reference signature at:

- `static/uploads/reference.png`

If it’s missing, the app will show an error.

## Contributing
Feel free to fork and improve the model/training pipeline.



