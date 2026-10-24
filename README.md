# EEDA, Easy EDA

A small Flask app for uploading CSV files, viewing and editing columns, and creating histogram, violin, count, box, scatter, and correlation plots.

## Run locally

Requires Python 3. Install the dependencies and start the app:

```bash
python -m venv .venv
```

Activate the virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser and upload a CSV file. Uploads are limited to 16 MB.
