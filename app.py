import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import uuid
import io
import base64
from flask import Flask,render_template, jsonify,request, send_from_directory

#create chart images
#without openning a window
matplotlib.use("Agg")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16*1024*1024 #upload lim:16MB

rows_per_page = 50
datasets = {}

def error_response(message,status = 404):
    return jsonify({"error":message}),status

def find_dataset(dataset_id):
    if not isinstance(dataset_id,str):
        return None
    else:
        return datasets.get(dataset_id)

def make_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value,np.generic):
        value = value.item()
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,(pd.Timestamp,pd.Timedelta)):
        return value.isoformat() 
    return value


def make_dataset_response(dataset_id,frame,page=0):
    page_count = max(1,(len(frame) + rows_per_page-1) // rows_per_page)
    page = min(max(page,0),page_count -1)
    first_row = page * rows_per_page
    page_of_rows = frame.iloc[first_row : first_row + rows_per_page]

    rows = []
    for row_id,row in page_of_rows.iterrows():
        rows.append({
            "id" : int(row_id),
            "values": [make_json_value(value) for value in row.tolist()],
        })

    numeric_columns = frame.select_dtypes(include = "number").columns.tolist()
    return {
        "dataset_id" : dataset_id,
        "columns" : [str(column) for column in frame.columns],
        "numeric_columns": [str(column) for column in numeric_columns],
        "rows": rows,
        "row_count" : len(frame),
        "column_count":len(frame.columns),
        #this only gives total number of null values in a dataset
        #need improvement of this,we need more specific info
        #like which feature has how many null values
        "missing_count":int(frame.isna().sum().sum()),
        "page":page,
        "page_count":page_count,
        "page_size": rows_per_page,
    }

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/upload")
def upload_csv():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return error_response("Choose a CSV file to upload.")
    if not uploaded_file.filename.lower().endswith(".csv"):
        return error_response("The uploaded file must have a .csv extension.")

    try:
        frame = pd.read_csv(uploaded_file.stream)
    except (pd.errors.ParseError, pd.errors.EmptyDataError,UnicodeDecodeError,ValueError) as error:
        return error_response(f"could not read this file:{error}")

    dataset_id = uuid.uuid4().hex
    datasets[dataset_id] = frame
    return jsonify(make_dataset_response(dataset_id,frame))

@app.get("/api/data")
def get_data():
    dataset_id = request.args.get("dataset_id")
    frame = find_dataset(dataset_id)

    if frame is None:
        return error_response("File not found,try upload that again.",404)

    try:
        page = int(request.args.get("page","0"))
    except ValueError:
        return error_response("Page must be a whole number")
    
    return jsonify(make_dataset_response(dataset_id,frame,page))

@app.post("/api/columns")
def edit_columns():
    data = request.get_json()
    if not isinstance(data,dict):
        return error_response("it is not json,need json")

    dataset_id = data.get("dataset_id")
    frame = find_dataset(dataset_id)
    if frame is None:
            return error_response("File not found,try upload that again.",404)

    action = data.get("action")
    columns = data.get("columns")

    if action not in ("keep", "delete"):
        return error_response("Choose either keep or delete as the column action.")
    if not isinstance(columns, list) or not columns:
        return error_response("Select at least one column first.")
    if any(not isinstance(column, str) for column in columns):
        return error_response("Selected column names are invalid.")

    selected_columns = set(columns).intersection(frame.columns)

    if action == "keep":
        remaining_columns = []
        for column in frame.columns:
            if column in selected_columns:
                remaining_columns.append(column)

    else:
        remaining_columns = []
        for column in frame.columns:
            if column not in selected_columns:
                remaining_columns.append(column)

    if not remaining_columns:
        return error_response("This action would remove every column. Keep at least one feature.")

    updated_frame = frame[remaining_columns]
    datasets[dataset_id] = updated_frame
    return jsonify(make_dataset_response(dataset_id, updated_frame))


def make_chart(frame, data):
    chart_type = data.get("type")

    if chart_type == "histogram":
        column = data.get("column")
        if not isinstance(column, str) or column not in frame.columns:
            raise ValueError("Choose a valid column for the histogram.")

        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError("This column has no numeric values to plot.")

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.hist(values, bins="auto", density=True, alpha=0.65, label="Histogram")

        # A KDE estimates a smooth curve from nearby data points.
        # Sample large datasets so this calculation stays quick.
        if len(values) > 1 and np.std(values) > 0:
            spread = np.std(values, ddof=1)
            sample = values
            if len(sample) > 5000:
                sample = sample[np.linspace(0, len(sample) - 1, 5000, dtype=int)]

            bandwidth = 1.06 * spread * len(values) ** (-0.2)
            x_values = np.linspace(values.min() - spread, values.max() + spread, 250)
            distances = (x_values[:, None] - sample) / bandwidth
            density = np.exp(-0.5 * distances**2).mean(axis=1)
            density /= bandwidth * np.sqrt(2 * np.pi)
            axis.plot(x_values, density, color="#f08c52", label="KDE")

        axis.set(title=f"Distribution of {column}", xlabel=column, ylabel="Density")
        axis.legend()
        figure.tight_layout()
        return figure, ""

    if chart_type == "scatter":
        x_column = data.get("x")
        y_column = data.get("y")
        if (
            not isinstance(x_column, str)
            or not isinstance(y_column, str)
            or x_column not in frame.columns
            or y_column not in frame.columns
        ):
            raise ValueError("Choose valid columns for the scatter plot.")

        points = pd.DataFrame({
            "x": pd.to_numeric(frame[x_column], errors="coerce"),
            "y": pd.to_numeric(frame[y_column], errors="coerce"),
        }).dropna()
        points = points[np.isfinite(points).all(axis=1)]
        if points.empty:
            raise ValueError("These columns have no numeric rows in common.")
        if len(points) > 10_000:
            points = points.iloc[np.linspace(0, len(points) - 1, 10_000, dtype=int)]

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.scatter(points["x"], points["y"], alpha=0.7, color="#6258e8")
        axis.set(title=f"{y_column} vs {x_column}", xlabel=x_column, ylabel=y_column)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        return figure, ""

    if chart_type == "heatmap":
        numeric_data = frame.select_dtypes(include="number")
        if numeric_data.shape[1] < 2:
            raise ValueError("A correlation heatmap needs at least two numeric columns.")

        note = ""
        if numeric_data.shape[1] > 25:
            numeric_data = numeric_data.iloc[:, :25]
            note = "Showing the first 25 numeric columns."

        correlations = numeric_data.corr()
        labels = [str(column) for column in correlations.columns]
        figure, axis = plt.subplots(figsize=(8, 6))
        color_map = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_title("Numeric feature correlation")
        figure.colorbar(color_map, ax=axis, label="Correlation")
        figure.tight_layout()
        return figure, note

    raise ValueError("Choose histogram, scatter, or heatmap.")


@app.post("/api/plot")
def create_plot():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("A JSON chart request is required.")

    dataset_id = data.get("dataset_id")
    frame = find_dataset(dataset_id)
    if frame is None:
        return error_response("Dataset not found. Upload it again to continue.", 404)

    try:
        figure, note = make_chart(frame, data)
        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, format="png", dpi=140, bbox_inches="tight")
        plt.close(figure)
    except ValueError as error:
        return error_response(str(error))

    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    return jsonify({
        "image": f"data:image/png;base64,{encoded_image}",
        "note": note,
    })

if __name__ == "__main__":
    app.run(debug=True)
