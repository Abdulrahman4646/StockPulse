from pyspark.sql import SparkSession
import pandas as pd
import os
import uuid
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import src.report as report_builder
import sqlite3
import src.data_analysis as da

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

TEMP_DIR = os.path.join("frontend", "temp_uploads")
UPLOAD_DIR = os.path.join("uploads")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

REQUIRED_COLS = ["close", "high", "low", "open", "volume"]

spark = SparkSession.builder.appName("StockAnalysis").getOrCreate()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/preview", methods=["POST"])
def preview():
    try:
        file = request.files['file']
        if not file or file.filename == "":
            return jsonify({"status": "error", "message": "No file uploaded"})

        ext = os.path.splitext(file.filename)[1].lower()
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_path = os.path.join(TEMP_DIR, temp_filename)
        file.save(temp_path)

        if ext == ".csv":
            df_pd = pd.read_csv(temp_path)
        elif ext in [".xls", ".xlsx"]:
            df_pd = pd.read_excel(temp_path)
        elif ext == ".json":
            df_pd = pd.read_json(temp_path)
        elif ext == ".sql":
            conn = sqlite3.connect(temp_path)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            first_table = tables.iloc[0, 0]
            df_pd = pd.read_sql(f"SELECT * FROM {first_table}", conn)
            conn.close()
        else:
            return jsonify({"status": "error", "message": "Unsupported file format"})

        preview_html = df_pd.head(10).to_html(classes="preview-table", index=False)

        try:
            os.remove(temp_path)
        except:
            pass

        return jsonify({"status": "ok", "html": preview_html})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['file']
        if not file or file.filename == "":
            return redirect(url_for("home"))

        ext = os.path.splitext(file.filename)[1].lower()
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_path = os.path.join(TEMP_DIR, temp_filename)
        file.save(temp_path)

        if ext in [".csv", ".xls", ".xlsx", ".json", ".sql"]:
            df_spark = da.read_from_user(temp_path)
        else:
            return jsonify({"error": "Unsupported file format."})

        df_spark = df_spark.toDF(*[c.strip().lower() for c in df_spark.columns])

        if not all(col in df_spark.columns for col in REQUIRED_COLS):
            return jsonify({"error": f"File must contain columns: {', '.join(REQUIRED_COLS)}"})

        db_clean = da.preprocessing(df_spark)
        db_features = da.feature_extraction(db_clean)
        df_final, selected_cols = da.feature_selection(db_features)


        n_estimators = request.form.get("n_estimators", type=int, default=200)
        contamination = request.form.get("contamination", type=float, default=0.02)
        random_state = request.form.get("random_state", type=int, default=42)

        pdb = da.data_model(df_final, selected_cols,
                            n_estimators=n_estimators,
                            contamination=contamination,
                            random_state=random_state)

        result_path = os.path.join(app.static_folder, "results.csv")
        pdb.to_csv(result_path, index=False, encoding="utf-8")

        try:
            os.remove(temp_path)
        except:
            pass

        return redirect(url_for("analysis"))

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/analysis")
def analysis():
    result_path = os.path.join(app.static_folder, "results.csv")
    if not os.path.exists(result_path):
        return redirect(url_for("home"))

    df = pd.read_csv(result_path)

    eval_payload = None
    try:
        if "label" in df.columns:
            eval_payload = da.evaluate_model(
                pdf=df,
                y_true_col="label",
                static_dir=app.static_folder,
                save_plots=False
            )
        else:
            eval_payload = da.evaluate_model_unsupervised(
                pdf=df,
                static_dir=app.static_folder
            )
    except Exception as e:
        eval_payload = {"error": str(e)}

    return render_template("analysis.html", eval=eval_payload)


@app.route("/report")
def report_page():
    result_path = os.path.join(app.static_folder, "results.csv")
    if not os.path.exists(result_path):
        return redirect(url_for("analysis"))

    df = pd.read_csv(result_path)

    label_col = None  

    payload = report_builder.build_report_payload(
        df=df,
        static_dir=app.static_folder,
        label_col=label_col,
        save_plots=False
    )

    report_builder.save_report_as_pdf(payload, static_dir=app.static_folder)

    return render_template("report.html", data=payload)


@app.route("/dashboard")
def dashboard_page():
    return render_template("Dashboard.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
