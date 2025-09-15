from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, lag, log, to_date, when
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib
import os, json
import sqlite3
import numpy as np

spark = SparkSession.builder.appName("StockAnalysis").getOrCreate()


def read_from_user(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = spark.read.csv(file_path, header=True, inferSchema=True)
    elif ext in [".xls", ".xlsx"]:
        pdf = pd.read_excel(file_path)
        df = spark.createDataFrame(pdf)
    elif ext == ".json":
        pdf = pd.read_json(file_path)
        df = spark.createDataFrame(pdf)
    elif ext == ".sql":
        conn = sqlite3.connect(file_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        first_table = tables.iloc[0, 0]
        pdf = pd.read_sql(f"SELECT * FROM {first_table}", conn)
        conn.close()
        df = spark.createDataFrame(pdf)
    else:
        raise ValueError("Unsupported file format.")
    return df


def preprocessing(df_input):
    if isinstance(df_input, pd.DataFrame):
        df_spark = spark.createDataFrame(df_input)
    else:
        df_spark = df_input

    df_spark = df_spark.toDF(*[c.strip().lower() for c in df_spark.columns])
    date_candidates = ["date", "price"]
    date_col = next((c for c in date_candidates if c in df_spark.columns), None)
    if date_col is None:
        raise KeyError(f"No valid date column found. Expected one of: {date_candidates}")

    if date_col != "date":
        df_spark = df_spark.withColumnRenamed(date_col, "date")

    df_spark = df_spark.withColumn(
        "date",
        when(col("date").rlike(r"^\d{4}-\d{2}-\d{2}$"), col("date")).otherwise(None)
    )
    df_spark = df_spark.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    df_spark = df_spark.filter(col("date").isNotNull())

    numeric_cols = [c for c in df_spark.columns if c != "date"]
    for c in numeric_cols:
        df_spark = df_spark.withColumn(c, col(c).cast(DoubleType()))

    return df_spark


def feature_extraction(df_spark):
    df_spark = df_spark.orderBy("date")
    window7 = Window.orderBy("date").rowsBetween(-6, 0)
    window30 = Window.orderBy("date").rowsBetween(-29, 0)
    df_spark = df_spark.withColumn("ma_7", mean("close").over(window7))
    df_spark = df_spark.withColumn("ma_30", mean("close").over(window30))
    window1 = Window.orderBy("date")
    df_spark = df_spark.withColumn("close_lag1", lag("close", 1).over(window1))
    df_spark = df_spark.withColumn("log_return", log(col("close") / col("close_lag1")))
    df_spark = df_spark.withColumn("return", (col("close") - col("close_lag1")) / col("close_lag1"))
    window20 = Window.orderBy("date").rowsBetween(-19, 0)
    df_spark = df_spark.withColumn("sma_20", mean("close").over(window20))
    df_spark = df_spark.withColumn("std_20", stddev("close").over(window20))
    df_spark = df_spark.withColumn("upper_band", col("sma_20") + 2 * col("std_20"))
    df_spark = df_spark.withColumn("lower_band", col("sma_20") - 2 * col("std_20"))
    df_spark = df_spark.na.drop()
    return df_spark.toPandas()


def feature_selection(pdf_clean):

    numeric_df = pdf_clean.select_dtypes(include=['number']).copy()
    if numeric_df.shape[1] == 0:

        df_reduced = pdf_clean.copy()
        selected_cols = list(df_reduced.columns)
        return df_reduced, selected_cols


    selector = VarianceThreshold(threshold=1e-8)
    selector.fit(numeric_df)
    selected_cols_num = numeric_df.columns[selector.get_support()].tolist()

    df_reduced = numeric_df[selected_cols_num].copy()


    if "date" in pdf_clean.columns:
        df_reduced["date"] = pdf_clean["date"].values
        selected_cols = selected_cols_num + ["date"]
    else:
        selected_cols = selected_cols_num

    return df_reduced, selected_cols


def data_model(pdf, selected_cols, n_estimators=200, contamination=0.02, random_state=42):

    df = pdf.copy()


    features_all = [c for c in selected_cols if c != "date"]
    X_all = df[features_all].astype(float)
    valid_mask = X_all.notna().all(axis=1)
    X_all = X_all[valid_mask]
    idx_all = X_all.index


    if "date" in df.columns:

        df_sorted = df.loc[valid_mask].copy()

        try:
            df_sorted["date"] = pd.to_datetime(df_sorted["date"])
        except Exception:
            pass
        if pd.api.types.is_datetime64_any_dtype(df_sorted.get("date", pd.Series(dtype="datetime64[ns]"))):
            df_sorted = df_sorted.sort_values("date")
            X_all = X_all.loc[df_sorted.index]
            idx_all = X_all.index

    n = len(X_all)
    if n < 50:
        split = n
        X_train = X_all
        X_test = X_all.iloc[:0]
    else:
        split = int(n * 0.7)
        X_train = X_all.iloc[:split]
        X_test = X_all.iloc[split:]


    sel = VarianceThreshold(threshold=1e-8)
    Xtr = sel.fit_transform(X_train)
    keep_cols = X_train.columns[sel.get_support()].tolist()


    X_train = X_train[keep_cols]
    X_full = X_all[keep_cols]


    iso = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        bootstrap=True,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    ).fit(X_train)


    train_scores = iso.decision_function(X_train)
    full_scores = iso.decision_function(X_full)



    tail_q = float(np.clip(contamination, 0.005, 0.1))
    threshold = np.quantile(train_scores, tail_q)

    final_pred = np.where(full_scores <= threshold, -1, 1)


    df.loc[idx_all, "anomaly"] = final_pred
    df.loc[idx_all, "anomaly_score"] = full_scores


    joblib.dump(iso, "IsolationForest.pkl")

    return df


def evaluate_model_unsupervised(
        pdf: pd.DataFrame,
        score_col: str = "anomaly_score",
        pred_col: str = "anomaly",
        static_dir: str = "frontend/static"
) -> dict:

    os.makedirs(static_dir, exist_ok=True)
    df = pdf.copy()

    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in results.")

    total = int(len(df))
    anoms = int((df[pred_col] == -1).sum()) if pred_col in df.columns else None
    anom_rate = (anoms / total * 100.0) if (anoms is not None and total) else None


    s = pd.to_numeric(df[score_col], errors="coerce")
    score_stats = {
        "mean_score": float(np.nanmean(s)),
        "std_dev_score": float(np.nanstd(s)),
        "min_score": float(np.nanmin(s)),
        "percentile_01": float(np.nanpercentile(s, 1)),
        "percentile_05": float(np.nanpercentile(s, 5)),
        "percentile_10": float(np.nanpercentile(s, 10)),
        "median_score": float(np.nanpercentile(s, 50)),
        "percentile_90": float(np.nanpercentile(s, 90)),
        "percentile_95": float(np.nanpercentile(s, 95)),
        "max_score": float(np.nanmax(s)),
    }


    suggested_thresholds = {
        "percentile_01": float(np.nanpercentile(s, 1)),
        "percentile_05": float(np.nanpercentile(s, 5)),
        "percentile_10": float(np.nanpercentile(s, 10)),
    }


    rolling_std = None
    if "date" in df.columns:
        try:
            dfd = df.copy()
            dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
            dfd = dfd.dropna(subset=["date"]).sort_values("date")
            if pred_col in dfd.columns:
                roll = (dfd[pred_col] == -1).astype(int).rolling(window=max(10, int(len(dfd) * 0.05)),
                                                                 min_periods=5).mean()
                rolling_std = float(np.nanstd(roll))
        except Exception:
            rolling_std = None

    result = {
        "unsupervised": True,
        "total_rows": total,
        "anomalies": anoms,
        "anomaly_rate_pct": float(f"{anom_rate:.2f}") if anom_rate is not None else None,
        "score_stats": score_stats,
        "suggested_thresholds": suggested_thresholds,
        "rolling_anomaly_rate_std": rolling_std
    }


    try:
        with open(os.path.join(static_dir, "evaluation_unsupervised.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return result
