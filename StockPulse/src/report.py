import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional
from app import da as da1
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet


def _exists(p):
    try: return os.path.exists(p)
    except: return False

def create_candlestick_with_anomalies(df: pd.DataFrame, static_dir: str = "frontend/static"):

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if not {"date", "open", "high", "low", "close", "anomaly"}.issubset(df.columns):
        raise ValueError("DataFrame must contain columns: date, open, high, low, close, anomaly")

    df_sorted = df.sort_values("date").copy()
    df_sorted["date_num"] = mdates.date2num(pd.to_datetime(df_sorted["date"]))

    fig, ax = plt.subplots(figsize=(12,6))


    for idx, row in df_sorted.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.plot([row['date_num'], row['date_num']], [row['low'], row['high']], color='black')
        ax.add_patch(plt.Rectangle(
            (row['date_num'] - 0.3, min(row['open'], row['close'])),
            0.6,
            abs(row['close'] - row['open']),
            color=color
        ))


    anomalies = df_sorted[df_sorted['anomaly'] == -1]
    ax.scatter(anomalies['date_num'], anomalies['close'], color='red', s=20, marker='o', label='Anomaly')

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.set_title("Candlestick Chart with Anomalies Highlighted", fontsize=16)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(static_dir, exist_ok=True)
    path = os.path.join(static_dir, "fig_candles.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

def build_report_payload(
    df: pd.DataFrame,
    static_dir: str = "frontend/static",
    label_col: Optional[str] = None,
    save_plots: bool = False
) -> dict:

    os.makedirs(static_dir, exist_ok=True)


    try:
        candlestick_path = create_candlestick_with_anomalies(df, static_dir=static_dir)
    except Exception as e:
        print("Candlestick creation error:", e)
        candlestick_path = None


    eval_payload = None
    supervised = False
    if label_col and label_col in df.columns:
        try:
            eval_payload = da1.evaluate_model(
                pdf=df,
                y_true_col=label_col,
                static_dir=static_dir,
                save_plots=save_plots
            )
            supervised = True
        except Exception as e:
            eval_payload = {"error": str(e)}
    else:
        try:
            eval_payload = da1.evaluate_model_unsupervised(
                pdf=df,
                static_dir=static_dir
            )
            supervised = False
        except Exception as e:
            eval_payload = {"error": str(e)}


    rows = len(df)
    cols = len(df.columns)
    anoms = int((df["anomaly"] == -1).sum()) if "anomaly" in df.columns else 0


    top_cols = ["date","close","volume","upper_band","lower_band","anomaly_score","anomaly"]
    for c in top_cols:
        if c not in df.columns:
            df[c] = pd.NA
    top_anoms = pd.DataFrame(columns=top_cols)
    if "anomaly" in df.columns and "anomaly_score" in df.columns:
        tmp = df[df["anomaly"] == -1].copy()

        top_anoms = tmp.sort_values("anomaly_score", ascending=True)[top_cols].head(25)


    images = [os.path.basename(candlestick_path)] if candlestick_path else []


    pdf_file = os.path.join(static_dir, "report.pdf")
    downloads = [os.path.basename(pdf_file)] if _exists(pdf_file) else []

    insights = []


    if anoms > 0:
        insights.append(f"Detected {anoms} anomalies. These may indicate unusual volatility or market manipulation.")
    else:
        insights.append("No anomalies detected. Market behavior appears relatively stable.")


    if "return" in df.columns:
        mean_return = df["return"].mean()
        if mean_return > 0:
            insights.append(f"Average returns are positive ({mean_return:.2%}). Consider riding upward momentum with caution.")
        else:
            insights.append(f"Average returns are negative ({mean_return:.2%}). Focus on capital preservation and risk control.")


    volatility = df["return"].std()
    if volatility > 0.02:  # قيمة افتراضية للتقلب العالي
        insights.append(f"High volatility detected (std={volatility:.2%}). Use smaller positions or stop-loss strategies.")
    else:
        insights.append(f"Low volatility observed (std={volatility:.2%}). Trend-following strategies may be effective.")


    if "close" in df.columns:
        last_close = df["close"].iloc[-1]
        avg_close = df["close"].mean()
        if last_close > avg_close:
            insights.append(f"Current price ({last_close:.2f}) is above the average ({avg_close:.2f}). This suggests bullish sentiment.")
        else:
            insights.append(f"Current price ({last_close:.2f}) is below the average ({avg_close:.2f}). This suggests bearish sentiment.")


    if "volume" in df.columns:
        avg_vol = df["volume"].mean()
        last_vol = df["volume"].iloc[-1]
        if last_vol > avg_vol * 1.5:
            insights.append("Recent trading volume is significantly higher than average. This may confirm a strong move.")
        elif last_vol < avg_vol * 0.5:
            insights.append("Recent trading volume is much lower than average. Market interest may be fading.")


    if "upper_band" in df.columns and "lower_band" in df.columns and "close" in df.columns:
        last_close = df["close"].iloc[-1]
        upper = df["upper_band"].iloc[-1]
        lower = df["lower_band"].iloc[-1]
        if last_close >= upper:
            insights.append("Price touched the upper Bollinger Band. Potential overbought condition.")
        elif last_close <= lower:
            insights.append("Price touched the lower Bollinger Band. Potential oversold condition.")


    insights.append("Highlighted anomalies in the candlestick chart indicate potential risk or opportunity points.")
    insights.append("Diversify across different assets to reduce exposure to single-stock risk.")
    insights.append("Always set stop-loss levels when anomalies suggest high volatility.")
    insights.append("Use anomalies as signals for further investigation, not as standalone trading triggers.")




    meta = {
        "title": "StockPulse – Analysis Report",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows": rows,
        "cols": cols,
        "anomalies": anoms,
        "mode": "supervised" if supervised else "unsupervised",
    }


    payload = {
        "meta": meta,
        "evaluation": eval_payload,
        "top_anomalies_table": top_anoms.to_dict(orient="records"),
        "images": images,
        "downloads": downloads,
        "insights": insights,
    }


    try:
        with open(os.path.join(static_dir, "report_payload.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

    return payload



def save_report_as_pdf(payload: dict, static_dir: str = "frontend/static"):

    os.makedirs(static_dir, exist_ok=True)
    pdf_path = os.path.join(static_dir, "report.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []


    elements.append(Paragraph(payload["meta"]["title"], styles['Title']))
    elements.append(Spacer(1, 12))


    meta = payload["meta"]
    meta_text = f"""
    Generated at: {meta['generated_at']}<br/>
    Rows: {meta['rows']} | Columns: {meta['cols']}<br/>
    Anomalies detected: {meta['anomalies']}<br/>
    Mode: {meta['mode']}
    """
    elements.append(Paragraph(meta_text, styles['Normal']))
    elements.append(Spacer(1, 12))




    if "images" in payload and payload["images"]:
        chart_path = os.path.join(static_dir, payload["images"][0])
        if _exists(chart_path):
            elements.append(Paragraph("Candlestick Chart:", styles['Heading2']))
            elements.append(RLImage(chart_path, width=400, height=250))
            elements.append(Spacer(1, 12))

    if "insights" in payload and payload["insights"]:
        elements.append(Paragraph("Insights:", styles['Heading2']))
        for ins in payload["insights"]:
            elements.append(Paragraph(f"- {ins}", styles['Normal']))
        elements.append(Spacer(1, 12))


    if "top_anomalies_table" in payload and payload["top_anomalies_table"]:
        elements.append(Paragraph("Top Anomalies:", styles['Heading2']))
        table_data = [list(payload["top_anomalies_table"][0].keys())]  # header
        for row in payload["top_anomalies_table"]:
            table_data.append([str(row[col]) for col in row])
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4F81BD")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER')
        ]))
        elements.append(table)


    doc.build(elements)
    return pdf_path

