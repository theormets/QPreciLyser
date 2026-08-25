import os
import torch
import cv2
import gradio as gr
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics_config'

# ================== MODEL LOADING ==================
try:
    model = YOLO("best.pt")
    model.to("cpu")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
# ===================================================


# ----------------- HISTOGRAM -----------------
def plot_histogram(areas):
    fig, ax = plt.subplots()

    ax.hist(areas, bins=15, edgecolor="black")
    ax.set_title("Precipitate Size Distribution")
    ax.set_xlabel("Precipitate Size (px²)")
    ax.set_ylabel("Count")
    ax.grid(False)

    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    fig.savefig(tmp.name)
    plt.close(fig)

    return tmp.name, Image.open(tmp.name).convert("RGB")


# ----------------- CORE ANALYSIS -----------------
def analyze_image(img, micron_per_px, conf_thresh, iou_thresh):
    if model is None:
        return None, None, None, None, None, None, "❌ Model not loaded"
    if img is None:
        return None, None, None, None, None, None, "❌ Please upload an image"

    results = model.predict(
        source=img,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=640,
        max_det=3000,
        retina_masks=True
    )[0]

    if results.masks is None:
        return img, pd.DataFrame(), None, None, None, None, "⚠️ No precipitates detected"

    annotated = img.copy()
    rows = []
    areas_px = []
    total_precip_area_px = 0
    total_pixels = img.shape[0] * img.shape[1]

    masks = results.masks.data.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    for i, mask in enumerate(masks):

        mask_resized = cv2.resize(
            mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        smoothed = cv2.GaussianBlur(mask_binary, (5, 5), 0)

        _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        # 🔥 FIX: ensure correct datatype for findContours
        smoothed = cv2.convertScaleAbs(smoothed)

        contours, _ = cv2.findContours(
            smoothed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_L1
        )

        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)

        area_px = cv2.contourArea(cnt)
        total_precip_area_px += area_px
        areas_px.append(area_px)

        area_um2 = area_px * (micron_per_px ** 2)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0

        rows.append([
            i + 1,
            round(float(area_px), 1),
            round(float(area_um2), 4),
            round(circularity, 3),
            round(float(confidences[i]), 3)
        ])

        cv2.polylines(annotated, [cnt], True, (0, 255, 0), 2)

    area_fraction = (total_precip_area_px / total_pixels) * 100

    df = pd.DataFrame(
        rows,
        columns=["ID", "Area (px²)", "Area (µm²)", "Circularity", "Confidence"]
    )

    _, img_path = tempfile.mkstemp(suffix=".jpg")
    cv2.imwrite(img_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    hist_path, hist_img = plot_histogram(areas_px)

    excel_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
    df.to_excel(excel_path, index=False, engine="openpyxl")

    summary = (
        f"Total Precipitates: {len(df)} | "
        f"Area Fraction: {area_fraction:.2f}% | "
        f"Avg Size: {df['Area (µm²)'].mean():.4f} µm²"
    )

    return annotated, df, img_path, hist_img, hist_path, excel_path, summary


# ----------------- GRADIO UI -----------------
with gr.Blocks(title="PrecipiDetect") as demo:

    gr.Markdown("# 🔬 PrecipiDetect")
    gr.Markdown(
        "Description: Detect and quantify precipitates in microstructure images. "
        "The tool extracts dimensional features, calculates area metrics, and presents "
        "the precipitate size distribution through tabulated data and histograms."
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Microstructure Image")
            micron_input = gr.Number(label="Micron per Pixel", value=0.1)

            conf_slider = gr.Slider(
                0.01, 1.0,
                value=0.1,
                step=0.01,
                label="Confidence"
            )

            iou_slider = gr.Slider(
                0.01, 1.0,
                value=0.1,
                step=0.01,
                label="IOU"
            )

            analyze_btn = gr.Button("Analyze")

        with gr.Column():
            output_img = gr.Image(type="numpy", label="Annotated Image")
            output_summary = gr.Textbox(label="Summary")

    with gr.Row():
        output_df = gr.Dataframe(label="Precipitate Analysis Table")
        hist_img = gr.Image(label="Histogram Plot")

    with gr.Row():
        download_img = gr.File(label="Download Annotated Image")
        download_hist = gr.File(label="Download Histogram")
        download_excel = gr.File(label="Download Excel Report")

    analyze_btn.click(
        analyze_image,
        inputs=[input_img, micron_input, conf_slider, iou_slider],
        outputs=[
            output_img,
            output_df,
            download_img,
            hist_img,
            download_hist,
            download_excel,
            output_summary
        ],
    )

if __name__ == "__main__":
    demo.launch()
