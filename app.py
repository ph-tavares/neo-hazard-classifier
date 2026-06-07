"""Gradio app for the NEO hazard classifier.

Decoupled from training: it only LOADS model.joblib (no retraining). The host
(Hugging Face Spaces) can be swapped without touching the model — only this file.
"""
import joblib
import pandas as pd

import gradio as gr

from src.neo_features import FEATURE_COLUMNS, build_input_frame

_bundle = joblib.load("model.joblib")
_model = _bundle["model"]
_scaler = _bundle["scaler"]


def predict(absolute_magnitude_h, miss_distance_min_au, miss_distance_mean_au,
            relative_velocity_max_kms, n_close_approaches):
    frame = build_input_frame({
        "absolute_magnitude_h": absolute_magnitude_h,
        "miss_distance_min_au": miss_distance_min_au,
        "miss_distance_mean_au": miss_distance_mean_au,
        "relative_velocity_max_kms": relative_velocity_max_kms,
        "n_close_approaches": n_close_approaches,
    })
    scaled = pd.DataFrame(_scaler.transform(frame), columns=FEATURE_COLUMNS)
    proba = float(_model.predict_proba(scaled)[0, 1])
    label = "⚠️ Potencialmente perigoso (PHA)" if proba >= 0.5 else "✅ Não perigoso"
    return f"{label} — probabilidade de PHA: {proba:.1%}"


with gr.Blocks(title="Classificador de Risco de NEOs") as app:
    gr.Markdown(
        "# Classificador de Risco de NEOs (PHA)\n"
        "Modelo treinado com dados da **NASA NeoWs**. Insira as características de "
        "um asteroide para estimar se é **potencialmente perigoso**.\n\n"
        "*Classificador para NEOs com histórico de aproximação registrado — não "
        "estima a prevalência do catálogo inteiro.*"
    )
    with gr.Row():
        with gr.Column():
            h = gr.Number(label="Magnitude absoluta (H)", value=20.0)
            miss_min = gr.Number(label="Distância mínima de aproximação (UA)", value=0.05)
            miss_mean = gr.Number(label="Distância média de aproximação (UA)", value=0.20)
        with gr.Column():
            vel_max = gr.Number(label="Velocidade relativa máxima (km/s)", value=15.0)
            n_appr = gr.Number(label="Nº de aproximações registradas", value=5)
    out = gr.Textbox(label="Resultado", lines=2)
    btn = gr.Button("Classificar", variant="primary")
    btn.click(predict, inputs=[h, miss_min, miss_mean, vel_max, n_appr], outputs=out)
    gr.Examples(
        examples=[[15.27, 0.03, 0.10, 30.0, 28], [25.0, 0.30, 0.45, 8.0, 2]],
        inputs=[h, miss_min, miss_mean, vel_max, n_appr],
    )

if __name__ == "__main__":
    app.launch()
