# App/drug_app.py
import skops.io as sio
import gradio as gr

# Load pipeline (assumes path exists)
pipe = sio.load("../Model/drug_pipeline.skops", trusted=True)

def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """
    Predict drugs based on patient features.
    """
    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]
    label = f"Predicted Drug: {predicted_drug}"
    return label

inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]
outputs = [gr.Label(num_top_classes=5)]
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

title = "Drug Classification"
description = "Enter the details to correctly identify Drug type?"
article = "This app is part of the CI/CD for ML tutorial."

if __name__ == "__main__":
    gr.Interface(
        fn=predict_drug,
        inputs=inputs,
        outputs=outputs,
        examples=examples,
        title=title,
        description=description,
        article=article,
        theme=gr.themes.Soft(),
    ).launch()
