import skops.io as sio
import gradio as gr

# Load model with proper trusted types for deployment
model_path = "Model/drug_pipeline.skops"

# Get untrusted types first
unknown_types = sio.get_untrusted_types(file=model_path)

# Load model with trusted types (numpy.dtype is safe for sklearn models)
pipe = sio.load(model_path, trusted=unknown_types)

def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    return pipe.predict([features])[0]

inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]
outputs = gr.Label()

examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

title = "Drug Classification"
description = "Enter the details to correctly identify Drug type."
article = "This app is part of CI/CD ML pipeline."

if __name__ == "__main__":
    gr.Interface(
        fn=predict_drug,
        inputs=inputs,
        outputs=outputs,
        examples=examples,
        title=title,
        description=description,
        article=article,
    ).launch()