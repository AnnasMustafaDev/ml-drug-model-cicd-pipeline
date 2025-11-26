# Drug Classification ML CI/CD Pipeline

This repository contains an end-to-end **Machine Learning project** for **Drug Classification** with a fully automated **CI/CD pipeline** that trains the model, generates evaluation metrics, and deploys a Gradio app to **Hugging Face Spaces**.

---

## 📁 Project Structure

```
Project/
 ├── App/
 │    └── drug_app.py         # Gradio web application
 │    └── requirements.txt    # Dependencies for the app
 ├── Data/
 │    └── drug.csv            # Kaggle Drug Classification dataset
 ├── Model/
 │    └── drug_pipeline.skops # Trained model pipeline
 ├── Results/
 │    └── metrics.txt         # Evaluation metrics
 │    └── model_results.png   # Confusion matrix or plots
 ├── train.py                 # Training script
 ├── requirements.txt         # Project dependencies
 ├── Makefile                 # Build and deploy commands
 └── .github/workflows/
      ├── ci.yml              # Continuous Integration workflow
      └── cd.yml              # Continuous Deployment workflow
```

---

## ⚡ Setup & Installation

1. Clone the repository:

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the dataset `drug.csv` in the `Data/` folder.

---

## 🏋️‍♂️ Training the Model

Run the training script locally:

```bash
python train.py
```

This will generate:

* `Model/drug_pipeline.skops`
* `Results/metrics.txt`
* `Results/model_results.png`

---

## 🌐 Running the App Locally

```bash
python App/drug_app.py
```

This launches a Gradio interface to predict drug types based on patient features:

* Age
* Sex
* Blood Pressure
* Cholesterol
* Na_to_K ratio

---

## 🔧 CI/CD Workflow

### Continuous Integration (CI)

* Triggered on **push** or **pull request**.
* Steps:

  1. Checkout repository
  2. Setup Python environment
  3. Install dependencies
  4. Run `train.py` to retrain the model
  5. Generate evaluation report (`Makefile: eval`)
  6. Upload artifacts (model & results)

### Continuous Deployment (CD)

* Triggered after CI completes successfully.
* Steps:

  1. Checkout repository
  2. Setup Python environment
  3. Install dependencies
  4. Deploy the Gradio app + model to **Hugging Face Spaces** using the HF token.

### GitHub Secrets Required

* `USER_NAME` — Git username used in CI commits
* `USER_EMAIL` — Git email used in CI commits
* `HF` — Hugging Face access token (with write scope)
* `HF_REPO` — Hugging Face Space repo name (username/space-name)

> `GITHUB_TOKEN` is automatically provided by GitHub Actions.

---

## 🧪 Testing & Verification

1. Push a change to the repository to trigger CI/CD.
2. Check **GitHub Actions → Workflows** for CI/CD status.
3. If deployment is successful, the **Gradio app** will be live on Hugging Face Spaces.

---

## ⚡ Examples

Example input in the Gradio app:

| Age | Sex | Blood Pressure | Cholesterol | Na_to_K |
| --- | --- | -------------- | ----------- | ------- |
| 30  | M   | HIGH           | NORMAL      | 15.4    |
| 35  | F   | LOW            | NORMAL      | 8       |
| 50  | M   | HIGH           | HIGH        | 34      |

---

## 📦 Requirements

* Python >= 3.10
* skops >= 0.10
* gradio
* pandas, numpy, scikit-learn
* huggingface_hub

---

## 📖 References

* DataCamp Tutorial: [CI/CD for Machine Learning](https://www.datacamp.com/tutorial/ci-cd-for-machine-learning)
* Hugging Face Spaces: [https://huggingface.co/spaces](https://huggingface.co/spaces)

---

## ✅ Notes

* Make sure `Model/drug_pipeline.skops` exists before running the app.
* CI/CD is configured to work without CML; reports are generated locally and artifacts uploaded.
* The deployed app on Hugging Face Spaces automatically rebuilds whenever new files are pushed via CD workflow.

---
