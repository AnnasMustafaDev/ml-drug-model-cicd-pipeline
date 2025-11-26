# deploy_hf.py
from huggingface_hub import HfApi, login
import shutil
import os
import sys
from pathlib import Path

HF_TOKEN = os.environ.get("HF")
HF_REPO = os.environ.get("HF_REPO")  # e.g., "AnnasMustafaDev/drug-classification"

if not HF_TOKEN or not HF_REPO:
    sys.exit("HF token or HF_REPO not set as environment variables!")

# Login
login(token=HF_TOKEN)

# Initialize API
api = HfApi()

# Create local directory for staging
local_dir = Path("hf_space")
local_dir.mkdir(exist_ok=True)

# Copy app files
shutil.copytree("App", local_dir, dirs_exist_ok=True)
shutil.copytree("Model", local_dir / "Model", dirs_exist_ok=True)

# Upload to Hugging Face Space
try:
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=HF_REPO,
        repo_type="space",
        token=HF_TOKEN,
        commit_message="Update app + model"
    )
    print("✅ Deployment complete!")
except Exception as e:
    print(f"❌ Deployment failed: {e}")
    sys.exit(1)