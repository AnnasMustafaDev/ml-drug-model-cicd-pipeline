# deploy_hf.py
from huggingface_hub import Repository, login
import shutil
import os
import sys

HF_TOKEN = os.environ.get("HF")
HF_REPO = os.environ.get("HF_REPO")  # e.g., "AnnasMustafaDev/drug-classification"

if not HF_TOKEN or not HF_REPO:
    sys.exit("HF token or HF_REPO not set as environment variables!")

# Login
login(HF_TOKEN)

# Clone the HF repo
repo = Repository(local_dir="hf_space", clone_from=f"https://huggingface.co/spaces/{HF_REPO}", use_auth_token=HF_TOKEN)

# Copy app files
shutil.copytree("App", "hf_space", dirs_exist_ok=True)
shutil.copytree("Model", "hf_space/Model", dirs_exist_ok=True)

# Commit & push
repo.push_to_hub(commit_message="Update app + model")
print("✅ Deployment complete!")
