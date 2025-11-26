install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	@echo "## Model Metrics" > report.md
	@cat ./Results/metrics.txt >> report.md
	@echo '\n## Confusion Matrix Plot' >> report.md
	@echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md

# Save results/model into "update" branch using provided username/email
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add Model Results || true
	git commit -am "Update with new results" || echo "Nothing to commit"
	git push --force origin HEAD:update

# Deploy to Hugging Face (App folder + Model + Results)
deploy:
	python -m pip install huggingface_hub
	hf-cli login --token $(HF)
	# push App folder (Space) and Model/Results as files to the repo or HuggingFace
	# for Spaces: huggingface-cli repo create <space-name> --type space   (manually)
	# Here we push to the hf repo root (example).
	huggingface-cli repo create $(HF_REPO) --type space || true
	git clone https://huggingface.co/spaces/$(HF_REPO) hf-space-temp || true
	rsync -av --delete App/ hf-space-temp/
	rsync -av --delete Model/ hf-space-temp/Model/
	rsync -av --delete Results/ hf-space-temp/Results/
	cd hf-space-temp && git add . && git commit -m "Deploy update" && git push || echo "No changes to push"
	rm -rf hf-space-temp
