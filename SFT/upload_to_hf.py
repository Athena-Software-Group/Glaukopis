from huggingface_hub import HfApi, upload_folder, login

# 1. Log in right here (paste your Write token)
login(token="HF_TOKEN_HERE")

repo_id = "YOUR_REPO_ID/qwen2.5-14b-sft-lora"

# 2. Create repo and upload
api = HfApi()
api.create_repo(repo_id=repo_id, private=True, exist_ok=True)

upload_folder(
    repo_id=repo_id,
    folder_path="models/qwen2.5_14b_sft_lora",
    repo_type="model"
)