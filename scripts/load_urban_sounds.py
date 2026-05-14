import os
import huggingface_hub


audio_datasets_path = "data/input/UrbanSound8K/"
os.makedirs(audio_datasets_path, exist_ok=True)
huggingface_hub.snapshot_download(
    repo_id="MahiA/UrbanSound8K",
    repo_type="dataset",
    local_dir=os.path.join(audio_datasets_path, "UrbanSound8K"),
)
