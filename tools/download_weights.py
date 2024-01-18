import os
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

def prepare_image_encoder():
    print(f"Preparing image encoder weights...")
    local_dir = "./pretrained_models"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

if __name__ == '__main__':
    prepare_image_encoder()