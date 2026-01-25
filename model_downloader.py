from huggingface_hub import snapshot_download
# Sfrom modelscope import snapshot_download
# snapshot_path = snapshot_download(repo_id="Hcompany/Holo2-4B",
#                                    local_dir="checkpoints/Holo2-4B",
#                                    )

dataset_path = snapshot_download(
    repo_id="OS-Copilot/ScreenSpot-v2",
    repo_type="dataset",
    local_dir="./data/screenspot_v2",
)