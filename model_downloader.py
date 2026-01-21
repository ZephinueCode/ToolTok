from huggingface_hub import snapshot_download
# Sfrom modelscope import snapshot_download
snapshot_path = snapshot_download(repo_id="Hcompany/Holo2-4B",
                                   local_dir="checkpoints/Holo2-4B",
                                   )

# dataset_path = snapshot_download(
#     repo_id="osunlp/Multimodal-Mind2Web",
#     repo_type="dataset",
#     local_dir="./data/mind2web",
# )