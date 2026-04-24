# 自动化下载主流情感数据集脚本
# 保存路径：decoder_soft_prompt/data/{dataset_name}

from datasets import load_dataset
import os

def download_and_save(dataset_name, save_dir):
    print(f"Downloading {dataset_name} ...")
    ds = load_dataset(dataset_name)
    ds.save_to_disk(save_dir)
    print(f"Saved to {save_dir}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_and_save("mrm8488/goemotions", "data/goemotions")
    download_and_save("facebook/empathetic_dialogues", "data/empathetic_dialogues")
