import datetime
import os
import json
import wandb

import torch

from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

# 设置 Hugging Face 镜像站点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"


def get_current_time():
    """打印当前时间"""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def hf_mirror_download(repo_id, local_dir, is_one_file = False, filename=None, repo_type=None):
    """
        方便国内下载huggingface_hub中的仓库，同时支持下载整个仓库或者单个文件。

        Args:
            repo_id (`str`):
                A user or an organization name and a repo name separated by a `/`.
            local_dir (`str` or `Path`, *optional*):
                If provided, the downloaded file will be placed under this directory.
            is_one_file (`bool`, *optional*):
                Choose the way(whole repository / only one file) to download.
                Defaults to False, which means whole repository will be downloaded.
            filename (`str`):
                The name of the file in the repo.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if downloading from a dataset or space,
                `None` or `"model"` if downloading from a model. Default is `None`.
    """

    # 检查 repo_id 格式
    try:
        user, repo_name = repo_id.split('/')
    except ValueError:
        raise ValueError("repo_id 必须是 'user/repo' 格式，例如 'stabilityai/sdxl-turbo'")

    repo_path = os.path.join(local_dir, repo_name)  # 默认下到/userhome2/liweile/weileli_eeg2img/huggingface文件夹中，以仓库名作为子文件夹

    if not os.path.isdir(repo_path):
        os.makedirs(repo_path)

    if is_one_file:
        if not filename:
            raise ValueError("当 is_one_file=True 时，必须提供 filename 参数")
        file_path_local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=repo_path,
            repo_type=repo_type
        )
    else:
        file_path_local = snapshot_download(
            repo_id=repo_id,
            local_dir=repo_path,
            repo_type=repo_type
            # ignore_patterns=["*.md", "*.txt"]  # 可选：忽略不必要文件
        )
    print(f"文件已下载至：{file_path_local}")

    return file_path_local

def get_json():
    # Load the configuration from the JSON file
    config_path = "data_config.json"

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Access the paths from the config
    root_dir = config["root_dir"]
    data_path = config["data_path"]
    img_directory_training = config["img_directory_training"]
    img_directory_test = config["img_directory_test"]
    huggingface_repo_path = config["huggingface_repo_path"]

    return root_dir, data_path, img_directory_training, img_directory_test, huggingface_repo_path


class wandb_logger:
    def __init__(self, config):
        try:
            wandb.init(
                # Set the project where this run will be logged
                project=config['project'],
                name=config['name'],
                config=config,
                entity=config['entity'],
            )
        except:
            wandb.init(
                # Set the project where this run will be logged
                project=config.project,
                name=config.name,
                config=config,
                entity=config.entity,
            )

        self.config = config
        self.step = None

        _, self.data_path, _, _, _ = get_json()

    def log(self, data, step=None):
        if step is None:
            wandb.log(data)
        else:
            wandb.log(data, step=step)
            self.step = step

    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, figs):
        if self.step is None:
            wandb.log(figs)
        else:
            wandb.log(figs, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

    def watch(self, model, log):
        wandb.watch(model, log)

if __name__ == "__main__":
    hf_mirror_download("stabilityai/sdxl-turbo", "/userhome2/liweile/weileli_eeg2img/huggingface")