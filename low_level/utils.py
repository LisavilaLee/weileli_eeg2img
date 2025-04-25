import datetime
import os

from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

# 设置 Hugging Face 镜像站点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"


def print_current_time():
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

    repo_path = os.path.join(local_dir, repo_id.split('/')[1])  # 默认下到huggingface文件夹中，以仓库名作为子文件夹

    if not os.path.isdir(repo_path):
        os.makedirs(repo_path)

    if is_one_file:
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
            local_dir_use_symlinks=False,  # 把所有文件真正复制过来（不是软链接）
            # ignore_patterns=["*.md", "*.txt"]  # 可选：忽略不必要文件
        )
    print(f"文件已下载至：{file_path_local}")

    return file_path_local
