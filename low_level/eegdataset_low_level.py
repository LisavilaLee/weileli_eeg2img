import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.deprecated.stable_diffusion_variants.pipeline_stable_diffusion_inpaint_legacy import \
    preprocess_image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests

import os
import datetime
from tqdm import tqdm

from args_low_level import get_parser
from utils import hf_mirror_download

import clip
import open_clip

import json

# Load the configuration from the JSON file
config_path = "data_config.json"

with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Access the paths from the config
root_dir = config["root_dir"]
data_path = config["data_path"]
features_path = config["features_path"]
img_directory_training = config["img_directory_training"]
img_directory_test = config["img_directory_test"]

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


preprocess_train = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
])


class EEGDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """

    def __init__(self, data_path, exclude_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes=None,
                 pictures=None, val_size=None, device='cpu'):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject
        self.val_size = val_size
        self.device = device

        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img = self.load_data()

        self.data = self.extract_eeg(self.data, time_window)

        print(f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: EEGDataset load & extract data done.")

        """
            pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", 
            torch_dtype=torch.float, variant="fp16")
            vlmodel = pipe.vae
            
            posterior = vlmodel.encode(low_level_image).latent_dist
        """

        # sdxl_tubo_weight_path = hf_mirror_download(repo_id="stabilityai/sdxl-turbo", local_dir=os.path.join(root_dir, "huggingface"))
        sdxl_tubo_weight_path = "/userhome2/liweile/EEG_Image_decode/sdxl_turbo/"
        pipe = DiffusionPipeline.from_pretrained(sdxl_tubo_weight_path, torch_dtype=torch.float, variant="fp16")
        self.vlmodel = pipe.vae.to(self.device)

        vae_custom_features_filename = "/userhome2/liweile/weileli_eeg2img/features_data/train_image_vae_512.pt"
        self.img_features = self.ImageEncoder(self.img)

        torch.save({'image_latent': self.img_features.cpu(),}, vae_custom_features_filename)

        features_filename = os.path.join(f"/userhome2/liweile/EEG_Image_decode/Generation", 'variables',
                                         f'train_image_latent_512.pt')
        saved_features = torch.load(features_filename, weights_only=True)
        print(f"saved_features: {saved_features.keys()}")
        self.saved_img_features = saved_features['image_latent']
        print(f"saved_img_features.shape {self.saved_img_features.shape}")

        # if self.classes is None and self.pictures is None:
        #     features_filename = os.path.join('variables',
        #                                      f'train_image_latent_512.pt') if self.train else os.path.join(
        #         'variables', f'test_image_latent_512.pt')
        #
        #     if os.path.exists(features_filename):
        #         print(
        #             f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: Exist features file {features_filename}.")
        #         saved_features = torch.load(features_filename, weights_only=True)
        #         print(f"saved_features: {saved_features.keys()}")
        #         self.text_features = None
        #         self.img_features = saved_features['image_latent']
        #         # print(f"img_features shape: {self.img_features.shape}")
        #     else:
        #         print("Error")
        # else:
        #     self.text_features = self.Textencoder(self.text)
        #     self.img_features = self.ImageEncoder(self.img)

        print(f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: Text features and image features done.")

    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []

        if self.train:
            directory = img_directory_training
        else:
            directory = img_directory_test

        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()

        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        for dir in dirnames:

            try:
                idx = dir.index('_')
                description = dir[idx + 1:]
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue

            new_description = f"This picture is {description}"
            texts.append(new_description)

        if self.train:
            img_directory = img_directory_training
        else:
            img_directory = img_directory_test

        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()

        if self.classes is not None and self.pictures is not None:
            print(
                f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: self.classes is not None and self.pictures is not None.")
            images = []
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if
                                  img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            print(
                f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: self.classes is not None and self.pictures is None.")
            images = []
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if
                                  img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            print(f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: self.classes is None.")
            images = []
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
        else:

            print("Error")

        for subject in self.subjects:
            if self.train:
                if subject == self.exclude_subject:
                    continue
                    # print("subject:", subject)
                file_name = 'preprocessed_eeg_training.npy'

                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path,
                               allow_pickle=True)  # key of data: ['preprocessed_eeg_data', 'ch_names', 'times']

                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                # shape of data['times']: (300, ). In preprocessing_utils.py, sampling time range is [-0.2, 1.0]
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']

                n_classes = 1654
                samples_per_class = 10

                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + 1]
                            labels = torch.full((1,), c, dtype=torch.long).detach()
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)

                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[
                                                      start_index: start_index + samples_per_class]
                        labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        # preprocessing_eeg_data在数据集官方文件中已说明按照字母序类别和类别对应图片名称顺序排序完毕。
                        preprocessed_eeg_data_class = preprocessed_eeg_data[
                                                      start_index: start_index + samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)


            else:
                if subject == self.exclude_subject or self.exclude_subject == None:
                    file_name = 'preprocessed_eeg_test.npy'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    # print(f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: Load subject {subject} test data done.")
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                    times = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']
                    n_classes = 200  # Each class contains 1 images

                    samples_per_class = 1

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:  # If we've defined specific classes and the current class is not in the list, skip
                            continue
                        start_index = i * samples_per_class  # Update start_index for each class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index + samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)  # Add labels to the label list
                else:
                    continue

        if self.train:
            # data_list: list[1654->tensor([10, 4, 63, 250])]
            # -1：自动推导出第一个维度（这里就是 16540 × 4 = 66160）；
            # *data_list[0].shape[2:] → 即 (63, 250)：表示只保留 通道数和时间点数；
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])  # [1654x4x10=66160, 63, 250]
        else:
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)

        label_tensor = torch.cat(label_list, dim=0)

        if self.train:
            # label_tensor: (subjects * classes * 10 * 4)
            label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)
                mapping = {val.item(): index for index, val in enumerate(unique_values)}
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

        else:
            pass

        self.times = times
        self.ch_names = ch_names

        #   Data tensor shape: torch.Size([66160, 63, 250]), label tensor shape: torch.Size([66160]), text length: 1654, image length: 16540
        print(
            f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")

        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)

        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]

        return extracted_data

    def Textencoder(self, text):
        text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(self.device)

        with torch.no_grad():
            text_features = self.vlmodel.encode_text(text_inputs)

        text_features = F.normalize(text_features, dim=-1).detach()

        print(f"text_features shape: {text_features.shape}")

        return text_features

    def ImageEncoder(self, images):
        print(
            f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: Image encoder Begin. Total images number is {len(images)}.")
        batch_size = 20
        image_features_list = []

        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(
                self.device)

            with torch.no_grad():
                batch_image_features = self.vlmodel.encode(image_inputs).latent_dist.mode()
                # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)

        print(f"image_features shape: {image_features.shape}")

        return image_features

    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index]
        label = self.labels[index]

        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes) * 1 * 80
                index_n_sub_train = len(self.classes) * 10 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes) * 1 * 80
                index_n_sub_train = len(self.classes) * 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)

        text = self.text[text_index]
        img = self.img[img_index]

        text_features = -1
        img_features = self.img_features[img_index]

        return x, label, text, text_features, img, img_features

    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same


if __name__ == "__main__":
    # Instantiate the dataset and dataloader
    # data_path = "/home/ldy/Workspace/THINGS/EEG/osfstorage-archive"  # Replace with the path to your data
    data_path = data_path
    train_dataset = EEGDataset(data_path, subjects=['sub-01'], train=True, device="cuda:5")
    # test_dataset = EEGDataset(data_path, subjects=['sub-01'], train=False)
    # train_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=True)
    # test_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=False)
    # train_dataset = EEGDataset(data_path, train=True)
    # test_dataset = EEGDataset(data_path, train=False)
    # 训练的eeg数据：torch.Size([16540, 4, 17, 100]) [训练图像数量，训练图像重复数量，通道数，脑电信号时间点]
    # 测试的eeg数据：torch.Size([200, 80, 17, 100])
    # 1秒 'times': array([-0.2 , -0.19, -0.18, ... , 0.76,  0.77,  0.78, 0.79])}
    # 17个通道'ch_names': ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']
    # 100 Hz
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    #
    # i = 80 * 1 - 1
    # x, label, text, text_features, img, img_features = test_dataset[i]
    # print(f"Index {i}, Label: {label}, text: {text}")
    # Image.open(img)



