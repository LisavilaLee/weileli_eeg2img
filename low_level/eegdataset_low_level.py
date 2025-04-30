import torch
from diffusers import DiffusionPipeline
import numpy as np
from torchvision import transforms
from PIL import Image

import os
import utils
from tqdm import tqdm

from utils import get_json, hf_mirror_download, get_current_time

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

    def __init__(self, exclude_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes=None,
                 pictures=None, val_size=None, pipe=None, device='cpu'):
        self.root_dir, self.data_path, self.img_directory_training, self.img_directory_test, self.huggingface_repo_path= get_json()
        self.train = train
        self.subject_list = os.listdir(self.data_path)
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

        if pipe is None:
            try:
                print("Pipe not detected. Starting download of SDXL-Turbo model...")
                sdxl_path = hf_mirror_download("stabilityai/sdxl-turbo", self.huggingface_repo_path)

                pipe = DiffusionPipeline.from_pretrained(
                    sdxl_path,
                    torch_dtype=torch.float,
                    variant="fp16"
                )
            except Exception as e:
                print("Failed to load the model:", e)
                raise RuntimeError("EEGDataset is unable to initialize the SDXL-Turbo diffusion pipeline.") from e
        self.vlmodel = pipe.vae.to(self.device)

        features_filename = os.path.join(self.root_dir, 'features_data/train_image_vae_512.pt') if self.train else os.path.join(
            self.root_dir, 'features_data/test_image_vae_512.pt')

        if os.path.exists(features_filename):
            print(
                f"{[get_current_time()]}: Exist VAE image latent features file {features_filename}.")
            saved_features = torch.load(features_filename, weights_only=True)
            self.img_features = saved_features['image_latent']
        else:
            self.img_features = self.ImageEncoder(self.img)     # shape: (16540, 4, 64, 64)
            torch.save({'image_latent': self.img_features}, features_filename)

    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []

        if self.train:
            directory = self.img_directory_training
        else:
            directory = self.img_directory_test

        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()

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
            img_directory = self.img_directory_training
        else:
            img_directory = self.img_directory_test

        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()

        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(folder_path, img) for img in all_images)

        for subject in self.subjects:
            if self.train:
                if subject == self.exclude_subject:
                    continue
                    # print("subject:", subject)
                file_name = 'preprocessed_eeg_training.npy'

                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)  # key of data: ['preprocessed_eeg_data', 'ch_names', 'times']

                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                # shape of data['times']: (300, ). In preprocessing_utils.py, sampling time range is [-0.2, 1.0]
                times = torch.from_numpy(data['times']).detach()[50:]   # times: (st=0.0s, ed=0.996s, step=0.004)
                ch_names = data['ch_names']

                n_classes = 1654
                samples_per_class = 10

                for i in range(n_classes):
                    start_index = i * samples_per_class
                    # preprocessing_eeg_data 在数据集官方文件中已说明按照字母序类别和类别对应图片名称顺序排序完毕。
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
            # data_list: list[1654] -> tensor([10, 4, 63, 250])
            # -1：自动推导出第一个维度（这里就是 16540 × 4 = 66160）；
            # *data_list[0].shape[2:] → 即 (63, 250)：表示只保留 通道数和时间点；
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])  # [1654x4x10=66160, 63, 250]
        else:
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)

        label_tensor = torch.cat(label_list, dim=0)

        if self.train:
            # label_tensor: (subjects * classes * 10 * 4)
            label_tensor = label_tensor.repeat_interleave(4)

        else:
            pass

        self.times = times
        self.ch_names = ch_names
        # ch_names: ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8',
        # 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz',
        # 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7',
        # 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']

        #   Data tensor shape: torch.Size([66160, 63, 250]), label tensor shape: torch.Size([66160]), text length: 1654, image length: 16540
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)

        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]

        return extracted_data

    def ImageEncoder(self, images):
        print(
            f"{[get_current_time()]}: Image VAE latent features encoder Begin.")
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
    train_dataset = EEGDataset(subjects=['sub-01'], train=False, device="cuda:1")
