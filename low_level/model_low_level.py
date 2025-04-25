import os
import argparse
from args_low_level import get_parser

parser = get_parser()
args = parser.parse_args()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

import tqdm
from eegdataset_low_level import EEGDataset
from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
# from util import wandb_logger

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
import torch.optim.lr_scheduler as lr_scheduler
import datetime
import itertools
import csv

# VAE
image_processor = VaeImageProcessor()

path_sdxl = "/userhome2/liweile/EEG_Image_decode/sdxl_turbo/"
pipe = DiffusionPipeline.from_pretrained(path_sdxl, torch_dtype=torch.float, variant="fp16")

if hasattr(pipe, 'vae'):
    for param in pipe.vae.parameters():
        param.requires_grad = False

vae = pipe.vae.to(args.gpu)
vae.requires_grad_(False)
vae.eval()

import torch
import torch.nn as nn


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x


class encoder_low_level(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024,
                 num_blocks=1):
        super(encoder_low_level, self).__init__()
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, 128) for _ in range(num_subjects)])

        # CNN upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(8064, 1024, kernel_size=4, stride=2, padding=1),  # (1, 1) -> (2, 2)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # (2, 2) -> (4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (4, 4) -> (8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (8, 8) -> (16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (16, 16) -> (32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32) -> (64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),  # Keep size (64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0),  # Output shape (4, 64, 64)
        )

    def forward(self, x):
        # Apply subject-wise linear layer
        x = self.subject_wise_linear[0](x)  # Output shape: (batchsize, 63, 128)
        # Reshape to match the input size for the upsampler
        x = x.view(x.size(0), 8064, 1, 1)  # Reshape to (batch_size, 8064, 1, 1)
        out = self.upsampler(x)  # Pass through the upsampler
        return out


def train_model(eegmodel, imgmodel, dataloader, optimizer, device, text_features_all, img_features_all, save_dir,
                epoch):
    eegmodel.train()
    total_loss = 0

    mae_loss_fn = nn.L1Loss()
    image_reconstructed = False  # Flag to track if the image has been reconstructed
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')
    if not os.path.exists(epoch_save_dir):
        os.makedirs(epoch_save_dir)
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        eeg_features = eegmodel(eeg_data[:, :, :250]).float()

        regress_loss = mae_loss_fn(eeg_features, img_features)

        loss = regress_loss
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            if not image_reconstructed:
                z = eeg_features
                x_rec = vae.decode(z).sample
                x_train = vae.decode(img_features).sample
                image_rec = image_processor.postprocess(x_rec, output_type='pil')
                image_train = image_processor.postprocess(x_train, output_type='pil')
                # Use label to create a unique file name
                for i, label in enumerate(labels.tolist()):
                    save_path = os.path.join(epoch_save_dir, f"reconstructed_image_{label}.png")
                    image_rec[i].save(save_path)

                    save_path2 = os.path.join(epoch_save_dir, f"train_image_{label}.png")
                    image_train[i].save(save_path2)
                    image_reconstructed = True

        del eeg_features, img_features, eeg_data

    torch.cuda.empty_cache()

    average_loss = total_loss / (batch_idx + 1)
    accuracy = 0
    top5_acc = 0
    return average_loss, accuracy, top5_acc


def evaluate_model(eegmodel, imgmodel, dataloader, device, text_features_all, img_features_all, k, save_dir, epoch):
    eegmodel.eval()
    total_loss = 0
    mae_loss_fn = nn.L1Loss()
    accuracy = 0
    top5_acc = 0

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            # eeg_data = eeg_data.permute(0, 2, 1)
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            eeg_features = eegmodel(eeg_data[:, :, :250]).float()
            regress_loss = mae_loss_fn(eeg_features, img_features)
            total_loss += regress_loss.item()

            if epoch % 10 == 0:
                epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')
                if not os.path.exists(epoch_save_dir):
                    os.makedirs(epoch_save_dir)
                z = eeg_features
                x_rec = vae.decode(z).sample
                image_rec = image_processor.postprocess(x_rec, output_type='pil')

                # Use label to create a unique file name
                for i, label in enumerate(labels.tolist()):
                    base_save_path = os.path.join(epoch_save_dir, f"reconstructed_image_{label + 1}.png")
                    save_path = base_save_path
                    k = 0
                    # Check if the file already exists
                    while os.path.exists(save_path):
                        save_path = os.path.join(epoch_save_dir, f"reconstructed_image_{label + 1}_{k}.png")
                        k += 1
                    # Save the image
                    image_rec[i].save(save_path)
                del eeg_features, img_features, eeg_data, image_rec, x_rec
                continue
            del eeg_features, img_features, eeg_data

    torch.cuda.empty_cache()
    average_loss = total_loss / (batch_idx + 1)
    return average_loss, accuracy, top5_acc


def main_train_loop(sub, current_time, eeg_model, img_model, train_dataloader, test_dataloader, optimizer, device,
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all,
                    config, logger=None):
    # Introduce cosine annealing scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    # logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model, logger)
    logger.watch(img_model, logger)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_accuracy = 0.0

    results = []  # List to store results for each epoch

    for epoch in range(config.epochs):

        # Add date-time prefix to save_dir
        print(f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: epoch {epoch + 1}/{config.epochs}")
        train_save_dir = f'{current_time}_vae_train_imgs'
        print(f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: Start training...")
        train_loss, train_accuracy, features_tensor = train_model(eeg_model, img_model, train_dataloader, optimizer,
                                                                  device, text_features_train_all,
                                                                  img_features_train_all, save_dir=train_save_dir,
                                                                  epoch=epoch)
        if (epoch + 1) % 5 == 0:
            # Get the current time and format it as a string (e.g., '2024-01-17_15-30-00')
            if config.insubject == True:
                os.makedirs(f"./models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)
                file_path = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch + 1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            else:
                os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)
                file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/{epoch + 1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Update learning rate
        scheduler.step()

        # Add date-time prefix to save_dir
        test_save_dir = f'{current_time}_vae_imgs'
        print(f"{[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]}: Start testing...")
        test_loss, test_accuracy, top5_acc = evaluate_model(eeg_model, img_model, test_dataloader, device,
                                                            text_features_test_all, img_features_test_all, k=200,
                                                            save_dir=test_save_dir, epoch=epoch)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Append results for this epoch
        epoch_results = {
            "epoch": epoch + 1,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

        results.append(epoch_results)
        # If the test accuracy in the current epoch is the best, save the model and related information
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()

            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "Epoch": epoch
        })

        print(
            f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        torch.cuda.empty_cache()

    logger.finish()
    return results


def main():
    # Set device based on the argument
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    data_path = args.data_path
    subjects = args.subjects
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in subjects:
        # Re-initialize the models for each subject
        eeg_model = globals()[args.encoder_type]()
        img_model = globals()[args.img_encoder]()

        eeg_model.to(device)
        img_model.to(device)

        optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters(), img_model.parameters()), lr=args.lr)

        if args.insubject:
            train_dataset = EEGDataset(data_path, subjects=[sub], train=True)
            test_dataset = EEGDataset(data_path, subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(data_path, exclude_subject=sub, train=True)
            test_dataset = EEGDataset(data_path, exclude_subject=sub, train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        text_features_train_all = None
        text_features_test_all = None
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, current_time, eeg_model, img_model, train_loader, test_loader, optimizer, device,
                                  text_features_train_all, text_features_test_all, img_features_train_all,
                                  img_features_test_all,
                                  config=args, logger=args.logger)

        # Save results to a CSV file
        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)

        if args.insubject:
            results_file = os.path.join(results_dir, f"{args.encoder_type}_{sub}.csv")
        else:
            results_file = os.path.join(results_dir, f"{args.encoder_type}_cross_exclude_{sub}.csv")

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')


if __name__ == '__main__':
    main()
