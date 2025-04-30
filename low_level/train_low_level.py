import os
import argparse
import datetime
from args_low_level import get_parser

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

from eegdataset_low_level import EEGDataset
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
import torch.optim.lr_scheduler as lr_scheduler
import itertools
import csv

from model_low_level import Proj_img, encoder_low_level
from utils import get_current_time, get_json, hf_mirror_download, wandb_logger


def train_model(eegmodel, imgmodel, dataloader, optimizer, device, img_features_all, save_dir,
                epoch, vae, image_processor):
    eegmodel.train()
    total_loss = 0

    mae_loss_fn = nn.L1Loss()
    image_reconstructed = False  # Flag to track if the image has been reconstructed
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
    if not os.path.exists(epoch_save_dir):
        os.makedirs(epoch_save_dir)
    for batch_idx, (eeg_data, labels, _, _, img, img_features) in enumerate(dataloader):
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
                x_rec = vae.decode(eeg_features).sample
                x_train = vae.decode(img_features).sample
                image_rec = image_processor.postprocess(x_rec, output_type='pil')
                image_train = image_processor.postprocess(x_train, output_type='pil')
                # Use label to create a unique file name
                for i, label in enumerate(labels.tolist()):
                    save_path = os.path.join(epoch_save_dir, f"reconstructed_image_{label + 1}.png")
                    image_rec[i].save(save_path)

                    save_path2 = os.path.join(epoch_save_dir, f"train_image_{label + 1}.png")
                    image_train[i].save(save_path2)
                    image_reconstructed = True

        del eeg_features, img_features, eeg_data

    torch.cuda.empty_cache()

    average_loss = total_loss / (batch_idx + 1)
    accuracy = 0
    top5_acc = 0
    return average_loss, accuracy, top5_acc


def evaluate_model(eegmodel, imgmodel, dataloader, device, img_features_all, k, save_dir, epoch, vae, image_processor):
    eegmodel.eval()
    total_loss = 0
    mae_loss_fn = nn.L1Loss()
    accuracy = 0
    top5_acc = 0

    with (torch.no_grad()):
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            # eeg_data = eeg_data.permute(0, 2, 1)
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            eeg_features = eegmodel(eeg_data[:, :, :250]).float()
            regress_loss = mae_loss_fn(eeg_features, img_features)
            total_loss += regress_loss.item()

            if (epoch + 1) % 10 == 0:
                epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
                if not os.path.exists(epoch_save_dir):
                    os.makedirs(epoch_save_dir)
                x_rec = vae.decode(eeg_features).sample
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


def main_train_loop(sub, eeg_model, img_model, train_dataloader, test_dataloader, optimizer, device,
                    img_features_train_all, img_features_test_all, config, vae, image_preocessor, current_time, logger=None):
    # Introduce cosine annealing scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model,logger)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_accuracy = 0.0

    results = []  # List to store results for each epoch

    for epoch in range(config.epochs):

        # Add date-time prefix to save_dir
        print(f"{[get_current_time()]}: epoch {epoch + 1}/{config.epochs}")
        train_save_dir = f'./low_level_img/{current_time}/train_imgs'
        if not os.path.exists(train_save_dir):
            os.makedirs(train_save_dir)
        print(f"{[get_current_time()]}: Start training...")
        train_loss, train_accuracy, features_tensor = train_model(eeg_model, img_model, train_dataloader, optimizer,
                                                                  device, img_features_train_all, save_dir=train_save_dir,
                                                                  epoch=epoch, vae=vae, image_processor=image_preocessor)
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
            print(f"{[get_current_time()]}: model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Update learning rate
        scheduler.step()

        # Add date-time prefix to save_dir
        test_save_dir = f'./low_level_img/{current_time}/test_imgs'
        if not os.path.exists(test_save_dir):
            os.makedirs(test_save_dir)
        print(f"{[get_current_time()]}: Start testing...")
        test_loss, test_accuracy, top5_acc = evaluate_model(eeg_model, img_model, test_dataloader, device, img_features_test_all, k=200,
                                                            save_dir=test_save_dir, epoch=epoch, vae=vae, image_processor=image_preocessor)
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
            "Epoch": epoch + 1
        })

        print(
            f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        torch.cuda.empty_cache()

    logger.finish()
    return results


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Set device based on the argument
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    subjects = args.subjects
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    _, _, _, _, huggingface_repo_path = get_json()

    # VAE
    image_processor = VaeImageProcessor()

    sdxl_path = hf_mirror_download("stabilityai/sdxl-turbo", huggingface_repo_path)
    pipe = DiffusionPipeline.from_pretrained(sdxl_path, torch_dtype=torch.float, variant="fp16")

    if hasattr(pipe, 'vae'):
        for param in pipe.vae.parameters():
            param.requires_grad = False

    vae = pipe.vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    # 单独被试数据开始训练
    for sub in subjects:
        # Re-initialize the models for each subject
        eeg_model = globals()[args.encoder_type]()
        img_model = globals()[args.img_encoder]()

        eeg_model.to(device)
        img_model.to(device)

        optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters(), img_model.parameters()), lr=args.lr)

        if args.insubject:
            train_dataset = EEGDataset(subjects=[sub], train=True, pipe=pipe, device=device)
            test_dataset = EEGDataset(subjects=[sub], train=False, pipe=pipe, device=device)
        else:
            train_dataset = EEGDataset(exclude_subject=sub, train=True, pipe=pipe, device=device)
            test_dataset = EEGDataset(exclude_subject=sub, train=False, pipe=pipe, device=device)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, eeg_model, img_model, train_loader, test_loader, optimizer, device,
                                  img_features_train_all, img_features_test_all, config=args, vae=vae,
                                  image_preocessor=image_processor, current_time=current_time, logger=args.logger)

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
