import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='EEG Model Training Script')
    parser.add_argument('--project', type=str, default="weileli-eeg2img", help='WandB project name')
    parser.add_argument('--entity', type=str, default="weileli-bcmi", help='WandB entity name')
    parser.add_argument('--name', type=str, default="low-level-atms", help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--insubject', default=True, help='Flag to indicate within-subject training')
    parser.add_argument('--encoder_type', type=str, default='ATMS_low_level',
                        choices=['ATMS_low_level'], help='Encoder type')
    parser.add_argument('--img_encoder', type=str, default='Proj_img', help='Image encoder type')
    parser.add_argument('--logger', default=True, help='Enable logging')
    parser.add_argument('--gpu', type=str, default='cuda:1', help='GPU device to use')
    parser.add_argument('--subjects', nargs='+', default=['sub-01'], help='List of subject IDs')

    return parser
