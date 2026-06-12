import os
import time
import argparse
import time
import matplotlib.pyplot as plt
from typing import Callable, Iterator, cast
from pathlib import Path
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from config import TrainConfig, load_config, load_dataset, load_model, load_optimizer
from dataset import parse_directory, Tensor_to_PIL, PIL_to_Tensor,VIFDataset, read_and_split, ImageLabels
from models.models2 import DBTFuse1, FuseInput, FuseOutput
from models.loss import ALL_Loss

# from tqdm import tqdm


def parse_commandline():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--train', dest='train', action='store_true', default=False, help='train model')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='test model')
    # parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=False, help='use pretrained model')
    return parser.parse_args()


def timer(fn: Callable):
    def warpper(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)
        used_time = round(time.time() - start_time)
        minutes, seconds = divmod(used_time, 60)
        print(f'{fn.__name__} {minutes}:{seconds}s', end=' ')
        return ret
    return warpper


def plot_loss(train_losses, valid_losses, save_path):
    """Plot and save the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(valid_losses, label="Valid Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def train(config: TrainConfig, device: torch.device,a,b, c: float, d: float, save_loss_plot_path: str):
    model: DBTFuse1 = cast(DBTFuse1, load_model(config.model, device))
    # if args.pretrained and hasattr(config.model, 'pretrained') and not args.train:
    #     print(f'Loading pretrained weights from {config.model.pretrained}')
    #     model.load_state_dict(torch.load(config.model.pretrained))
    

    # print(f'Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    criterion = ALL_Loss(a=a,b=b, c=c, d=d).to(device)
    criterion.to(device)
    
    trainloader, validloader = load_dataset(config.dataset)[:2]
    optim = load_optimizer(config.train.optimizer, model.parameters())
    lr_scheduler = StepLR(optim, step_size=50, gamma=0.5)
    accumulation_steps = config.train.accumulation_steps
    best_valid_loss = float('inf')
    print(f'train <{model._get_name()}> on {device.type}')

    # Track losses
    train_losses = []
    valid_losses = []

    for epoch in range(config.train.epochs):
        # print("training epochs", config.train.epochs)
       
        train_loss = train_loop(
            model, criterion, trainloader,
            optim, accumulation_steps, device
        )
        train_losses.append(train_loss)  # Record training loss

        torch.cuda.empty_cache()
        lr_scheduler.step()
        # current_lr = lr_scheduler.get_last_lr()[0]
        print(f'{epoch + 1:3}/{config.train.epochs}: {train_loss=}')

   
        valid_loss = valid_loop(model, criterion, validloader, device=device)
        valid_losses.append(valid_loss)  # Record validation loss

        if valid_loss < best_valid_loss:
            checkpoint(model)
            best_valid_loss = valid_loss
            print("weight update")
        
        # Plot and save the loss curve after each epoch
    plot_loss(train_losses, valid_losses, os.path.join(save_loss_plot_path, f'loss_alpha10_mu1{epoch + 1}.png'))


def pad(vi: Tensor, ir: Tensor, patch_size: int):
    H, W = vi.shape[-2:]
    pad_h = 0 if (mod_h := H % patch_size) == 0 else patch_size - mod_h
    pad_w = 0 if (mod_w := W % patch_size) == 0 else patch_size - mod_w
    left, top = pad_w >> 1, pad_h >> 1
    right, down = pad_w - left, pad_h - top
    box = (left, right, top, down)
    vi = F.pad(vi, box, mode='reflect')
    ir = F.pad(ir, box, mode='reflect')
    return FuseInput(vi, ir), torch.Size((H, W))


def tensor_to_image(img: Tensor, size: torch.Size) -> Image.Image:
    # img is already a floating-point tensor in the [0, 1] range
    H, W = size
    pad_h = img.size(-2) - H
    pad_w = img.size(-1) - W
    sh, sw = pad_h >> 1, pad_w >> 1
    # Crop and remove the batch dimension
    img_cropped = img.squeeze(0)[:, sh:sh + H, sw:sw + W]
    return cast(Image.Image, Tensor_to_PIL(img_cropped))

class Color:
    def __init__(self, kr=0.299, kg=0.587, kb=0.114):
        self.kr = kr
        self.kg = kg
        self.kb = kb

    def RGB_to_YCbCr(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Convert an RGB tensor in the [0, 1] range to YCbCr.
        Y is in [0, 1], while Cb and Cr are in [-0.5, 0.5].
        """
        r, g, b = image.chunk(3, dim=-3)
        y = self.kr * r + self.kg * g + self.kb * b
        cb = (b - y) / (2 * (1 - self.kb))
        cr = (r - y) / (2 * (1 - self.kr))
        return y, cb, cr

    def YCbCr_to_RGB(self, y: Tensor, cb: Tensor, cr: Tensor) -> Tensor:
        """
        Convert Y ([0, 1]), Cb, and Cr ([-0.5, 0.5]) back to RGB in [0, 1].
        """
        r = y + 2 * cr * (1 - self.kr)
        b = y + 2 * cb * (1 - self.kb)
        g = (y - self.kr * r - self.kb * b) / self.kg
        
        # Concatenate the result and clamp it to [0, 1] to avoid floating-point errors
        rgb = torch.cat([r, g, b], dim=-3).clamp(0.0, 1.0)
        return rgb
    
def test(config: TrainConfig, path: str, device: torch.device):
    save_path = Path(path)
    model: DBTFuse1 = load_model(config.model, device)
    
    if config.model.pretrained:
        model.load_state_dict(torch.load(config.model.pretrained, map_location=device))
        model_name = Path(config.model.pretrained).stem
    else:
        model_name = "untrained_model"
    
    save_path = save_path.with_name(save_path.name + "-Test").joinpath(model_name)
    print(f'{save_path=}')
    os.makedirs(save_path, exist_ok=True)
    
    model.to(device)
    model.eval()
    
    # Update 1: instantiate the Color helper
    color_util = Color()

    with torch.no_grad():
        for (vi_color_raw, vi_gray_raw, ir_raw, filename) in load_data(path):
            vi_color, vi_gray, ir = vi_color_raw.to(device), vi_gray_raw.to(device), ir_raw.to(device)

            # 1. Prepare model inputs (range [-1, 1])
            inputs, size = pad(vi_gray, ir, config.model.params['patch_size'])
            
            # 2. Get the model output and convert it to [0, 1]
            output: FuseOutput = model(inputs)
            y_fused = (output.fusion + 1.0) / 2.0

            # 3. Prepare the visible-light image for colorization (range [0, 1])
            vi_color_normalized = (vi_color + 1.0) / 2.0
            
            # 4. Match spatial size by padding the normalized color image
            H, W = size
            pad_h = y_fused.size(-2) - H
            pad_w = y_fused.size(-1) - W
            left, top = pad_w >> 1, pad_h >> 1
            right, down = pad_w - left, pad_h - top
            vi_color_padded = F.pad(vi_color_normalized, (left, right, top, down), mode='reflect')

            # 5. Extract the chroma channels (Cb, Cr)
            _, cb, cr = color_util.RGB_to_YCbCr(vi_color_padded)

            # 6. Combine the new luminance (Y) with the original chroma (Cb, Cr) and convert back to RGB
            final_rgb_fusion = color_util.YCbCr_to_RGB(y_fused, cb, cr)

            # 7. Save the image
            image: Image.Image = tensor_to_image(final_rgb_fusion, size)
            save_file = os.path.join(save_path, filename)
            image.save(save_file)
            print(f'saved: {save_file}{" " * 20}', end='\r')


def load_data(dir: str) -> Iterator[tuple[Tensor, Tensor, Tensor, str]]:
    if 'PETMRI' in dir:
        msrs = Path(dir)
        with open(msrs.joinpath('labels.txt'), 'r') as fp:
            filenames = fp.read().splitlines()
            for filename in filenames:
                vi_file = msrs.joinpath('vi', filename)
                ir_file = msrs.joinpath('ir', filename)

                # a) Color VI (used for colorization)
                vi_color_pil = Image.open(vi_file).convert('RGB')
                vi_color = PIL_to_Tensor(vi_color_pil).unsqueeze(0)

                # b) Grayscale VI (used as model input)
                vi_gray = PIL_to_Tensor(vi_color_pil.convert('L')).unsqueeze(0)

                # c) Grayscale IR
                ir = PIL_to_Tensor(Image.open(ir_file).convert('L')).unsqueeze(0)

                # Always return four values
                yield vi_color, vi_gray, ir, filename   
    elif 'SPECTMRI' in dir:
        road = Path(dir)
        with open(road.joinpath('labels.txt'), 'r') as fp:
            filenames = fp.read().splitlines()
            for filename in filenames:
                vi_file = road.joinpath('vi', filename)
                ir_file = road.joinpath('ir', filename)
                
                # Update 2: load the VI image twice
                # a) Load the color VI for colorization
                vi_color_pil = Image.open(vi_file).convert('RGB')
                vi_color = PIL_to_Tensor(vi_color_pil).unsqueeze(0)
                # b) Load the grayscale VI for model input
                vi_gray = PIL_to_Tensor(vi_color_pil.convert('L')).unsqueeze(0)
                # c) Load the grayscale IR
                ir = PIL_to_Tensor(Image.open(ir_file).convert('L')).unsqueeze(0)
                
                # Update 3: yield four values
                yield vi_color, vi_gray, ir, filename   
    elif 'CTMRI' in dir:
        road = Path(dir)
        with open(road.joinpath('labels.txt'), 'r') as fp:
            filenames = fp.read().splitlines()
            for filename in filenames:
                vi_file = road.joinpath('vi', filename)
                ir_file = road.joinpath('ir', filename)
                
                # Update 2: load the VI image twice
                # a) Load the color VI for colorization
                vi_color_pil = Image.open(vi_file).convert('RGB')
                vi_color = PIL_to_Tensor(vi_color_pil).unsqueeze(0)
                # b) Load the grayscale VI for model input
                vi_gray = PIL_to_Tensor(vi_color_pil.convert('L')).unsqueeze(0)
                # c) Load the grayscale IR
                ir = PIL_to_Tensor(Image.open(ir_file).convert('L')).unsqueeze(0)
                
                # Update 3: yield four values
                yield vi_color, vi_gray, ir, filename         
    else:
        raise ValueError()




@timer
def train_loop(
    model: DBTFuse1,
    criterion: nn.Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    accumulation_steps: int,
    device: torch.device,
) -> float:
    train_loss = 0.0
    model.train()
    for step, (vi, ir) in enumerate(trainloader):
        inputs = FuseInput(vi.to(device), ir.to(device))
        outputs: FuseOutput = model(inputs)
        loss: Tensor = criterion(inputs, outputs)
        train_loss += loss.item()
        loss = loss / accumulation_steps
        loss.backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(trainloader):
            optimizer.step()
            optimizer.zero_grad()
        print(f'{step}/{len(trainloader)}{" " * 10}', end='\r', flush=True)
    return train_loss / len(trainloader)


@timer
def valid_loop(
    model: DBTFuse1,
    criterion: nn.Module,
    validloader: DataLoader, 
    device: torch.device
) -> float:
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for _, (vi, ir) in enumerate(validloader):
            inputs = FuseInput(vi.to(device), ir.to(device))
            outputs: FuseOutput = model(inputs)
            loss: Tensor = criterion(inputs, outputs)
            valid_loss += loss.item()
        return valid_loss / len(validloader)


def checkpoint(model: nn.Module):
    model_name = model.__class__.__name__
    folder_path = '/root/autodl-tmp/tongyong/medical/result'
    filepath = os.path.join(folder_path, 'mde1125')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(model.state_dict(), filepath)
    print(f'saved: {filepath}\n')


if __name__ == '__main__':
    config_path = "/root/autodl-tmp/tongyong/medical/config/model2.yaml"
    config = load_config(config_path)

    # Specify the path for saving loss plots
    save_loss_plot_path = "/root/autodl-tmp/tongyong/medical/result"  # Change this to the folder where you want to save images
    os.makedirs(save_loss_plot_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_commandline()
    if args.train:
        train(config, device=torch.device('cuda'),a=1,b=1,c=2,d=5, save_loss_plot_path=save_loss_plot_path)
    elif args.test:
        
        config.model.pretrained ="/root/autodl-tmp/tongyong/medical/result/mde1125"
        assert os.path.isfile(config.model.pretrained), FileNotFoundError
        test(config, "/root/autodl-tmp/data/testdata/PETMRI", device)
        test(config, "/root/autodl-tmp/data/testdata/SPECTMRI", device)
        test(config, "/root/autodl-tmp/data/testdata/CTMRI", device)
