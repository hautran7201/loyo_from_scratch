import hydra
import json
import os
import wandb
from omegaconf import OmegaConf
from logging import NullHandler
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR

# Model
from model import YoloV1
from loss import YoloV1Loss
from darknet import Darknet
# Dataset
from dataset import VOCDataset
# Utils
from omegaconf import DictConfig
from utils import (cellboxes_to_boxes, non_max_suppression, plot_image, get_bboxes, mean_average_precision, day_time)


hydra.output_subdir = None
@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    # Device
    device = config.model.train.device


    # Wandb
    # ==================================
    day, time = day_time()
    if config.model.wandb:
        wandb.login()
        # Loss log
        run = wandb.init(
            project="voc_yolov1",
            group=f"experiment_{day}_{time}", 
            job_type="train",
            tags=["baseline", "paper1"],
        )


    # Image transforms
    # ==================================
    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)), 
            transforms.ToTensor()
        ]
    )

    print('--> Start training\n')


    # Dataset           
    # ==================================
    data_config = config.model.train.dataset
    train_dataset = hydra.utils.instantiate(
        data_config.train_data,
        transform=transform
    )
    test_dataset = hydra.utils.instantiate(
        data_config.test_data,
        transform=transform
    )
    subset_sampler = SubsetRandomSampler(range(1000))
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=data_config.shuffle,
        batch_size=data_config.batch_size,
        pin_memory=data_config.pin_memory,
        drop_last=data_config.drop_last
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=subset_sampler,
        # batch_size=data_config.batch_size,
        pin_memory=data_config.pin_memory,
        drop_last=data_config.drop_last
    )


    # Darknet model         
    # ==================================
    darknet = Darknet(conv_only=True, init_weight=True)


    # Yolo model          
    # ==================================
    Model = YoloV1(
        S=config.model.S,
        B=config.model.B,
        C=config.model.C,
        features=darknet.modules
    ).to(device)


    # Loss
    # ==================================
    loss_cf = config.model.train.loss
    Loss = YoloV1Loss(
        S=config.model.S,
        B=config.model.B,
        C=config.model.C,
        coord_lambda=loss_cf.coord_lambda,
        noobj_lambda=loss_cf.noobj_lambda,
        device=device
    )


    # Optimzier
    # ==================================
    optimizer = torch.optim.Adam(
        Model.parameters(), 
        lr=config.model.train.lr, 
        weight_decay=config.model.train.weight_decay
    )
    """optimizer = torch.optim.SGD(
        Model.parameters(), 
        lr=config.model.train.lr, 
        momentum=config.model.train.momentum, 
        weight_decay=config.model.train.weight_decay)"""


    # Learning rate schedule
    # ==================================
    def lambda_lr(step):
        epoch = step // data_config.batch_size
        if epoch < 1:
            lbd = 10 * (step / data_config.batch_size)
        elif epoch < 75:
            lbd = 10
        elif epoch < 105:
            lbd = 1
        else:
            lbd = 0.1
        return lbd
    lr_scheduler = LambdaLR(optimizer, lambda_lr)


    # Train
    # ==================================
    log = {}
    mean_avg_prec = 0
    Model.train()
    for epoch in range(config.model.train.epochs):
        
        pred_boxes, target_boxes = get_bboxes(
            test_dataloader, Model, iou_threshold=0.5, threshold=0.4, device=device
        )            
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        print(f"Train mAP: {mean_avg_prec}")
        mean_loss = []
        pbar = tqdm(enumerate(train_dataloader), leave=True)
        for idx, batch in pbar:
            # ==> Data
            x, y = batch[0].to(device), batch[1].to(device)

            # ==> Forward 
            preds = Model(x)
            loss = Loss(preds, y)
            mean_loss.append(loss.item())

            # ==> Backward and update 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            pbar.set_description(
                f'Iteration {idx:05d}:' 
                + f'loss: {sum(mean_loss)/len(mean_loss)}'
            )


        # Wandb log
        # ==================================
        wandb.log(
            {
                "loss": sum(mean_loss)/len(mean_loss), 
                "mean_avg_prec": mean_avg_prec
            }
        )

        log[f'epoch {epoch}'] = sum(mean_loss)/len(mean_loss) 


    # Wandb model artifact
    # ==================================
    # ==> Save loss and model to local
    hydra.utils.instantiate(
        config.model.log,
        log_name=os.path.join('log', day, time),
        model=Model,
        loss=log
    )
    if config.model.wandb:
        artifact = wandb.Artifact(name=f"experiment_{day}_{time}_model", type="model")
        artifact.add_file(os.path.join('log', day, time, 'model.pt'))	
    

if __name__ == '__main__':
    main()