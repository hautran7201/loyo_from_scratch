import torch
import hydra
import json
import os 
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from logging import NullHandler
from tqdm import tqdm


from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR

from dataset import VOCDataset
from omegaconf import DictConfig
from model import YoloV1
from raw_model import Yolov1
from loss import YoloV1Loss
from utils import (cellboxes_to_boxes, non_max_suppression, plot_image, get_bboxes, mean_average_precision)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    # Device
    device = config.model.train.device

    # Transforms
    """transform = transforms.Compose(
        [
            transforms.Resize((448, 448)), 
            transforms.ToTensor()
        ]
    )"""
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

    if config.model.do.training:
        print('--> Start training\n')

        # Dataset
        data_config = config.model.train.dataset
        train_dataset = hydra.utils.instantiate(
            data_config.train_data,
            transform=transform
        )
        test_dataset = hydra.utils.instantiate(
            data_config.test_data,
            transform=transform
        )
        subset_sampler = SubsetRandomSampler(range(500))
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

        # Model 
        # Model = YoloV1(config.model).to(device)
        Model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)

        # Loss
        Loss = YoloV1Loss(config.model.train.loss)

        # Optimizer 
        optimizer = hydra.utils.instantiate(
            config.model.train.optimizer,
            params = Model.parameters()
        )

        # Learning rate schedule
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
        scheduler = LambdaLR(optimizer, lambda_lr)

        # Train 
        log = {}
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
                x, y = batch[0].to(device), batch[1].to(device)

                preds = Model(x)
                loss = Loss(preds, y)
                mean_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                pbar.set_description(
                    f'Epoch {epoch:03d}'
                    + f'Iteration {idx:05d}:' 
                    + f'loss: {sum(mean_loss)/len(mean_loss)}'
                )

            log[f'epoch {epoch}'] = sum(mean_loss)/len(mean_loss) 

        # Save loss and model
        hydra.utils.instantiate(
            config.model.log,
            model=Model,
            loss=log
        )

    if config.model.do.evaluating:
        # Dataset
        data_config = config.model.evaluate.dataset
        test_dataset = hydra.utils.instantiate(
            data_config.test_data,
            transform=transform
        )
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=data_config.shuffle,
            batch_size=data_config.batch_size,
            pin_memory=data_config.pin_memory,
            drop_last=data_config.drop_last
        )

        # Load model
        Model = YoloV1.from_pretrained(config.model.evaluate.pretrained_path)

        # Device
        device = config.model.train.device

        # Evaluate
        pred_boxes, target_boxes = get_bboxes(
                test_dataloader, Model, iou_threshold=0.5, threshold=0.4, device=device
        )            
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        print(f"Train mAP: {mean_avg_prec}")


if __name__ == '__main__':
    main()