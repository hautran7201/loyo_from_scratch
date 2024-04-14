import os 
import hydra
import random
import torch
import torchvision.transforms as transforms
from utils import plot_image, cellboxes_to_boxes, non_max_suppression
from PIL import Image
from model import YoloV1
from omegaconf import DictConfig


@hydra.main(version_base=None)
def inference(config: DictConfig):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = YoloV1.from_pretrained(config.model_path).to(device)

    # Get data path
    image_file_names = random.sample(
        os.listdir(config.data.image_folder),
        config.data.sample_size
    )

    label_paths = []
    if 'label_folder' in config.data:
        label_file_names = [
            os.listdir(config.data.label_folder)
        ]
        file_names = [filename for filename in image_file_names if filename in label_file_names]
        label_paths = [
            os.path.join(config.data.label_folder, ) for path in file_names
        ]
    else:
        file_names = image_file_names

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)), 
            transforms.ToTensor()
        ]
    )

    # Load image
    for image_name in file_names:
        image_path = os.path.join(config.data.image_folder, image_name)
        image = Image.open(image_path)
        # boxes = torch.tensor(boxes)

        if transform:
            image = transform(image).unsqueeze(0).to(device)

        bboxes = cellboxes_to_boxes(model(image))
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        plot_image(image[0].permute(1,2,0).to("cpu"), bboxes, image_name=image_name)
        print('save sucessfull image')


if __name__ == '__main__':
    inference()