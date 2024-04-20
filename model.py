import torch 
import torch.nn as nn
import hydra
import os
from omegaconf import OmegaConf 


class YoloV1(nn.Module):
    def __init__(self, S, B, C, features):
        super().__init__()

        self.S = S # Grid size
        self.B = B # Number of bounding box
        self.C = C # Number of class

        # Darknet model
        self.features = features

        # Yolo model
        self.con_layers = self.make_conv_layers()
        self.fc_layers = self.make_fc_layers()


    def make_conv_layers(self):
        conv = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        return conv
    
    def make_fc_layers(self):
        net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.S * self.S * 1024, 4086),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B*2 + self.C)),
            nn.Sigmoid()
        )

        return net


    @staticmethod
    def from_pretrained(path):
        dirlist = os.listdir(path)

        assert 'config.yaml' in dirlist, "Config file don't exists !!!"

        # Load config
        config = OmegaConf.load(os.path.join(path, 'config.yaml'))
        model = YoloV1(config.model)

        assert 'model.pt' in dirlist, "Weight file don't exists !!!"

        # Load weigth
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location=torch.device(device))
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        return model

    def forward(self, x):

        x = self.feature(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x