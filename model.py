import torch 
import torch.nn as nn
import hydra
import os
from omegaconf import OmegaConf 

def architecture(in_channels, fc_hidden_feature, fc_depth_feature):
    output = [
        # Stage 1
        {'_target_': 'torch.nn.Conv2d', 'in_channels': in_channels, 'out_channels': 64, 'kernel_size': 7, 'stride': 2, 'padding': 3},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.MaxPool2d', 'kernel_size': 2, 'stride': 2},

        # Stage 2
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 64, 'out_channels': 192, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.MaxPool2d', 'kernel_size': 2, 'stride': 2},

        # Stage 3
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 192, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.MaxPool2d', 'kernel_size': 2, 'stride': 2},

        # Stage 4
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        # 
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        # 
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        #
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
          
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.MaxPool2d', 'kernel_size': 2, 'stride': 2},

        # Stage 5
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        #
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},

        {'_target_': 'torch.nn.Conv2d', 'in_channels': 1024, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 1024, 'out_channels': 1024, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},

        # Stage 6
        {'_target_': 'torch.nn.Conv2d', 'in_channels': 1024, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},

        # Stage 7
        {'_target_': 'torch.nn.Flatten'},
        {'_target_': 'torch.nn.Linear', 'in_features': 7*7*1024, 'out_features': fc_hidden_feature},
        {'_target_': 'torch.nn.Dropout', 'p': 0.5},
        {'_target_': 'torch.nn.LeakyReLU', 'negative_slope': 0.1},
        {'_target_': 'torch.nn.Linear', 'in_features': fc_hidden_feature, 'out_features': 7*7*fc_depth_feature},
    ]

    return output


class YoloV1(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.depth = 5*config.B + config.C
        self.in_channels = config.in_channels

        self.architecture = architecture(
            self.in_channels,
            config.fc_hidden_feature,
            self.depth,
        )
        
        self.block = [
            hydra.utils.instantiate(
                self.architecture[0],
                in_channels=self.in_channels
            )                
        ]

        for layer in self.architecture[1:]:
            self.block.append(
                hydra.utils.instantiate(
                    layer
                )
            )

        self.block = nn.Sequential(*self.block)         

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
        # x: (Batch, Feature, Height, Width)

        output = self.block(x)
        return output