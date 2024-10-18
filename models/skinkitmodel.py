import torch.nn as nn
import torch.nn.functional as F

class SkinKitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(32,64,3),
            nn.Conv2d(64,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),                       
        )
        
        self.classifier=nn.Sequential(
            nn.Linear(53*53*32, 512),
            nn.ReLU(),
            nn.Dropout(0.5,inplace=False),
            nn.Linear(512, 256),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5,inplace=False),
            nn.Linear(128,11)       
        )
        
    def forward(self, data):
        output=self.features(data)
        # print(output.shape)
        output=output.view(output.shape[0], -1)
        return self.classifier(output)