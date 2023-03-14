import torch.nn as nn

class LatentMapper(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(2560, 2048),
            nn.InstanceNorm1d(2048),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048, 2048),
            nn.InstanceNorm1d(2048),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048,1024),
            nn.InstanceNorm1d(1024),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024,1024),
            nn.InstanceNorm1d(1024),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)
