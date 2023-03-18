import torch.nn as nn

class LatentMapper(nn.Module):
    def __init__(self, dropout_rate=0.2, batch_size = 2):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048, track_running_stats=False if batch_size == 1 else True),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, track_running_stats=False if batch_size == 1 else True),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024, track_running_stats=False if batch_size == 1 else True),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024, track_running_stats=False if batch_size == 1 else True),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, track_running_stats=False if batch_size == 1 else True),
            nn.LeakyReLU(negative_slope=slope),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)
