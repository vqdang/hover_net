
import torch
import torch.nn as nn
import torch.nn.functional as F

####
class Net(nn.Module):
    """ 
    A base class provides a common weight initialization scheme.
    """

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if 'norm' in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if 'linear' in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return x

