import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

__all__ = ['MHBNN', 'mhbnn'] #can only import MHBNN class and mhbnn function

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class MHBNN(nn.Module):

    def __init__(self, num_classes, num_local_features):
        super(MHBNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        #number of local features, d in original paper
        self.one_by_one_conv_output = num_local_features
        #nn.Parameter to register the parameter in model.parameters() to make them trainable
        #if Variable, not trainable
        self.lambdas = nn.Parameter(torch.ones(self.one_by_one_conv_output)*0.5, requires_grad=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.one_by_one_conv_output**2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.one_by_one_conv = nn.Conv2d(256, self.one_by_one_conv_output, kernel_size=1)
        
        
    
    def sign_sqrt(self, x):
        x = torch.mul(torch.sign(x),torch.sqrt(torch.abs(x)+1e-12))
        return x
    
    
    def bilinear_pooling(self, x):
        '''
        x = [x1, x2, ..., xn], size d*N, d is number of features of each patch, N is number of patches of all views 
        return: d*d
        '''
        return torch.mm(x, x.t())
    
    def harmonize(self, s):
        '''
        s: (d, ) tensor, sigular values
        return: (d, ) tensor after harmonized by box-cox transform
        '''
        harmonized_s = torch.zeros_like(s)
        n = s.size(0)
        for i in range(n):
            if torch.abs(self.lambdas[i]) > 1e-12:
                harmonized_s[i] = (s[i].pow(self.lambdas[i]) - 1) / self.lambdas[i]
            else:
                harmonized_s[i] = torch.log(s[i])
        return harmonized_s


    def forward(self, x):
        x = x.transpose(0, 1)
        
        view_pool = []
        
        for v in x:
            v = self.features(v)
            v = self.sign_sqrt(v) #early sqrt sublayer
            v = self.one_by_one_conv(v) #conv sublayer
            view_pool.append(v)
        y = torch.stack(view_pool)
        y = y.transpose(0, 1)
        res = []
        #each member in batch
        for b in y:
            #[num_local_features, views, w, h]
            b = b.transpose(0, 1).contiguous()
            #[num_local_features, views*w*h]
            b = b.view(b.size(0), -1)
            #[num_local_features, num_local_features]
            b = self.bilinear_pooling(b) #bininear pooling
            u, s, v = torch.svd(b)
            harmonized_s = self.harmonize(s) #harmonize singular values
            b = torch.mm(torch.mm(u, torch.diag(harmonized_s)), v.t())
            #[num_local_features*num_local_features]
            b = b.view(-1) #vectorized
            b = self.sign_sqrt(b) #late sqrt layer
            b = b / (torch.norm(b, 2)+(1e-8)) #l2 norm sub-layer
            res.append(b)

        pooled_view = torch.stack(res) #assembly into batch, [batch_size, num_local_features**2]
        
        pooled_view = self.classifier(pooled_view)
        return pooled_view


def mhbnn(pretrained=False, num_classes=40, num_local_features = 10):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MHBNN(num_classes, num_local_features)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model