import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18, resnet34, resnet50

def load_resnet(resnet_type, pretrained=True):
    if resnet_type == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif resnet_type == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif resnet_type == 'resnet50':
        return resnet50(pretrained=pretrained)
    else:
        raise ValueError("resnet18, resnet34, and resnet50 are supported now.")

class CNN_Network(nn.Module):
    def __init__(self,args):
        super(CNN_Network,self).__init__()

        #####   Model   #####
        if 'efficientnet' in args.network:
            self.model = EfficientNet.from_pretrained(args.network)
            self.model._fc = nn.Linear(self.model._fc.in_features, 100)

        elif 'resnet' in args.network:
            self.model = load_resnet(args.network, pretrained=True)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout),
                nn.Linear(self.model.fc.in_features, 100)
            )
        elif "densenet" in args.network:
            self.model = model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 100)   
        else:
            raise ValueError("resnet and efficientnet are only supported now.")   

        self.fclayer1 = nn.Linear(100, 100)
        self.fclayer2 = nn.Linear(100, args.num_classes)

        
    def forward(self,img):
        fc = self.model(img)

        output = self.fclayer1(fc)
        output =  self.fclayer2(output)
            
        return output
    

if __name__ == '__main__':
    CNN_Network()