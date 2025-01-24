import torch.nn as nn
import torch
from torchvision.models import resnet50, ResNet50_Weights

##Vanilla FCN Model
class FCN(nn.Module):

    def __init__(self, n_class):
        
        super().__init__()
        self.n_class = n_class
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)


    def forward(self, x):
    
        # Complete the forward function for the rest of the encoder
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1)))
        x3 = self.bnd3(self.relu(self.conv3(x2)))
        x4 = self.bnd4(self.relu(self.conv4(x3)))
        x5 = self.bnd5(self.relu(self.conv5(x4)))

        # Complete the forward function for the rest of the decoder
        y1 = self.bn1(self.relu(self.deconv1(x5)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))

        score = self.classifier(y5)

        return score  # size=(N, n_class, H, W)
    
# Architecture for Part 5a)
class Alternative(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnd = nn.BatchNorm2d(64)
        self.fully_connected = nn.Conv2d(64, 64, kernel_size=1)
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)
        
    def forward(self, x):
        x1 = self.fully_connected(self.bnd(self.relu(self.conv1(x))))
        x2 = self.fully_connected(self.bnd(self.relu(self.conv2(x1))))
        x3 = self.fully_connected(self.bnd(self.relu(self.conv2(x2))))
        x4 = self.fully_connected(self.bnd(self.relu(self.conv2(x3))))
        x5 = self.classifier(self.bnd(self.relu(self.conv2(x4))))

        return x5  # size=(N, n_class, H, W)

## Resnet Model
class Resnet(nn.Module):

    def __init__(self, n_class):
        
        super().__init__()
        self.n_class = n_class
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        
        self.relu = nn.ReLU(inplace=True)
        
        self.deconv1 = nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    #TODO Complete the forward pass
    def forward(self, x):
    
        x = self.encoder(x)

        # Complete the forward function for the rest of the decoder
        y1 = self.bn1(self.relu(self.deconv1(x)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))

        score = self.classifier(y5)

        return score  # size=(N, n_class, H, W)

## U-Net Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.doubleconv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.doubleconv(x)

class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.skipblock=ConvBlock(in_channels,in_channels)
        self.convblock=ConvBlock(in_channels,out_channels)
        self.maxpool=nn.MaxPool2d(2)
    
    def forward(self,x):
        s=self.skipblock(x)
        x=self.convblock(x)
        p=self.maxpool(x)
        return s,p

class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        self.convblock=ConvBlock(in_channels,out_channels)
    def forward(self,p,s):
        x=self.up(p)
        y=torch.cat([s,x],dim=1)
        return self.convblock(y)
       

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class     
        ##In Conv 
        self.inp=ConvBlock(3,64)
        ##Down blocks
        self.d1=DownBlock(64,128)
        self.d2=DownBlock(128,256)
        self.d3=DownBlock(256,512)
        self.d4=DownBlock(512,1024)
        ##Bridge 
        self.b=ConvBlock(1024,1024)
        ##Up blocks
        self.u1=UpBlock(1024,512)
        self.u2=UpBlock(512,256)
        self.u3=UpBlock(256,128)
        self.u4=UpBlock(128,64)
        ##Out conv   
        self.out=nn.Conv2d(64,self.n_class,kernel_size=1)

    def forward(self, x):
        x=self.inp(x)
        ## Down passes
        s1,p1=self.d1(x)
        s2,p2=self.d2(p1)
        s3,p3=self.d3(p2)
        s4,p4=self.d4(p3) 
        ##Bridge 
        b1=self.b(p4)
        ##Up passes 
        y1=self.u1(b1,s4) 
        y2=self.u2(y1,s3) 
        y3=self.u3(y2,s2) 
        y4=self.u4(y3,s1) 
        ##Output
        score = self.out(y4)
        return score  #size=(N, n_class, H, W)
