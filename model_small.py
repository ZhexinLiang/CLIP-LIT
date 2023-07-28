import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_emb_oneBranch_symmetry_noreflect(nn.Module):

    def __init__(self, in_channels=3, out_channels=3,bias=False):
        super(UNet_emb_oneBranch_symmetry_noreflect, self).__init__()

        self.cond1 = nn.Conv2d(in_channels,32,3,1,1,bias=True) 
        self.cond_add1 = nn.Conv2d(32,out_channels,3,1,1,bias=True)           

        self.condx = nn.Conv2d(32,64,3,1,1,bias=True) 
        self.condy = nn.Conv2d(64,32,3,1,1,bias=True) 

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.ResidualBlock1=ResidualBlock(32,32)
        self.ResidualBlock2=ResidualBlock(32,32)
        self.ResidualBlock3=ResidualBlock(64,64)
        self.ResidualBlock4=ResidualBlock(64,64)
        self.ResidualBlock5=ResidualBlock(32,32)
        self.ResidualBlock6=ResidualBlock(32,32)

        self.PPM1 = PPM1(32,8,bins=(1,2,3,6))


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight.data)
                m.weight.data.normal_(0.0, 0.02)
                #nn.init.zeros_(m.bias.data)


    def forward(self, x):

        light_conv1=self.lrelu(self.cond1(x))
        res1=self.ResidualBlock1(light_conv1)
        
        res2=self.ResidualBlock2(res1)
        res2=self.PPM1(res2)
        res2=self.condx(res2)
        
        res3=self.ResidualBlock3(res2)
        res4=self.ResidualBlock4(res3)

        res4=self.condy(res4)
        res5=self.ResidualBlock5(res4)
        
        res6=self.ResidualBlock6(res5)
        
        light_map=self.relu(self.cond_add1(res6))
 
        return light_map

class UNet_emb_oneBranch_symmetry(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3,bias=False):
        super(UNet_emb_oneBranch_symmetry, self).__init__()

        self.cond1 = nn.Conv2d(in_channels,32,3,1,1,bias=True,padding_mode='reflect') 
        self.cond_add1 = nn.Conv2d(32,out_channels,3,1,1,bias=True,padding_mode='reflect')           

        self.condx = nn.Conv2d(32,64,3,1,1,bias=True,padding_mode='reflect') 
        self.condy = nn.Conv2d(64,32,3,1,1,bias=True,padding_mode='reflect') 

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.ResidualBlock1=ResidualBlock(32,32)
        self.ResidualBlock2=ResidualBlock(32,32)
        self.ResidualBlock3=ResidualBlock(64,64)
        self.ResidualBlock4=ResidualBlock(64,64)
        self.ResidualBlock5=ResidualBlock(32,32)
        self.ResidualBlock6=ResidualBlock(32,32)

        self.PPM1 = PPM1(32,8,bins=(1,2,3,6))


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight.data)
                m.weight.data.normal_(0.0, 0.02)
                #nn.init.zeros_(m.bias.data)


    def forward(self, x):
        
        light_conv1=self.lrelu(self.cond1(x))
        res1=self.ResidualBlock1(light_conv1)
        
        res2=self.ResidualBlock2(res1)
        res2=self.PPM1(res2)
        res2=self.condx(res2)
        
        res3=self.ResidualBlock3(res2)
        res4=self.ResidualBlock4(res3)
        res4=self.condy(res4)
        
        res5=self.ResidualBlock5(res4)
        res6=self.ResidualBlock6(res5)

        light_map=self.relu(self.cond_add1(res6))

        return light_map

class PPM1(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM1, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
                nn.Conv2d(in_dim+reduction_dim*4, in_dim, kernel_size=3, padding=1, bias=False),
                nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat       
          
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.lrelu(out)
        return out

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False,padding_mode='reflect')
