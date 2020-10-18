import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



"""


"""



class BotteneckResNet(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding=1,output_padding=1, transpose=False, is1d=False):
        super(BotteneckResNet, self).__init__()
        self.istransposeconv = transpose
        if is1d:
            self.conv_bn = nn.Sequential(
                                nn.Conv1d(inp, oup, kernel, stride, padding=padding),
                                nn.BatchNorm1d(oup),
                                nn.ReLU(inplace=True))
        else:

            self.conv_bn = nn.Sequential(
                                nn.Conv2d(inp, oup, kernel, stride, padding=padding),
                                nn.BatchNorm2d(oup),
                                nn.ReLU(inplace=True))    
        
        if is1d:
            self.transpose_conv_bn = nn.Sequential(
                                        nn.ConvTranspose1d(inp, oup, kernel, stride, padding=padding, output_padding=1),
                                        nn.BatchNorm1d(oup),
                                        nn.ReLU(inplace=True))
        else:
            self.transpose_conv_bn = nn.Sequential(
                                        nn.ConvTranspose2d(inp, oup, kernel, stride, padding=padding),
                                        nn.BatchNorm2d(oup),
                                        nn.ReLU(inplace=True))

                                        
        self.relu = nn.ReLU()

    
    def forward(self, x):
        if not self.istransposeconv:
            # print(f'X before : {x.shape}')
            x1 = self.conv_bn(x)
            # print(f'X1 before : {x1.shape}')

            x = self.relu(x+x1)
        else:
            x = self.transpose_conv_bn(x)

        return x


class AudioMixtureResNet1D(nn.Module):
    def __init__(self, n_channel_inp=80,  n_channel_fc0=1536, n_channel_conv=1536, n_channel_fc6=256):
        super(AudioMixtureResNet1D, self).__init__()

        self.fc0 = nn.Linear(n_channel_inp, n_channel_fc0)
        self.conv1_5 = self._make_conv_layers(n_channel_conv, n_channel_conv, k=5, s=1, p=2, n=5)
        self.fc6 = nn.Linear(n_channel_conv, n_channel_fc6)
        

    
    def _make_conv_layers(self, in_c=1536, out_c=1536, k=5, s=1, p=2, n=5):
        layers = []
        for i in range(n):
            layers.append(BotteneckResNet(in_c, out_c, k, s, p, transpose=False, is1d=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0,2,1)  # Batch x time x n_channel
        x = self.fc0(x).permute(0,2,1)  # Batch x n_channel x time
        # print(f'Shape after fc0: ', x.shape)

        x = self.conv1_5(x)
        # print(f'Shape after conv1_5: ', x.shape)

        # fc6
        x = x.permute(0,2,1)
        x = self.fc6(x).permute(0,2,1)
        # print(f'Shape after fc6: ', x.shape)

        return x


class ResnetDilated1D(nn.Module):
    def __init__(self, orig_resnet, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ResnetDilated1D, self).__init__()
        from functools import partial

        self.pool_type = pool_type

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])

        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size//2)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        x = self.features(x)

        if not pool:
            return x
        
        # print(f'Shape after before pooling : {x.shape}')


        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        # print(f'Shape after pooling : {x.shape}')
        x = x.view(B, T, 512).permute(0,2,1) # Batch x 512 xT
        return x


class VideoResNet1D(nn.Module):
    def __init__(self, n_channel_inp=512,  n_channel_fc0=1536, n_channel_conv=1536, n_channel_fc10=256):
        super(VideoResNet1D, self).__init__()

        self.fc0 = nn.Linear(n_channel_inp, n_channel_fc0)
        self.conv1_2 = self.__make_conv_layers(n_channel_fc0, n_channel_conv, k=5, s=1, padding=2, n=2)
        self.conv3 = BotteneckResNet(n_channel_conv, n_channel_conv, kernel=5, stride=2, padding=2, output_padding=1,transpose=True, is1d=True)
        self.conv4_6 = self.__make_conv_layers(n_channel_fc0, n_channel_conv,k=5,s=1, padding=2, n=3)
        self.conv7 = BotteneckResNet(n_channel_conv, n_channel_conv, kernel=5, stride=2, padding=2, output_padding=1, transpose=True, is1d=True)
        self.conv8_9 = self.__make_conv_layers(n_channel_fc0, n_channel_conv,k=5,s=1,padding=2, n=2)
        self.fc10 = nn.Linear(n_channel_conv, n_channel_fc10)


    def __make_conv_layers(self, in_c, out_c, k=5, s=1, padding=2, n=2):
        layers = []
        for i in range(n):
            layers.append(BotteneckResNet(in_c, out_c, kernel=k, stride=s, padding=padding, transpose=False, is1d=True))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.fc0(x).permute(0,2,1)
        # print(f"shape x after fc0: {x.shape}")
        
        x = self.conv1_2(x)
        # print(f"shape x after conv1_2: {x.shape}")

        x = self.conv3(x)
        # print(f"shape x after conv3 {x.shape}")

        x = self.conv4_6(x)
        # print(f"shape x after conv4_6 {x.shape}")


        x = self.conv7(x)
        # print(f"shape x after conv7 {x.shape}")

        x = self.conv8_9(x)
        x = x.permute(0,2,1)
        # print(f"shape x after conv8_9 {x.shape}")


        x = self.fc10(x)
        x = x.permute(0,2,1)
        # print(f"shape x after fc10 {x.shape}")

        return x


class AVFusion1D(nn.Module):
    def __init__(self, n_channel_inp=512,  F=80):
        super(AVFusion1D, self).__init__()
        self.bilstm = nn.LSTM(n_channel_inp, hidden_size=200, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(400, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc_mask = nn.Linear(600, F)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x,_ = self.bilstm(x)
        # print(f'shape after bilstm :{x.shape}')
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_mask(x)
        x = x.permute(0,2,1)

        return x



if __name__=="__main__":
    T = 11

    # Audio model 
    # x = torch.rand((2, 80, 4*T))
    # audiomodel = AudioMixtureResNet1D(n_channel_inp=80,  n_channel_fc0=1536, n_channel_conv=1536, n_channel_fc6=256)
    # x = audiomodel(x)
    # print("Shape output audio: ", x.shape)

    # Audio model 2D
    # x = torch.rand((2, 1, 4*T, 80))
    # audiomodel = AudioMixtureResNet2D(n_channel_inp=80,  n_channel_fc0=1536, n_channel_conv=1536, n_channel_fc6=256)
    # x = audiomodel(x)
    # print("Shape output: ", x.shape)


    # Visual model
    # v = torch.rand((2, 512, T))
    # 
    # v = videomodel(v)
    # print("Shape output video: ", v.shape)

    # Fusion model
    # f = torch.rand((2, 512, 300))
    # fucsion = AVFusion1D(512, 80)
    # f = fucsion(f)
    # print("Shape output other: ", f.shape)

    # Video models
    original_resnet = torchvision.models.resnet18(pretrained=True)
    resnet_dilated = ResnetDilated1D(original_resnet, dilate_scale=8)
    videomodel = VideoResNet1D(n_channel_inp=512,  n_channel_fc0=1536, n_channel_conv=1536, n_channel_fc10=256)
    
    # Audio model
    audiomodel = AudioMixtureResNet1D(n_channel_inp=256,  n_channel_fc0=1536, n_channel_conv=1536, n_channel_fc6=256)
    
    # Fusion model
    fucsion = AVFusion1D(512, 256)


    # Inference
    v = torch.rand((2, 3, T, 224,224))  # B x n_channels x T x H x W
    v = resnet_dilated.forward_multiframe(v)  # B x 512 x T
    print(f"Video Dilated: {v.shape}")
    v = videomodel(v)  # B x 256 x 4T
    print(f'Video sampling : {v.shape}')

    a = torch.rand((2, 256, 4*T))
    a = audiomodel(a)  # B x 256 x 4T
    print(f'Audio sampling : {a.shape}')
    
    v_a = torch.cat((v, a), dim=1)
    print(f'Fusion input: {v_a.shape}')

    v_a = fucsion(v_a)
    print(f'Fusion output: {v_a.shape}')






