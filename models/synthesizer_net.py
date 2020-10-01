import torch
import torch.nn as nn
import torch.nn.functional as F


class SynthesizeOnlyAudio(nn.Module):
    def __init__(self, in_c, out_c):
        super(SynthesizeOnlyAudio, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        # self.conv2d = nn.Conv2d(self.in_c, self.out_c, kernel_size=3, padding=1)
        self.hidden_size = 24
        self.bilstm = nn.LSTM(self.in_c*256,hidden_size=self.hidden_size,num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.hidden_size, 600)
        self.fc1 = nn.Linear(600, 600)
        self.fc2 = nn.Linear(600, 600)
        self.masks = nn.Linear(600, self.out_c*256)


    def forward(self, feat_sound):
        (B, C, H, W) = feat_sound.size()
        x = feat_sound.permute(0,3,1,2).view(B, W,-1) #B, W, C*H
        x, _ = self.bilstm(x) # B, W, hidđen
        x = self.fc(x) # B, W, 600
        x = self.fc1(x) # B, W, 600
        x = self.fc2(x) # B, W, 600
        x = self.masks(x) # B, W, self.out_c*256

        masks = x.view(B, W, 2, H).permute(0,2,3,1)


        

        out = []
        for i in range(self.out_c):
            out.append(masks[:,i,:,:].unsqueeze(1))
        # print(out)
        return out
        


class InnerProd(nn.Module):
    def __init__(self, fc_dim):
        super(InnerProd, self).__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)) \
            .view(B, 1, *sound_size[2:])
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, C)
        z = (feat_img * self.scale).view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI*WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img * self.scale, feat_sound) \
            .view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z


class Bias(nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        # self.bias = nn.Parameter(-torch.ones(1))

    def forward(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img, feat_sound.view(B, C, H * W)).view(B, 1, H, W)
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        z = feat_img.view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI*WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img, feat_sound) \
            .view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z
