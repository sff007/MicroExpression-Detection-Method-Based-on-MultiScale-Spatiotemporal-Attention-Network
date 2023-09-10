import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class PatchEmbed(nn.Module):
    def __init__(self, img_height=192,img_width=160, patch_size=32, in_c=3, embed_dim=768, norm_layer=None,):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()



    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x1 = self.proj(x)
        x2 = x1.fflatten(2)
        x3 = x2.transpose(1, 2)
        x4 = self.norm(x3)
        return x4

class NonLocal(nn.Module):
    def __init__(self, channel):
        super(NonLocal, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1,0, bias=False)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1,0, bias=False)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上进行的
        x_phi = self.conv_phi(x).view(b, c, -1)
        # 获取theta特征，维度为[N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # 获取g特征，维度为[N, H * W, C/2]
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # 对phi和theta进行矩阵乘，[N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        # softmax拉到0~1之间
        mul_theta_phi = self.softmax(mul_theta_phi)
        # 与g特征进行矩阵乘运算，[N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # 1X1卷积扩充通道数
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x # 残差连接
        return out

class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)#（4,128）
        y = self.fc(y).view(b, c, 1, 1)#（4,128,1,1）
        return x * y.expand_as(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, sample_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 16
        last_channel = 128
        interverted_residual_setting = [
            # t, c, n, s
            [1, 24, 1, (1, 1, 1)],
            # [6, 24, 2, (2, 2, 2)],
            [6, 32, 1, (2, 2, 2)],
            # [6, 64, 1, (2, 2, 2)],
            [6, 96, 1, (1, 1, 1)],
            # [6, 160, 3, (2, 2, 2)],
            # [6, 320, 1, (1, 1, 1)],
        ]

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, (1, 2, 2))]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        self.selatt = NonLocal(channel=128)
        self.se = SEModel(channel=128)
        # self._initialize_weights()

    def forward(self, x):
        # (4,10,3,350,350)  B T C H W
        x = x.permute(0,2,1,3,4).contiguous() #(4,3,10,350,350)
        ze = torch.zeros_like(x)
        t0 = x[:, :, 0, :, :].unsqueeze(2)
        t1 = x[:, :, 1, :, :].unsqueeze(2)
        t2 = x[:, :, 2, :, :].unsqueeze(2)
        t3 = x[:, :, 3, :, :].unsqueeze(2)
        t4 = x[:, :, 4, :, :].unsqueeze(2)
        t5 = x[:, :, 5, :, :].unsqueeze(2)
        t6 = x[:, :, 6, :, :].unsqueeze(2)
        t7 = x[:, :, 7, :, :].unsqueeze(2)
        t8 = x[:, :, 8, :, :].unsqueeze(2)
        t9 = x[:, :, 9, :, :].unsqueeze(2)
        a1 = torch.cat([t0, t1],dim=2)
        a2 = torch.cat([t0, t1,t2,t3], dim=2)
        a3 = torch.cat([t0, t1,t2,t3,t4,t5], dim=2)
        a4 = torch.cat([t0, t1,t2,t3,t4,t5,t6,t7], dim=2)
        a5 = torch.cat([t0, t1,t2,t3,t4,t5,t6,t7,t8,t9], dim=2)

        x0 = self.features(a1) # (B,C,T,H,W)  (4,128,5,48,40)--->(4,5,128,48,40)
        x1 = self.features(a2)
        x2 = self.features(a3)
        x3 = self.features(a4)
        x4 = self.features(a5)

        x = torch.cat([x0,x1,x2,x3,x4],dim=2)

        linshi = x.permute(0,2,1,3,4).contiguous() #(4,5,128,48,40)
        map1 = linshi[:,0,:,:,:]
        att1 = self.selatt(map1)
        map2 = linshi[:,1,:,:,:]
        att2= self.selatt(map2)
        map3 = linshi[:,2,:,:,:]
        att3 = self.selatt(map3)
        map4 = linshi[:,3,:,:,:]
        att4 = self.selatt(map4)
        map5 = linshi[:,4,:,:,:]
        att5 = self.selatt(map5)
        nonLocalResult = torch.stack([att1, att2, att3, att4, att5], dim=1)#(B,T,C,H,W)
        nonlocalmap = nonLocalResult.permute(0, 2, 1, 3, 4).contiguous()#(B,C,T,H,W) (4,128,5,48,40)
        result = torch.mean(nonlocalmap, dim=2)
        seResult = self.se(result)#(4,128,48,40)
        seResult = seResult.unsqueeze(2)
        x2 = F.avg_pool3d(seResult, seResult.data.size()[-3:])#(4,128,1,1,1)
        x3 = x2.view(x2.size(0), -1)#(4,128)
        x4 = self.classifier(x3)
        return x4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNetV2(**kwargs)
    return model


if __name__ == "__main__":
    model = get_model(num_classes=2, sample_size=112, width_mult=1.)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    input_var = Variable(torch.randn(4, 10, 3, 192, 160))
    output = model(input_var)
    print(output)
    print(output.shape)

