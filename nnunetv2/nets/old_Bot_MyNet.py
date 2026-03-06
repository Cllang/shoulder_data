# 作者：Lang
# 2024年09月10日10时32分15秒
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch.cuda.amp import autocast

class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(_ConvINReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
    ):
        super().__init__()

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class AnisotropicConvBlock(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
    ):
        super().__init__()

        
        # self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1))
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        # self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class SDSBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(2, 2, 2)):
        super().__init__()

        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.pool1 = nn.AvgPool3d(kernel_size=(1, kernel_size[1], kernel_size[2]))
        self.pool2 = nn.AvgPool3d(kernel_size=(kernel_size[0], 1, kernel_size[2]))
        self.pool3 = nn.AvgPool3d(kernel_size=(kernel_size[0], kernel_size[1], 1))

        inter_channel = in_channel // 2

        self.trans_layer = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0)
        self.conv1_1 = _ConvIN3D(inter_channel, inter_channel, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.conv1_2 = _ConvIN3D(inter_channel, inter_channel, (1, 3, 1), stride=1, padding=(0, 1, 0))
        self.conv1_3 = _ConvIN3D(inter_channel, inter_channel, (1, 1, 3), stride=1, padding=(0, 0, 1))
        self.conv1_4 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1)

        self.conv2_1 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_2 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv3 = _ConvIN3D(inter_channel*2, inter_channel, 1, stride=1, padding=0)
        self.score_layer = nn.Conv3d(inter_channel, out_channel, 1, bias=False)

        self.mamba_layer = MambaLayer(dim=out_channel)

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.trans_layer(x)

        x1_1 = F.interpolate(self.pool1(self.conv1_1(x0)), size, mode='trilinear', align_corners=True)  # 跟图有点出入，这里先池化再上采样
        x1_2 = F.interpolate(self.pool2(self.conv1_2(x0)), size, mode='trilinear', align_corners=True)
        x1_3 = F.interpolate(self.pool3(self.conv1_3(x0)), size, mode='trilinear', align_corners=True)
        out1 = self.conv1_4(self.relu1(x1_1 + x1_2 + x1_3))

        x2_1 = self.conv2_1(x0)
        x2_2 = F.interpolate(self.conv2_2(x2_1), size, mode='trilinear', align_corners=True)
        out2 = self.relu2(x0 + x2_2)

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = self.mamba_layer(self.score_layer(self.relu3(x0 + out)))

        return out

class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, mode='nearest'):
        super(Upsample_block, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        self.pool_size = pool_size  # 其实是stride
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_size, mode=self.mode)
        x = self.conv(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                 use_1x1conv=True):
        super(encoder_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.stride = stride
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs

        self.conv1 = AnisotropicConvBlock(
            input_channels=self.in_channels,
            output_channels=self.out_channels,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            stride=self.stride,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=use_1x1conv
        )
        self.conv2 = BasicResBlock(  # 还没用
            conv_op=nn.Conv3d,
            input_channels=self.out_channels,
            output_channels=self.out_channels,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            stride=1,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=use_1x1conv
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        return self.conv2(conv1)
        # return conv1


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_op, norm_op_kwargs, kernel_size, stride, nonlin, nonlin_kwargs,
                 pool_size,
                 use_1x1conv=True):
        super(decoder_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.stride = stride
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.pool_size = pool_size
        self.kernel_size = kernel_size

        self.up = Upsample_block(self.in_channels, self.out_channels, pool_size=self.pool_size)  # mode='nearest'

        self.conv_up = BasicResBlock(
            conv_op=nn.Conv3d,
            input_channels=2 * self.out_channels,
            output_channels=self.out_channels,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            kernel_size=self.kernel_size,
            padding=1,
            stride=self.stride,
            use_1x1conv=use_1x1conv,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs
        )

    def forward(self, x, enc):
        x1 = self.up(x)
        x2 = torch.cat((x1, enc), dim=1)
        # deep_supervision
        x3 = self.conv_up(x2)
        return x3


class myNet(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=1,
            # feat_size=[32, 48, 96, 192, 384]
            feat_size=[16, 32, 64, 128, 256]
    ) -> None:
        super().__init__()
        self.in_channels = in_chans
        self.out_channels = out_chans
        self.stem = nn.Sequential(
            nn.Conv3d(self.in_channels, feat_size[0], kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.InstanceNorm3d(feat_size[0], eps=1e-5, affine=True)
        )
        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.nonlin = nn.LeakyReLU
        self.nonlin_kwargs = {'inplace': True}
        self.stride = [(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        self.pool_size = [(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        self.kernel_size = [(1, 3, 3), (1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]

        self.stem = AnisotropicConvBlock(
            input_channels=self.in_channels,
            output_channels=feat_size[0],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            stride=self.stride[0],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=True
        )

        self.encoder1 = encoder_block(
            in_channels=feat_size[0],
            out_channels=feat_size[1],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            stride=self.stride[1],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=True
        )
        self.encoder2 = encoder_block(
            in_channels=feat_size[1],
            out_channels=feat_size[2],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            stride=self.stride[2],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=True
        )
        self.encoder3 = encoder_block(
            in_channels=feat_size[2],
            out_channels=feat_size[3],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            stride=self.stride[3],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=True
        )
        self.encoder4 = encoder_block(
            in_channels=feat_size[3],
            out_channels=feat_size[4],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            stride=self.stride[4],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=True
        )
        self.encoder5 = encoder_block(
            in_channels=feat_size[4],
            out_channels=feat_size[5],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            stride=self.stride[5],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=True
        )
        self.encoder6 = encoder_block(
            in_channels=feat_size[5],
            out_channels=feat_size[6],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            stride=self.stride[6],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=True
        )
        self.sds_block = SDSBlock(
            in_channel=feat_size[5],
            out_channel=feat_size[5]
        )
        self.decoder6 = decoder_block(
            in_channels=feat_size[6],
            out_channels=feat_size[5],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            kernel_size=self.kernel_size[6],
            stride=self.stride[0],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            pool_size=self.pool_size[6],
            use_1x1conv=True
        )
        self.decoder5 = decoder_block(
            in_channels=feat_size[5],
            out_channels=feat_size[4],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            kernel_size=self.kernel_size[5],
            stride=self.stride[0],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            pool_size=self.pool_size[5],
            use_1x1conv=True
        )
        self.decoder4 = decoder_block(
            in_channels=feat_size[4],
            out_channels=feat_size[3],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            kernel_size=self.kernel_size[4],
            stride=self.stride[0],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            pool_size=self.pool_size[4],
            use_1x1conv=True
        )
        self.decoder3 = decoder_block(
            in_channels=feat_size[3],
            out_channels=feat_size[2],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            kernel_size=self.kernel_size[3],
            stride=self.stride[0],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            pool_size=self.pool_size[1],
            use_1x1conv=True
        )
        self.decoder2 = decoder_block(
            in_channels=feat_size[2],
            out_channels=feat_size[1],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            kernel_size=self.kernel_size[3],
            stride=self.stride[0],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            pool_size=self.pool_size[1],
            use_1x1conv=True
        )
        self.decoder1 = decoder_block(
            in_channels=feat_size[1],
            out_channels=feat_size[0],
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            kernel_size=self.kernel_size[3],
            stride=self.stride[0],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            pool_size=self.pool_size[1],
            use_1x1conv=True
        )

        self.out = BasicResBlock(
            conv_op=nn.Conv3d,
            input_channels=feat_size[0],
            output_channels=self.out_channels,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            # kernel_size=self.kernel_size[3],
            # stride=self.stride[0],
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            use_1x1conv=True
        )

    def forward(self, x):
        x1 = self.stem(x)
        encode1 = self.encoder1(x1)
        encode2 = self.encoder2(encode1)
        encode3 = self.encoder3(encode2)
        encode4 = self.encoder4(encode3)  # torch.Size([1, 256, 8, 32, 32])
        encode5 = self.encoder5(encode4)
        # encode6 = self.encoder6(encode5)
        bottle = self.sds_block(encode5)  # torch.Size([1, 256, 8, 32, 32])
        # context_block = self.context_block(encode4)
        # bottle = self.mamba_layer(context_block)
        # print(bottle.shape)
        # print(encode5.shape)
        # decode6 = self.decoder6(bottle,encode5)
        decode5 = self.decoder5(bottle,encode4)
        decode4 = self.decoder4(decode5, encode3)
        decode3 = self.decoder3(decode4, encode2)
        decode2 = self.decoder2(decode3, encode1)
        decode1 = self.decoder1(decode2, x1)
        out = self.out(decode1)
        return out


def load_pretrained_ckpt(
        model,
        ckpt_path="./data/pretrained/3dSAM_med/sam_med3d.pth"
):
    print(f"Loading weights from: {ckpt_path}")
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias",
                   "patch_embed.proj.weight", "patch_embed.proj.bias",
                   "patch_embed.norm.weight", "patch_embed.norm.weight"]

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_dict = model.state_dict()
    for k, v in ckpt['model'].items():
        if k in skip_params:
            print(f"Skipping weights: {k}")
            continue
        kr = f"vssm_encoder.{k}"
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        if kr in model_dict.keys():
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
            model_dict[kr] = v
        else:
            print(f"Passing weights: {k}")

    model.load_state_dict(model_dict)

    return model


def get_myNet_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = False,
        use_pretrain: bool = False
):
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = myNet(
        in_chans=num_input_channels,
        out_chans=label_manager.num_segmentation_heads,
        # feat_size=[48, 96, 192, 384],
        feat_size=[8, 16, 32, 64, 128, 256, 384]
    )

    if use_pretrain:
        model = load_pretrained_ckpt(model)

    return model


if __name__ == '__main__':
    from thop import profile

    model = myNet(
        in_chans=1,
        out_chans=1,
        # feat_size=[48, 96, 192, 384],
        feat_size=[8, 16, 32, 64, 128, 256, 384]
    ).cuda()
    model_dict = model.state_dict()

    input_image = torch.randn([1, 1, 12, 384, 384]).float().cuda()

    # ckpt = torch.load('/root/autodl-tmp/medical_resnet/resnet_18.pth',map_location='cpu')
    # for k, v in ckpt['state_dict'].items():
    #     print(k,v)
    #     break

    flops, params = profile(model, inputs=(input_image,))
    print("The flops is {} GB".format(flops / (1024 * 1024 * 1024)))
    print("The params is {} MB".format(params / (1024 * 1024)))


