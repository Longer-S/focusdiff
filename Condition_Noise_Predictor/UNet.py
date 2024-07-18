from torch import nn
import numpy as np
from abc import abstractmethod
import sys
import os
import torch # 现在可以进行绝对导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '.', '.')))
from __init__ import time_embedding
from __init__ import Downsample
from __init__ import Upsample
from torch.nn import functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.misc import (NestedTensor, nested_tensor_from_tensor)
import ltr.models.neck.position_encoding as pos_encoding
import ltr.models.neck.featurefusion_network as feature_network
import ltr.admin.settings as ws_settings
from util.cbam import CBAM
import math
# use GN for norm layer
def group_norm(channels):
    # print(channels)
    return nn.GroupNorm(4, channels)


# 包含 time_embedding 的 block
class TimeBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        函数不能为空，但可以添加注释
        """


class TimeSequential(nn.Sequential, TimeBlock):
    def forward(self, x, emb):
        for layer in self:
            # 判断该 layer 中是否包含 time_embedding
            if isinstance(layer, TimeBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                group_norm(features),
                nn.SiLU()
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                group_norm(d),
                                nn.SiLU())
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
 
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        return feats_V
class FFParser(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x
    



class LearnableGaussianBlur(nn.Module):
    def __init__(self,dim=3,h=128,w=128,kernel_size=3, initial_sigma=1.0):
        super(LearnableGaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.tensor(initial_sigma))
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, dtype=torch.float32) * 0.02)

    def forward(self, x):
        kernel = self.create_gaussian_kernel(self.kernel_size, self.sigma)
        kernel = kernel.expand(x.size(1), 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=x.size(1))
        weight = self.complex_weight
        return x*weight

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        device = sigma.device
        x = torch.arange(kernel_size, dtype=torch.float32).to(device) - kernel_size // 2
        x = x.repeat(kernel_size, 1)
        y = x.t()
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)
    

# ResBlock 继承 TimeBlock
# 所有的 ResBlock 中均包含 time_embedding，其他 layer 不包含 time_embedding
class ResBlock(TimeBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        
        h = self.conv1(x)
        h = h.view(t.shape[0], -1, *h.shape[-3:])    
        B, FS, C, H, W = h.shape
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None].unsqueeze(1) 
        h = h.view(B*FS, C, H, W) # BxFS C H W
        h = self.conv2(h)
        return h + self.shortcut(x)

class NoiseEmbeddingModule(nn.Module):
    def __init__(self, in_channels,out_channels,time_channels):
        super(NoiseEmbeddingModule, self).__init__()
        self.noise_embedding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),        
            nn.GroupNorm(4, out_channels),
            nn.SiLU()
        )
        # print(out_channels)
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
    
    def forward(self, x,t):
        return self.noise_embedding(x)+self.time_emb(t)[:, :, None, None]
    
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        # print(qkv.shape)
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)

        # print(q.shape)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        # print(print(h.shape))
        h = h.reshape(B, -1, H, W)
        # print("s",print(h.shape))
        h = self.proj(h)
        return h + x

class Fusion(nn.Sequential):

    def __init__(self, input):
        super(Fusion, self).__init__()
        self.convA = nn.Sequential(
            nn.Conv2d(input*2, input, kernel_size=1, stride=1, padding=1),
            nn.GroupNorm(4,input),
            nn.ReLU(True),
        )

    def forward(self, x, concat_with):
        # up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convA(torch.cat([x, concat_with], dim=1))


class NoisePred(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 model_channels,
                 num_res_blocks,
                 dropout,
                 time_embed_dim_mult,
                 down_sample_mult,
                 ):
        super().__init__()
        self.setting = ws_settings.Settings()
        self.in_channels = in_channels              #3
        self.out_channels = out_channels            #3
        self.model_channels = model_channels        #16采样通道数
        self.num_res_blocks = num_res_blocks        #2
        self.dropout = dropout
        self.down_sample_mult = down_sample_mult    #[1, 2, 4, 8]

        # time embedding
        time_embed_dim = model_channels * time_embed_dim_mult  #16*4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 下采样和上采样的通道数
        down_channels = [model_channels * i for i in down_sample_mult]
        up_channels = down_channels[::-1]

        # 每个块中 ResBlock 的数量
        downBlock_chanNum = [num_res_blocks + 1] * (len(down_sample_mult) - 1)
        downBlock_chanNum.append(num_res_blocks)
        upBlock_chanNum = downBlock_chanNum[::-1]
        self.downBlock_chanNum_cumsum = np.cumsum(downBlock_chanNum)
        self.upBlock_chanNum_cumsum = np.cumsum(upBlock_chanNum)[:-1]

        # 初始卷积层
        self.inBlock = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # 下采样
        # 共四个才采样块，每个下采样块包括两个包含 time_embed 的 ResBlock 和一个不包含 time_embed 的 DownSample 块
        self.downBlock = nn.ModuleList()
        down_init_channel = model_channels
        for level, channel in enumerate(down_channels):
            for i in range(num_res_blocks):
                layer1 = ResBlock(in_channels=down_init_channel,
                                  out_channels=channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout)
                if i!=num_res_blocks-1:
                    down_init_channel = channel
                else:
                    down_init_channel = channel*2
                self.downBlock.append(TimeSequential(layer1))
            # 最后一步不做下采样
            if level != len(down_sample_mult) - 1:
                down_layer = Downsample(channels=channel)
                self.downBlock.append(TimeSequential(down_layer))

        # middle block
        self.middleBlock = nn.ModuleList()
        for i in range(num_res_blocks):
            layer2 = ResBlock(in_channels=down_channels[-1]*2,
                              out_channels=down_channels[-1]*2,
                              time_channels=time_embed_dim,
                              dropout=dropout)
            self.middleBlock.append(TimeSequential(layer2))
            if i==num_res_blocks-1:
                down_channelsBlock=nn.Conv2d(down_channels[-1]*2, down_channels[-1], kernel_size=1, stride=1, padding=0)
                self.middleBlock.append(TimeSequential(down_channelsBlock))
        # 上采样
        # 共四个上采样块，每个上采样块包括两个包含 time_embed 的 ResBlock 和一个不包含 time_embed 的 DownSample 块
        self.upBlock = nn.ModuleList()
        up_init_channel = down_channels[-1]
        for level, channel in enumerate(up_channels):
            if level == len(up_channels) - 1:
                out_channel = model_channels
            else:
                out_channel = channel // 2
            for _ in range(num_res_blocks):
                layer3 = ResBlock(in_channels=up_init_channel,
                                  out_channels=out_channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout)
                up_init_channel = out_channel
                if i!=num_res_blocks-1:
                    down_init_channel = channel
                else:
                    down_init_channel = channel*2

                self.upBlock.append(TimeSequential(layer3))
            if level > 0:
                up_layer = Upsample(channels=out_channel)
                self.upBlock.append(TimeSequential(up_layer))

        # out block
        self.outBlock = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
        self.noiseEmbedding=nn.ModuleList([NoiseEmbeddingModule(3, model_channels,time_embed_dim)] + [NoiseEmbeddingModule(model_channels * (2**(i-1)), model_channels * (2**i),time_embed_dim) for i in range(1, len(down_sample_mult))])

        self.pos_encoding5 = pos_encoding.PositionEmbeddingSine(num_pos_feats=self.setting.hidden_dim//2)
        self.featurefusion_network5 = feature_network.build_featurefusion_network(self.setting)
        # self.input_proj5 = nn.Conv2d(self.input_channel, self.hidden_dim, kernel_size=1)
        # self.fc_weights = nn.ModuleList([nn.Linear(C, 1) for C in [4,2]])
        # print("a")

        self.fusion = nn.ModuleList()
        self.noise_downBlock = nn.ModuleList()
        self.ffparser=nn.ModuleList()
        self.gaussian_Blur=nn.ModuleList()

        self.noise_downBlock.append(TimeSequential(nn.Conv2d(1, model_channels, kernel_size=3, padding=1)))            

        down_init_channel = model_channels
        self.out_nosie = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0)
        ds = 1
        for level, channel in enumerate(down_channels):
            for i in range(num_res_blocks):
                layer1 = [ResBlock(in_channels=down_init_channel,
                                  out_channels=channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout)]
                # if i!=num_res_blocks-1:
                #     down_init_channel = channel
                # else:
                #     down_init_channel = channel*2
                down_init_channel = channel
                if ds in [8,16]:
                    layer1.append(AttentionBlock(down_init_channel, num_heads=4))
                self.noise_downBlock.append(TimeSequential(*layer1))
            # self.gaussian_Blur.append(LearnableGaussianBlur(channel, 256 // (2 **(level+1)), 256 // (2 **(level+1)),kernel_size=3, initial_sigma=1.0))
            self.ffparser.append(FFParser(channel, 256*2 // (2 **(level+1)), 256*2 // (2 **(level+2))+1))
            self.fusion.append(Fusion(channel))
            # 最后一步不做下采样
            if level != len(down_sample_mult) - 1:
                down_layer = Downsample(channels=channel)
                self.noise_downBlock.append(TimeSequential(down_layer))
                ds *= 2
        # self.gaussian_Blur=nn.ModuleList(self.gaussian_Blur)
        
        self.sknet=SKConv(features=256,M=2, G=32, r=16, stride=1, L=32)

    
    def forward(self, x: torch.Tensor, y: torch.Tensor, timesteps):

        # y_e = y[:, 0:1, :, :].unsqueeze(1).expand(-1, x.shape[1],-1, -1, -1)
        # x = torch.cat((x, y_e), dim=2)
        B, FS, C, H, W = x.shape
        n_x = x.view(B*FS, C, H, W) # BxFS C H W


        embedding = time_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(embedding)


        n1 = y
        h_n1=[]
        num_down = 1
        for down_block in self.noise_downBlock:
            n1 = down_block(n1, time_emb)
            if num_down in [3,6,9,12]:
                # h_n1.append(self.gaussian_Blur[num_down//3-1](n1))
                h_n1.append(self.ffparser[num_down//3-1](n1))
            num_down += 1          

        # 用于存放每个下采样步骤的输出
        res = []

        # in stage
        x = self.inBlock(n_x)

        # down stage
        h = x
        num_down = 1
        for down_block in self.downBlock:
            h = down_block(h, time_emb)
            if num_down in self.downBlock_chanNum_cumsum:

                w_h = h.view(B, FS, *h.shape[-3:]) # B FS C H/2**i W/2**i
                res.append(torch.mean(w_h, dim=1))
                # res.append(torch.max(w_h, dim=1)[0]) # B C H/2**i W/2**i
                stack_pool = h.view(B, FS, *h.shape[-3:])    
                sum_pool_max = torch.mean(stack_pool, dim=1).unsqueeze(1).expand_as(stack_pool).contiguous().view(B*FS, *h.shape[-3:])
                # sum_pool_max  = torch.max(stack_pool, dim=1)[0].unsqueeze(1).expand_as(stack_pool).contiguous().view(B*FS, *h.shape[-3:])
                h = torch.cat([h, sum_pool_max], dim=1)

            #add
            elif(num_down in [1,4,7,10]):
                th = h.view(B, FS, *h.shape[-3:])   
                # new=torch.cat([h, h_n1.pop(0).unsqueeze(1).expand_as(th).contiguous().view(B*FS, *h.shape[-3:])], dim=1)
                # h=h+h_n1.pop(0).unsqueeze(1).expand_as(th).contiguous().view(B*FS, *h.shape[-3:])
                self.fusion[num_down//3](h,h_n1.pop(0).unsqueeze(1).expand_as(th).contiguous().view(B*FS, *h.shape[-3:]))
            num_down += 1

        # middle stage
        for middle_block in self.middleBlock:
            h = middle_block(h, time_emb)
        w_h = h.view(B, FS, *h.shape[-3:]) # B FS C H W
        h = torch.mean(w_h, dim=1)
        # h = torch.max(w_h, dim=1)[0]
        h = h + res.pop()
        # assert len(res) == len(self.upBlock_chanNum_cumsum)
        

        h=self.sknet(h)


        # n=y
        # h_n=[]
        # for i, l in enumerate(self.noiseEmbedding):
        #     n = self.noiseEmbedding[i](n,time_emb) 
        #     if(i!=len(self.noiseEmbedding)-1):
        #         n = F.max_pool2d(n, kernel_size=2, stride=2, padding=0)
        #     h_n.append(n)

        



        # self.rgb_d = CBAM(out_dim, 1)
        # todo 改进通道注意力 CBAM 筛选权重前百分之最大 y用传统算法的全焦图像去引导去噪 F.softmax(cost,1).size() 
        # h = torch.cat([h, h], dim=1)


        # # 5th layer's transformer 16 * 16
        # if not isinstance(n1, NestedTensor):
        #     noisy_f_nest = nested_tensor_from_tensor(n1)
        # if not isinstance(h, NestedTensor):
        #     h_nest = nested_tensor_from_tensor(h)

        # feat_ir5, mask_ir5 = noisy_f_nest.decompose()
        # feat_vis5, mask_vis5 = h_nest.decompose()

        # pos_ir5 = self.pos_encoding5(noisy_f_nest)
        # pos_vis5 = self.pos_encoding5(h_nest)

        # hs5 = self.featurefusion_network5(feat_ir5, mask_ir5, feat_vis5, mask_vis5, pos_ir5, pos_vis5)

        # h = hs5.contiguous().view(int(self.setting.batch_size), feat_ir5.shape[2],
        #                                     feat_ir5.shape[3], -1).permute(0, 3, 2, 1)  #推理的时候self.setting.batch_size设置为1

        




        # up stage

        num_up = 1
        for up_block in self.upBlock:
            # 对于非2的幂次方的img_size，残差连接时会存在下采样和上采样尺寸不一致的现象
            # 以 res.pop()为标准，对 h进行裁剪
            if num_up in self.upBlock_chanNum_cumsum:  # [2,5,8]
                h = up_block(h, time_emb)
                h_crop = h[:, :, :res[-1].shape[2], :res[-1].shape[3]]
                # h = h_crop + res.pop()+F.interpolate(h_n1.pop(-2), size=[h_crop.size(2), h_crop.size(3)], mode='bilinear', align_corners=True)
                h = h_crop + res.pop()
            else:
                h = up_block(h, time_emb)
            num_up += 1
        assert len(res) == 0
        n=self.out_nosie(y)
        # n=F.interpolate(h_n[0], size=[h.size(2), h.size(3)], mode='bilinear', align_corners=True)
        h=n+h
        # out stage
        out = self.outBlock(h)#可以在把噪声加上
        return out
    
if __name__ == '__main__':

    net=NoisePred(3,3,32,2,0.1,4,[1, 2, 4, 8]) #不变 只需该batchsize
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(net(torch.randn(2,2,3,128,128),torch.randn(2,1,128,128),torch.randint(0, 100, (2,)).long()))