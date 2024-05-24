import torch
from .reconstruction_model import Reconstruction3DEncoder, Reconstruction3DDecoder, VST3DDecoder
from .VST_block import SwinTransformer3D
from einops import rearrange
from model import MemModule
import torch.nn as nn
from torch.nn import functional as F

class VST(torch.nn.Module):
    def __init__(self, mem_dim=2000, shrink_thres=0.0025):  # for reconstruction
        super(VST, self).__init__()
        self.reconstruction = True
        # self.chnum_in = chnum_in

        # self.encoder = Reconstruction3DEncoder(chnum_in=1)  # black and white
        # self.decoder = Reconstruction3DDecoder(chnum_in=1)  # black and white
        self.transformer_encoder = SwinTransformer3D()
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=768, shrink_thres=shrink_thres)
        self.period = nn.Sequential(
            nn.Conv3d(768, 768, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(768),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size,256,4,4,4)
            nn.Flatten(1),
            nn.Linear(768*4*4*4,4096),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,200),
        )
        self.transformer_decoder = VST3DDecoder(chnum_out=3)
        # self.encoder = Reconstruction3DEncoder(chnum_in=3)  # RGB
        # self.decoder = Reconstruction3DDecoder(chnum_in=3)  # RGB

    def forward(self, x):
        
        feature = self.transformer_encoder(x)
        #feature (batch_size,768,4,8,8)
        recon_index = self.period(feature)
        # print(recon_index[0])
        res_mem = self.mem_rep(feature, recon_index)
        feature = res_mem['output']
        att = res_mem['att']
        output = self.transformer_decoder(feature.clone())

        return {'output': output, 'att': att, 'recon_index': recon_index}


