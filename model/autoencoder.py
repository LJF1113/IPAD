import torch
from .reconstruction_model import Reconstruction3DEncoder, Reconstruction3DDecoder

class convAE(torch.nn.Module):
    def __init__(self):  # for reconstruction
        super(convAE, self).__init__()

        self.reconstruction = True

        # self.encoder = Reconstruction3DEncoder(chnum_in=1)  # black and white
        # self.decoder = Reconstruction3DDecoder(chnum_in=1)  # black and white
        self.encoder = Reconstruction3DEncoder(chnum_in=3)  # RGB
        self.decoder = Reconstruction3DDecoder(chnum_in=3)  # RGB

    def forward(self, x):
        # print(x.shape)
        fea = self.encoder(x)
        # print(fea.shape)
        output = self.decoder(fea.clone())
        # print(output.shape)

        return output


