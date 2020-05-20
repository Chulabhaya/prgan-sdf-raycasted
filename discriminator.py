import torch
import torch.nn as nn


def discriminator_block(in_filters, out_filters, bn=False):
    # block = [   nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
    #            nn.LeakyReLU(0.1, inplace=True)]
    block = [
        nn.Conv2d(in_filters, out_filters, 3, 2, 1),
        nn.LeakyReLU(0.1, inplace=True),
    ]
    if bn:
        block.append(nn.BatchNorm2d(out_filters))

    return block


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, d_channels=64, batch_norm=False, img_size=16):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            *discriminator_block(in_channels, d_channels, bn=False),
            *discriminator_block(d_channels, d_channels * 2, bn=batch_norm),
            *discriminator_block(d_channels * 2, d_channels * 4, bn=batch_norm),
            #    *discriminator_block(d_channels*4, d_channels*8, bn=batch_norm),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 3
        self.adv_layer = nn.Sequential(nn.Linear(d_channels * 4 * ds_size ** 2, 1))
        # self.adv_layer = nn.Sequential(nn.Linear(d_channels*2*ds_size**2, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        logits = self.adv_layer(out)
        prob = self.sigmoid(logits)

        return logits, prob, out
