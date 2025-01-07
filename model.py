import torch
import torch.nn as nn
from layers import HybridPool2D
from layers import SwitchNorm2d
from layers import InceptionBlock_Real
from layers import HybridPool2DInception
from layers import ConvBlockIso
from layers import ConvBlock


class UNet_Real(nn.Module):
  def __init__(self):
    super(UNet_Real, self).__init__()
    self.SwitchNorm = SwitchNorm2d(3)

    self.Dropout = nn.Dropout(0.2)
    self.HybridPool_1 = HybridPool2D(return_shape=(64, 64), kernel_size=2,
                                    stride=2, padding="same")
    self.HybridPool_2 = HybridPool2D(return_shape=(32, 32), kernel_size=2,
                                     stride=2, padding="same")
    self.HybridPool_3 = HybridPool2D(return_shape=(16, 16), kernel_size=2,
                                     stride=2, padding="same")
    self.HybridPool_4 = HybridPool2D(return_shape=(8, 8), kernel_size=2,
                                     stride=2, padding="same")
    self.HybridPool_16 = HybridPool2D(return_shape=(8, 8), kernel_size=16,
                                      stride=16, padding="same")
    self.HybridPoolInception = HybridPool2DInception(return_shape=(8, 8), padding="valid",
                                                     output_shape=(128, 8, 8), kernel_size=2, stride=2)

    self.InceptionBlock1 = InceptionBlock_Real(3, 16, 128)
    self.InceptionBlock2 = InceptionBlock_Real(16, 32, 64)
    self.InceptionBlock3 = InceptionBlock_Real(32, 64, 32)
    self.InceptionBlock4 = InceptionBlock_Real(64, 128, 16)
    self.MiddleInception = InceptionBlock_Real(128, 256, 8)
    self.InceptionBlock5 = InceptionBlock_Real(256, 128, 16)
    self.InceptionBlock6 = InceptionBlock_Real(128, 64, 32)
    self.InceptionBlock7 = InceptionBlock_Real(64, 32, 64)
    self.InceptionBlock8 = InceptionBlock_Real(32, 16, 128)

    self.C2DT1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.C2DT2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.C2DT3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    self.C2DT4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

    self.ConvBlock1 = ConvBlockIso(51, 16, kernel_size=1, padding="same")
    self.ConvBlock2 = ConvBlockIso(112, 32, kernel_size=1, padding="same")
    self.ConvBlock3 = ConvBlockIso(224, 64, kernel_size=1, padding="same")
    self.ConvBlock4 = ConvBlockIso(448, 128, kernel_size=1, padding="same")

    self.ICB1 = ConvBlock(128, 256, kernel_size=1, padding=0)
    self.ICB2 = ConvBlock(128, 256, kernel_size=3, padding=1)
    self.ICB3 = ConvBlock(128, 256, kernel_size=5, padding=2)

    self.ConvBlock5 = ConvBlock(899, 256, kernel_size=1, padding="same")
    self.ConvBlock6 = ConvBlock(640, 128, kernel_size=1, padding="same")
    self.ConvBlock7 = ConvBlock(320, 64, kernel_size=1, padding="same")
    self.ConvBlock8 = ConvBlock(160, 32, kernel_size=1, padding="same")
    self.ConvBlock9 = ConvBlock(80, 16, kernel_size=1, padding="same")

    self.Conv2d = nn.Conv2d(16, 1, kernel_size=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input):

    intro = self.SwitchNorm(input)

    c2d_connection_1 = self.InceptionBlock1(intro)
    c2d_connection_1 = self.ConvBlock1(c2d_connection_1)
    op_1 = self.Dropout(c2d_connection_1)
    op_1 = self.HybridPool_1(op_1)

    c2d_connection_2 = self.InceptionBlock2(op_1)
    c2d_connection_2 = self.ConvBlock2(c2d_connection_2)
    op_2 = self.Dropout(c2d_connection_2)
    op_2 = self.HybridPool_2(op_2)

    c2d_connection_3 = self.InceptionBlock3(op_2)
    c2d_connection_3 = self.ConvBlock3(c2d_connection_3)
    op_3 = self.Dropout(c2d_connection_3)
    op_3 = self.HybridPool_3(op_3)

    c2d_connection_4 = self.InceptionBlock4(op_3)
    c2d_connection_4 = self.ConvBlock4(c2d_connection_4)
    op_4 = self.Dropout(c2d_connection_4)
    op_4 = self.HybridPool_4(op_4)

    conv_1 = self.ICB1(op_4)
    conv_2 = self.ICB2(op_4)
    conv_3 = self.ICB3(op_4)
    hybpl = self.HybridPoolInception(op_4)

    op_5 = torch.cat([conv_1, conv_2, conv_3, hybpl], dim=1)
    intro_2 = self.HybridPool_16(intro)
    op_5 = torch.cat([op_5, intro_2], dim=1)
    op_5 = self.ConvBlock5(op_5)

    op_6 = self.C2DT1(op_5)
    op_6 = torch.cat([op_6, c2d_connection_4], dim=1)
    op_6 = self.InceptionBlock5(op_6)
    op_6 = self.ConvBlock6(op_6)

    op_7 = self.Dropout(op_6)
    op_7 = self.C2DT2(op_7)
    op_7 = torch.cat([op_7, c2d_connection_3], dim=1)
    op_7 = self.InceptionBlock6(op_7)
    op_7 = self.ConvBlock7(op_7)

    op_8 = self.Dropout(op_7)
    op_8 = self.C2DT3(op_8)
    op_8 = torch.cat([op_8, c2d_connection_2], dim=1)
    op_8 = self.InceptionBlock7(op_8)
    op_8 = self.ConvBlock8(op_8)

    op_9 = self.Dropout(op_8)
    op_9 = self.C2DT4(op_9)
    op_9 = torch.cat([op_9, c2d_connection_1], dim=1)
    op_9 = self.InceptionBlock8(op_9)
    op_9 = self.ConvBlock9(op_9)

    final_out = self.Dropout(op_9)
    final_out = self.Conv2d(final_out)
    final_out = self.sigmoid(final_out)
    print(final_out.shape)

    return final_out