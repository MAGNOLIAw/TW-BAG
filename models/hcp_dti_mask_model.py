from .loss import *
from utils import *
from .networks import define_net
from .base_model import BaseModel

import torch
import nibabel as nib
import os

class HCPdtiMaskModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new options.
        """
        parser.add_argument('--lambda_tissue', type=float, default=1, help='weight for valid tissue L1 loss')
        parser.add_argument('--conv_type', type=str, default='unet')

        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        """
        Initialize this Inpaint class.
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['R']
        self.model_names = ['SR']
        self.opt = opt

        # define the inpainting network
        self.net_SR = define_net(self.opt.input_nc, opt.output_nc, opt.conv_type, opt.norm,
                                              self.opt.init_type, self.opt.init_gain, gpu_ids=self.opt.gpu_ids)

        if self.opt.isTrain:
            # define the loss functions
            self.tissue_loss = L1Loss(weight=opt.lambda_tissue)
            # define the optimizer
            self.optimizer_sr = torch.optim.Adam(self.net_SR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_sr)
        else:
            # define the loss functions
            self.tissue_loss = L1Loss(weight=opt.lambda_tissue)

    def set_input(self, input):
        """
        Read the data of input from dataloader then
        """
        if self.isTrain:
            self.brain = input['brain'].to(self.device, dtype=torch.float)  # masked brain
            self.mask = input['mask'].to(self.device, dtype=torch.float)
            self.gt = input['gt'].to(self.device, dtype=torch.float)  # original brain
            self.crop_region_mask = input['crop_region_mask'].to(self.device, dtype=torch.float)
        else:
            self.brain = input['brain'].to(self.device, dtype=torch.float)  # masked brain
            self.mask = input['mask'].to(self.device, dtype=torch.float)
            self.gt = input['gt'].to(self.device, dtype=torch.float)  # original brain
            self.crop_region_mask = input['crop_region_mask'].to(self.device, dtype=torch.float)


    def forward(self):
        """
        Run forward pass
        """
        self.input = torch.cat((self.brain, self.mask), dim=1)
        self.sr = self.net_SR(self.input)
        if not self.opt.isTrain:
            self.loss_R = self.tissue_loss(self.sr * self.mask, self.gt * self.mask)
        return self.sr

    def backward_sr(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss_R = self.tissue_loss(self.sr * self.mask, self.gt * self.mask)
        self.loss_R.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()

        # update optimizer of the inpainting network
        self.optimizer_sr.zero_grad()
        self.backward_sr()
        self.optimizer_sr.step()


