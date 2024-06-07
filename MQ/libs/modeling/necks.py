import torch
from torch import nn
from torch.nn import functional as F

from .models import register_neck
from .blocks import MaskedConv1D, LayerNorm
from .utils import ACConv




@register_neck("fpn")
class FPN1D(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.ac_conv = ACConv(d_in=in_channels[0], d_out=in_channels[0])        # assume all dim same
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln))
            # use depthwise conv here for efficiency
            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, bias=(not with_ln), groups=out_channel
            )
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)
            

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        # for i in range(len(self.lateral_convs)):
        #     x, _ = self.lateral_convs[i](
        #         inputs[i + self.start_level], fpn_masks[i + self.start_level]
        #     )
        #     laterals.append(x)
        for i in range(len(self.lateral_convs)):
            if i == len(self.lateral_convs) - 1:
                x, _ = self.ac_conv(inputs[-1], fpn_masks[i + self.start_level])
            else:
                x, _ = self.lateral_convs[i](
                    inputs[i + self.start_level], fpn_masks[i + self.start_level]
                )
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i],
                scale_factor=self.scale_factor,
                mode='nearest'
            )

        # fpn conv / norm -> outputs
        # mask will remain the same
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(used_backbone_levels):
            x, new_mask = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x, )
            new_fpn_masks += (new_mask, )

        return fpn_feats, new_fpn_masks
        
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):

    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)
@register_neck('identity')
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True,      # if to apply layer norm at the end,
        use_us_fpn=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor
        self.use_us_fpn = use_us_fpn

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        if self.use_us_fpn:
            # self.conv = MaskedConv1D(
            #     in_channels[0], out_channel, 1, bias=(not with_ln))
            # self.us_conv = nn.ConvTranspose1d(in_channels[0], out_channel, 4, 2, 1, bias=False)
            self.lateral_linears = nn.ModuleList()
            self.mx_linears      = nn.ModuleList()
            for i in range(self.start_level, self.end_level):
                lateral_linear = nn.Linear(in_channels[i], out_channel)
                mx_linear      = nn.Linear(out_channel, out_channel)
                self.lateral_linears.append(lateral_linear)
                self.mx_linears.append(mx_linear)
            self.post_linear = nn.Linear(len(in_channels) * out_channel, out_channel)


        self.fpn_norms = nn.ModuleList()
        # self.trm_decoders = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

            # decoder_layer = nn.TransformerDecoderLayer(d_model=out_channel, nhead=8)
            # self.trm_decoders.append(nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=1))

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):

            x = inputs[i + self.start_level]

            # ========== add decoder ============
            # mask = fpn_masks[i + self.start_level]
            # trm_mask = ~mask.squeeze(1)
            # x = self.trm_decoders[i](tgt=x.permute(2,0,1), 
            #                          memory=x.permute(2,0,1),
            #                          tgt_key_padding_mask=trm_mask,
            #                          memory_key_padding_mask=trm_mask
            #                         )
            # x = x.permute(1,2,0)
            # ========== add decoder ============

            x = self.fpn_norms[i](x)
            fpn_feats += (x, )
            new_fpn_masks += (fpn_masks[i + self.start_level], )

        # add upsample feature
        # 1.upsample to one feature
        if self.use_us_fpn:
            # first_feat = inputs[self.start_level]
            # first_mask = fpn_masks[self.start_level]
            # x1, _ = self.conv(first_feat, first_mask)        # [b,c,t]
            # x1 = F.interpolate(
            #     x1,
            #     scale_factor=self.scale_factor,
            #     mode='nearest'
            # )                                               # [b,c,2t]
            # x2 = self.us_conv(first_feat)    # [b,c,2t]
            # first_mask_out = F.interpolate(
            #     first_mask.to(x.dtype),
            #     size=x2.shape[-1],
            #     mode='nearest'
            # )
            # x2 = x2 * first_mask_out.detach()
            # first_mask_out = first_mask_out.bool()
            # x = (x1 + x2) / 2.0
            # fpn_feats += (x, )
            # new_fpn_masks += (first_mask_out, )
        # 2. like ms-tct fused one feature
            
            laterals = []
            out_size = inputs[0].shape[-1] * 2
            
            last_lateral = self.lateral_linears[-1](inputs[-1].permute(0,2,1)).permute(0,2,1)          #[b,c,t]
            last_lateral = resize(last_lateral, size=(out_size), mode='linear', align_corners=False)
            laterals.append(last_lateral)
            for i, feat in enumerate(inputs):
                if i == len(inputs) -1:
                    break
                tmp = self.lateral_linears[i](feat.permute(0,2,1)).permute(0,2,1)
                tmp = resize(tmp, size=(out_size), mode='linear', align_corners=False)
                lateral = self.mx_linears[i](last_lateral.permute(0,2,1)).permute(0,2,1) + tmp
                laterals.append(lateral)
            concat_lateral = torch.cat(laterals, dim=1)
            concat_lateral = self.post_linear(concat_lateral.permute(0,2,1)).permute(0,2,1)

            first_mask = fpn_masks[self.start_level]
            first_mask_out = F.interpolate(
                first_mask.to(x.dtype),
                size=out_size,
                mode='nearest'
            )
            first_mask_out = first_mask_out.bool()

            fpn_feats += (concat_lateral, )
            new_fpn_masks += (first_mask_out, )

        # add upsample feature

        return fpn_feats, new_fpn_masks
