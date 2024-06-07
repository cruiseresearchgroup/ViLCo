import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm, SGPBlock)
from .modeling_xlnet_x import XLNetModel, XLNetConfig
from .utils import DeformConv1d

@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        use_xl,
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        t_c_alpha=0.8,
        scale_factor = 2,      # dowsampling rate for the branch,
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
        use_dcn = False,
        dcn_start_layer = 0,
        use_cross_modal = False,
        n_txt_in = 768,
    ):
        super().__init__()
        assert len(arch) == 3
        self.t_c_alpha = t_c_alpha
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.use_xl = use_xl
        self.use_cross_modal = use_cross_modal

        self.n_in = n_in 
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            if use_dcn and idx >= dcn_start_layer:
                self.embd.append(DeformConv1d(in_channels, n_embd, n_embd_ks,
                        stride=1, padding='same', bias=(not with_ln)))
            else:
                self.embd.append(MaskedConv1D(
                        in_channels, n_embd, n_embd_ks,
                        stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                    )
                )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    t_c_alpha = t_c_alpha,
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=self.use_cross_modal
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    t_c_alpha=t_c_alpha,
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=self.use_cross_modal
                )
            )

        # ======= using sgp layer
        # self.stem = nn.ModuleList()
        # sgp_mlp_dim = 1024
        # for idx in range(arch[1]):
        #     self.stem.append(SGPBlock(n_embd, 1, 1, n_hidden=sgp_mlp_dim, k=1.5, init_conv_vars=1))
        # self.branch = nn.ModuleList()
        # for idx in range(arch[2]):
        #     self.branch.append(SGPBlock(n_embd, 1, 2, k=1.5, path_pdrop=path_pdrop, 
        #                         n_hidden=sgp_mlp_dim, downsample_type="max", init_conv_vars=1))
        # ======= using sgp layer

        if self.use_xl:
            import json
            f = open('configs/xlnet_config_' + str(n_embd) + '.json')
            js = json.load(f)
            xlnet_config = XLNetConfig.from_dict(js)
            self.xlnet = XLNetModel(xlnet_config)
        # self.downsample_conv = MaskedConv1D(n_embd, n_embd, kernel_size=3, stride=2, padding=3 // 2, groups=n_embd, bias=False)
        
        # txt_embedding network using linear projection
        if self.use_cross_modal:
            self.txt_embd = nn.ModuleList()
            self.txt_embd_norm = nn.ModuleList()
            for idx in range(arch[0]):
                if idx == 0:
                    in_channels = n_txt_in
                else:
                    in_channels = n_embd
                self.txt_embd.append(MaskedConv1D(
                    in_channels, n_embd, 1,
                    stride=1, padding=0, bias=(not with_ln)
                )
                )
                if with_ln:
                    self.txt_embd_norm.append(
                        LayerNorm(n_embd)
                    )
                else:
                    self.txt_embd_norm.append(nn.Identity())
                    
            self.txt_stem = nn.ModuleList()
            for idx in range(arch[1]):
                self.txt_stem.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=False,
                )
                )
            
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask, src_text=None, src_text_mask=None, use_xl=False):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)

        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        B, C, T = x.size()
        
        use_lower_bound = False
        if use_lower_bound:
            x, mask = self.embd[0](x, mask)
            x = self.relu(self.embd_norm[0](x))
            src_query = None
            src_query_mask = None
            if self.use_cross_modal and src_text is not None:
                src_text, src_text_mask = self.txt_embd[0](src_text, src_text_mask)
                src_text = self.relu(self.txt_embd_norm[0](src_text))
            out_feats = tuple()
            out_masks = tuple()
            # 1x resolution
            out_feats += (x, )
            out_masks += (mask, )
            if not self.use_cross_modal:
                x, mask = self.branch[0](x, mask)
            else: 
                x, mask = self.branch[0](x, mask, cross_y=src_query, cross_y_mask=src_query_mask)
            out_feats += (x, )
            out_masks += (mask, )
            return out_feats, out_masks

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)
            
        # txt_embedding network
        src_query = None
        src_query_mask = None
        if self.use_cross_modal and src_text is not None:
            for idx in range(len(self.txt_embd)):
                src_text, src_text_mask = self.txt_embd[idx](src_text, src_text_mask)
                src_text = self.relu(self.txt_embd_norm[idx](src_text))

            src_query = src_text
            src_query_mask = src_text_mask

            # txt_stem transformer
            for idx in range(len(self.txt_stem)):
                src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)
            src_query_mask = src_query_mask.squeeze(1).long()

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x, )
        out_masks += (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            if self.use_xl:
                if idx == 0: 
                    # =================== xlnet ==========================#
                    x = x.permute(0,2,1)
                    # x = self.xlnet(inputs_embeds=x, attention_mask=mask.squeeze(1).long(), target_mapping=src_query, target_attention_mask=src_query_mask)[0]
                    x = self.xlnet(inputs_embeds=x, attention_mask=mask.squeeze(1).long())[0]
                    x = x.permute(0,2,1)
                    # =================== xlnet ==========================#
            else:   
                if idx == 0:
                    # x, mask = self.stem[idx](x, mask, cross_y=src_query, cross_y_mask=src_query_mask)
                    x, mask = self.stem[idx](x, mask)
                    
            if idx in [1, 2]:
                x, mask = self.branch[idx](x, mask)
            else: 
                x, mask = self.branch[idx](x, mask, cross_y=src_query, cross_y_mask=src_query_mask)         # [b,c,t] [b,1,t]

            out_feats += (x, )
            out_masks += (mask, )


        return out_feats, out_masks         # 6 layer of different scale (continuously downsample)



































@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch = (2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor = 2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                    in_channels, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x, )
        out_masks += (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks
