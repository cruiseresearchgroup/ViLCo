import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D, LayerNorm)

import os
import pdb

import timm
import yaml
import numpy as np
from transformers import AutoModel

from abc import abstractmethod

from . import heads
from . import roberta
from .roberta import RobertaModel, _prepare_decoder_attention_mask
from .heads import *
from transformers import RobertaConfig
from functools import partial
import copy
import torch.distributed as dist

with open('./libs/modeling/EgoNCE_MLM_ITM_Config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
        
def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):   # this
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp: # this
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict


# class BaseModel(nn.Module):
#     """
#     Base class for all models
#     """
#     @abstractmethod
#     def forward(self, *inputs):
#         """
#         Forward pass logic

#         :return: Model output
#         """
#         raise NotImplementedError

#     def __str__(self):
#         """
#         Model prints with number of trainable parameters
#         """
#         model_parameters = filter(lambda p: p.requires_grad, self.parameters())
#         params = sum(np.prod(p.size()) for p in model_parameters)
#         return super().__str__() + '\nTrainable parameters: {}'.format(params)


# class FrozenInTime(BaseModel):
#     def __init__(self,
#                  video_params,
#                  text_params,
#                  projection_dim=4096,
#                  load_checkpoint=None,
#                  projection='minimal',
#                  load_temporal_fix='zeros', #bilinear
#                  config = config,
#                  task_names = None, #'EgoNCE_ITM_MLM',
#                  norm_layer = None,
#                  embed_dim=768):
#         super().__init__()

#         self.video_params = video_params
#         self.text_params = text_params
#         self.load_temporal_fix = load_temporal_fix
#         self.config = config
#         self.task_names = task_names
#         if not text_params['pretrained']:
#             raise NotImplementedError("Huggingface text models require pretrained init.")

#         if self.text_params['model'].startswith('roberta'):
#             self.text_model = RobertaModel.from_pretrained("roberta-base")
#         self.text_model.train()

#         pretrained = video_params['pretrained']
#         if video_params['model'] == "SpaceTimeTransformer":
#             self.num_frames = video_params['num_frames'] 
#             time_init = 'zeros'
#             attention_style = 'frozen-in-time'
#             arch_config = 'base_patch16_224'
#             vit_init = 'imagenet-21k'
#             if arch_config == 'base_patch16_224':
#                 vit_model = torch.load("./ckpt/pretrained/EgoVLPv2.pth", map_location="cpu")
#                 print("pre-trained model loaded")
#                 model = SpaceTimeTransformer(num_frames=self.num_frames,
#                                             time_init=time_init,
#                                             attention_style=attention_style)
#             else:
#                 raise NotImplementedError

#             model.head = nn.Identity()
#             model.pre_logits = nn.Identity()
#             ftr_dim = model.embed_dim
           
#             vit_checkpoint = vit_model
#             new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
#             model.load_state_dict(new_vit_dict, strict=False)
#             self.video_model = model
#         else:
#             raise NotImplementedError(f"{video_params['model']} not implemented")

#         # for backwards compatibility (old models)
#         self.video_model.fc = nn.Identity()

#         # Project to a common embedding
#         if projection == 'minimal':

#             txt_proj = nn.Sequential(
#                 nn.Linear(self.text_model.config.hidden_size, projection_dim, bias=False),
#                 nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
#                 nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
#                 )

#             vid_proj = nn.Sequential(
#                 nn.Linear(ftr_dim, projection_dim, bias=False),
#                 nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
#                 nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
#                 )
        
#         elif projection == '':
#             txt_proj = nn.Identity()
#             vid_proj = nn.Identity()
#         else:
#             raise NotImplementedError
#         self.txt_proj = txt_proj
#         self.vid_proj = vid_proj

#         if ('MLM' in self.task_names or 'ITM' in self.task_names):
#             # for FIBER-like cross-attention

#             bert_config = RobertaConfig(
#                 vocab_size=self.config["vocab_size"],
#                 hidden_size=self.config["hidden_size"],
#                 num_hidden_layers=self.config["num_layers"],
#                 num_attention_heads=self.config["num_heads"],
#                 intermediate_size=self.config["hidden_size"] * config["mlp_ratio"],
#                 #max_position_embeddings=maxlen, [was used in BTGOT script]
#                 hidden_dropout_prob=self.config["drop_rate"],
#                 attention_probs_dropout_prob=self.config["drop_rate"],
#             )

#             self.num_fuse_block=self.config["num_fuse_block"]
#             self.num_text_layer=self.config["num_layers"]
#             roberta.NUM_FUSE_BLOCK = self.video_model.NUM_FUSE_BLOCK=self.num_fuse_block
#             roberta.DIM_IMG=self.config["input_image_embed_size"]
#             self.video_model.DIM_TXT=self.config["input_text_embed_size"]

#             self.cross_modal_text_transform = nn.Linear(self.config["input_text_embed_size"], self.config["hidden_size"])
#             self.cross_modal_text_transform.apply(init_weights)
#             self.cross_modal_video_transform = nn.Linear(self.config["input_image_embed_size"], self.config["hidden_size"])
#             self.cross_modal_video_transform.apply(init_weights)

#             self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

#             self.num_patches = self.video_model.patch_embed.num_patches
#             self.patches_per_frame = self.num_patches//self.num_frames
#             norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#             self.norm = norm_layer(embed_dim)
#             self.pre_logits = nn.Identity()

#             self.avgpool = nn.AdaptiveAvgPool1d(1)
#             self.cross_modal_video_pooler = heads.Pooler(config["hidden_size"])
#             self.cross_modal_video_pooler.apply(init_weights)
#             self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
#             self.cross_modal_text_pooler.apply(init_weights)

#             ## einops transformations
#             self.einops_from_space = 'b (f n) d'
#             self.einops_to_space = '(b f) n d'
#             self.einops_from_time = 'b (f n) d'
#             self.einops_to_time = '(b n) f d'

#         if 'MLM' in self.task_names:
#             self.mlm_score = heads.MLMHead(bert_config)
#             self.mlm_score.apply(init_weights)

#         if 'ITM' in self.task_names:
#             self.itm_score = heads.ITMHead(config["hidden_size"] * 2)
#             self.itm_score.apply(init_weights)

#         if load_checkpoint not in ["", None]:
#             checkpoint = torch.load(load_checkpoint, map_location='cpu')
#             state_dict = checkpoint['state_dict']
#             new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
#             new_state_dict = self._inflate_positional_embeds(new_state_dict)
#             self.load_state_dict(new_state_dict, strict=True)
#             print("Model is loaded with pre-trained parameters")

#     def set_device(self, device):
#         self.device = device

#     def forward(self, data, video_only=False, return_embeds=True):
        
#         text_data = data['text']
#         video_data = data['video']

#         b, curr_frames, channels, _, _ = video_data.shape
#         video_data_itm = self.video_model.patch_embed(video_data)
#         video_data_itm = video_data_itm.flatten(2).transpose(2, 1)
#         video_data_itm = video_data_itm.reshape(b, -1, self.video_model.patch_embed.embed_dim)
        
#         BF = video_data_itm.shape[0]
#         cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         video_data_itm = torch.cat((cls_tokens, video_data_itm), dim=1)
#         # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
#         cls_embed = self.video_model.pos_embed[:, 0, :].unsqueeze(1)
#         tile_pos_embed = self.video_model.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
#         # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
#         tile_temporal_embed = self.video_model.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
#         total_pos_embed = tile_pos_embed + tile_temporal_embed
#         total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

#         n = self.patches_per_frame
#         f = curr_frames

#         curr_patches = video_data_itm.shape[1]
#         video_data_itm = video_data_itm + total_pos_embed[:, :curr_patches]
#         video_data_itm = self.video_model.pos_drop(video_data_itm)

#         unfused_blocks = self.num_text_layer - self.num_fuse_block
            

#         for blk_i, blk in enumerate(self.video_model.blocks[:unfused_blocks]):
#             if config['use_checkpoint']:
#                 video_data_itm = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
#                                 n, f, use_reentrant=False)
#             else:
#                 video_data_itm = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
#                                 time_n=n, space_f=f)                      
            
#         text_embeds = self.text_model.embeddings(input_ids=text_data['input_ids']) # before it was input_ids=text_ids
#         device = text_embeds.device
#         text_masks = text_data['attention_mask']
#         input_shape = text_masks.size()
#         extend_text_masks = self.text_model.get_extended_attention_mask(text_masks, input_shape, device)
#         for layer_i, layer in enumerate(self.text_model.encoder.layer[:unfused_blocks]):
#             if config['use_checkpoint']:
#                 text_embeds = torch.utils.checkpoint.checkpoint(layer, text_embeds, extend_text_masks, use_reentrant=False)[0]
#             else:
#                 text_embeds = layer(text_embeds, extend_text_masks)[0]

#         for blk_i, blk in enumerate(self.video_model.blocks[unfused_blocks:self.num_text_layer]):
#             if config['use_checkpoint']:
                    
#                 fuse_video_data = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, 
#                                           n, f, text_embeds, extend_text_masks, use_reentrant=False)
#                 text_embeds = torch.utils.checkpoint.checkpoint(self.text_model.encoder.layer[blk_i + unfused_blocks],
#                                           text_embeds, extend_text_masks, None, (video_data_itm), None, None, False, True, use_reentrant=False)[0]
#             else:
#                 fuse_video_data = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, 
#                                           y=text_embeds, y_mask=extend_text_masks, time_n=n, space_f=f)
#                 text_embeds = self.text_model.encoder.layer[blk_i + unfused_blocks](text_embeds, extend_text_masks, encoder_hidden_states=(video_data_itm), last_norm=True)[0]
#             video_data_itm = fuse_video_data

            
#         video_data_cls = self.norm(video_data_itm)[:, 0]
#         video_data_cls = self.pre_logits(video_data_cls)
#         video_data_ft = self.norm(video_data_itm)[:, 1:]
#         video_data_ft = video_data_ft.reshape((b, curr_frames, -1, video_data_ft.shape[-1]))
#         video_data_ft = torch.sum(video_data_ft, dim=2) / self.patches_per_frame
#         video_data_ft = self.pre_logits(video_data_ft)
#         video_data_ft = video_data_ft.transpose(2, 1)

#         return video_data_cls, video_data_ft
    
#     def compute_text(self, text_data):
#         if self.text_params['model'].startswith('bert'):
#             text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
#                 'pooler_output']
#         elif self.text_params['model'].startswith('distilbert'):
#             text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
#         elif self.text_params['model'].startswith('roberta'):
#             text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
#         else:
#             raise NotImplementedError
#         text_embeddings = self.txt_proj(text_embeddings)
#         return text_embeddings

#     def compute_text_tokens(self, text_data, is_proj=True):
#         if self.text_params['model'].startswith('bert'):
#             text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
#                 'pooler_output']    # not implement for bert
#         elif self.text_params['model'].startswith('distilbert'):
#             text_embeddings = self.text_model(**text_data).last_hidden_state
#         elif self.text_params['model'].startswith('roberta'):
#             text_embeddings = self.text_model(**text_data).last_hidden_state
#         else:
#             raise NotImplementedError
        
#         if is_proj:
#             text_embeddings = self.txt_proj(text_embeddings)
        
#         return text_embeddings

#     def compute_video(self, video_data):
#         video_embeddings = self.video_model(video_data)
#         video_embeddings = self.vid_proj(video_embeddings)
#         return video_embeddings

#     def _inflate_positional_embeds(self, new_state_dict):
#         # allow loading of timesformer with fewer num_frames
#         curr_keys = list(self.state_dict().keys())
#         if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
#             load_temporal_embed = new_state_dict['video_model.temporal_embed']
#             load_num_frames = load_temporal_embed.shape[1]
#             curr_num_frames = self.video_params['num_frames']
#             embed_dim = load_temporal_embed.shape[2]

#             if load_num_frames != curr_num_frames:
#                 if load_num_frames > curr_num_frames:
#                     print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
#                 else:
#                     print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     if self.load_temporal_fix == 'zeros':
#                         new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
#                         new_temporal_embed[:, :load_num_frames] = load_temporal_embed
#                     elif self.load_temporal_fix in ['interp', 'bilinear']:
#                         # interpolate
#                         # unsqueeze so pytorch thinks its an image
#                         mode = 'nearest'
#                         if self.load_temporal_fix == 'bilinear':
#                             mode = 'bilinear'
#                         load_temporal_embed = load_temporal_embed.unsqueeze(0)
#                         new_temporal_embed = F.interpolate(load_temporal_embed,
#                                                            (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
#                     else:
#                         raise NotImplementedError
#                 new_state_dict['video_model.temporal_embed'] = new_temporal_embed
#         # allow loading with smaller spatial patches. assumes custom border crop, to append the
#         # border patches to the input sequence
#         if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
#             load_pos_embed = new_state_dict['video_model.pos_embed']
#             load_num_patches = load_pos_embed.shape[1]
#             curr_pos_embed = self.state_dict()['video_model.pos_embed']
#             if load_num_patches != curr_pos_embed.shape[1]:
#                 raise NotImplementedError(
#                     'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

#         return new_state_dict


# def sim_matrix(a, b, eps=1e-8):
#     """
#     added eps for numerical stability
#     """
#     a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
#     a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
#     b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
#     sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
#     return sim_mt


# def sim_matrix_batch_val(a, b, eps=1e-8):
#     """
#     added eps for numerical stability
#     """
#     a_n, b_n = a.norm(dim=-1).unsqueeze(-1), b.norm(dim=-1).unsqueeze(-1)
#     a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
#     b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
#     sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
#     return sim_mt


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,
            n_vid_in,  # input video feature dimension
            n_txt_in,  # input text feature dimension
            n_embd,  # embedding dimension (after convolution)
            n_head,  # number of head for self-attention in transformers
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch=(2, 2, 2, 0, 5),  # (#convs, #stem transformers, #branch transformers)
            mha_win_size=[-1] * 6,  # size of local window for mha
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=False,  # if to attach layernorm after conv
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # droput rate for drop path
            use_abs_pe=False,  # use absolute position embedding
            use_rel_pe=False,  # use relative position embedding
            use_adapter=False,
    ):
        super().__init__()
        assert len(arch) == 5
        assert len(mha_win_size) == (1 + arch[3] + arch[4])
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # vid_embedding network using convs
        self.vid_embd = nn.ModuleList()
        self.vid_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_vid_in
            else:
                in_channels = n_embd
            self.vid_embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.vid_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.vid_embd_norm.append(nn.Identity())

        # txt_embedding network using linear projection
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

        # stem network using (vanilla) transformer
        self.vid_stem = nn.ModuleList()
        self.use_adapter = use_adapter
        for idx in range(arch[2]):
            self.vid_stem.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
                use_adapter=self.use_adapter,
            )
            )

        self.txt_stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.txt_stem.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=-1,
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=False,
            )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[3]):
            self.branch.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(self.scale_factor, self.scale_factor),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[1 + idx],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
        )

        for idx in range(arch[4]):
            self.branch.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(self.scale_factor, self.scale_factor),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[1 + idx],
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

    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()

        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)

        # vid_stem transformer
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks