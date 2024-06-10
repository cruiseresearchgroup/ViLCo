import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms
from torch.nn.init import normal_, constant_
from ..cl_methods import Prompt
from timm.utils.model_ema import ModelEmaV2

import random

def normal_distribution(x, mu=0, sigma=1):
    return (-(x - mu)**2 / (2 * sigma**2)).exp()

def multiple_normal_distribution(x, mu, sigma):
    return ((-(x - mu[:1])**2 / (2 * sigma[:1]**2)).exp() + (-(x - mu[1:])**2 / (2 * sigma[1:]**2)).exp()) / 2.0

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))
        
    def forward(self, x):
        return self.alpha * x + self.beta
    
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())
        
class MemoryBank:
    def __init__(self, size, feature_dim):
        self.size = size
        self.feature_dim = feature_dim
        self.memory = torch.randn(size, feature_dim).cuda()
        self.ptr = 0

    @torch.no_grad()
    def update(self, features):
        batch_size = features.size(0)
        assert batch_size <= self.size, "Batch size must be less than or equal to memory bank size"
        
        if self.ptr + batch_size <= self.size:
            self.memory[self.ptr:self.ptr + batch_size] = features
            self.ptr += batch_size
        else:
            overflow = (self.ptr + batch_size) - self.size
            self.memory[self.ptr:] = features[:self.size - self.ptr]
            self.memory[:overflow] = features[self.size - self.ptr:]
            self.ptr = overflow

    def get_all(self):
        return self.memory
    
class Conv2dAdapter(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels = None,
        down_sample = 5,
        mode = "before",  # enum before, after, parallel
        scale = None,
        act_layer = nn.GELU,
    ):
        super().__init__()
        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"

        hidden_dim = down_sample
        if isinstance(down_sample, float):
            hidden_dim = int(in_channels * down_sample)

        if out_channels is None:
            out_channels = in_channels

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            act_layer(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            Scaler(scale),
        )
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.layer[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.layer[0].bias)
        nn.init.zeros_(self.layer[2].weight)
        nn.init.zeros_(self.layer[2].bias)

    def forward(self, module, input, **kwargs):
        if self.mode == "before":
            return module(self.layer(input) + input, **kwargs)
        if self.mode == "after":
            return self.layer(module(input, **kwargs)) + input
        return module(input, **kwargs) + self.layer(input)

class Adapter(nn.Module):
    def __init__(
        self,
        embed_dim,
        down_sample = 5,
        mode = "parallel",  # enum before, after, parallel
        scale = None,
        act_layer = nn.GELU,
        stride = 1,
    ):
        super().__init__()

        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"

        # hidden_dim = down_sample
        # if isinstance(down_sample, float):
        hidden_dim = int(embed_dim * down_sample)

        self.layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, embed_dim // 2),
            # Scaler(scale),
        )
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.layer[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.layer[0].bias)
        nn.init.zeros_(self.layer[2].weight)
        nn.init.zeros_(self.layer[2].bias)

    def forward(self, module, input, mask, **kwargs):
        if self.mode == "before":
            return module(self.layer(input) + input, **kwargs)
        if self.mode == "after":
            return self.layer(module(input, **kwargs)) + input
        out, out_mask = module(input, mask, **kwargs)
        adapt_out = self.layer(input)
        adapt_out = adapt_out
        out = out + adapt_out
        return out, out_mask

class Scaler(nn.Module):
    def __init__(self, scale = None):
        super().__init__()

        if scale is None:
            self.register_parameter("scale", nn.Parameter(torch.tensor(1.0)))
        else:
            self.scale = scale

    def forward(self, input):
        return input * self.scale

    def extra_repr(self):
        learnable = isinstance(self.scale, nn.Parameter)
        return f"scale={self.scale:.4f}, learnable={learnable}"

def freeze(module, *submodules):
    if submodules:
        module = nn.ModuleList(
            [m for n, m in module.named_modules() if n in submodules]
        )
    for param in module.parameters():
        param.requires_grad_(False)
        param.grad = None

def unfreeze(module, *submodules):
    if submodules:
        module = nn.ModuleList(
            [m for n, m in module.named_modules() if n in submodules]
        )
    for param in module.parameters():
        param.requires_grad_(True)

class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            num_classes,
            prior_prob=0.01,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            empty_cls=[]
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights associated with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)
                
        self.reg_params = {}

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits,)

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )
        self.reg_params = {}

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)

        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            backbone_type,  # a string defines which backbone we use
            fpn_type,  # a string defines which fpn we use
            backbone_arch,  # a tuple defines # layers in embed / stem / branch
            scale_factor,  # scale factor between branch layers
            input_vid_dim,  # input video feat dim
            input_txt_dim,  # input text feat dim
            max_seq_len,  # max sequence length (used for training)
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            n_head,  # number of heads for self-attention in transformer
            n_mha_win_size,  # window size for self attention; -1 to use full seq
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            fpn_dim,  # feature dim on FPN
            fpn_with_ln,  # if to apply layer norm at the end of fpn
            fpn_start_level,  # start level of fpn
            head_dim,  # feature dim for head
            regression_range,  # regression range on each level of FPN
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            head_with_ln,  # attach layernorm to reg/cls heads
            use_abs_pe,  # if to use abs position encoding
            use_rel_pe,  # if to use rel position encoding
            num_classes,  # number of action classes
            train_cfg,  # other cfg for training
            test_cfg,  # other cfg for testing
            cl_cfg,
    ):
        super().__init__()
        self.input_txt_dim = input_txt_dim
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor ** i for i in range(
            fpn_start_level, backbone_arch[-2] + backbone_arch[-1] + 1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range), (self.fpn_strides, self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-2] + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-2] + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size

        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len %d must be divisible by fpn stride and window size %d" % (
                max_seq_len, stride)
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']
        use_adapter = cl_cfg['use_adapter']

        # backbone network: conv + transformer
        assert backbone_type == 'convTransformer'
        self.backbone = make_backbone(
            'convTransformer',
            **{
                'n_vid_in': input_vid_dim,
                'n_txt_in': input_txt_dim,
                'n_embd': embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch': backbone_arch,
                'mha_win_size': self.mha_win_size,
                'scale_factor': scale_factor,
                'with_ln': embd_with_ln,
                'attn_pdrop': 0.0,
                'proj_pdrop': self.train_dropout,
                'path_pdrop': self.train_droppath,
                'use_abs_pe': use_abs_pe,
                'use_rel_pe': use_rel_pe,
                'use_adapter': use_adapter,
            }
        )
        
        # video_params = {"model": "SpaceTimeTransformer", "arch_config": "base_patch16_224", "num_frames": 897, "pretrained": True, "time_init": "zeros"}
        # text_params = {"model": "roberta-base", "pretrained": True, "input": "text"}
        # projection_dim=4096
        # load_checkpoint=""
        # projection='minimal'
        # load_temporal_fix='bilinear'
        # task_names = 'EgoNCE_ITM_MLM'
        # norm_layer = None
        # embed_dim=768
        # self.encoder = FrozenInTime(video_params, text_params, 
        #                             projection_dim=projection_dim, load_checkpoint=load_checkpoint,
        #                             projection=projection, load_temporal_fix=load_temporal_fix,
        #                             task_names = task_names, norm_layer = norm_layer, embed_dim=embed_dim).cuda()

        # fpn network: identity
        assert fpn_type == 'identity'
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim] * (backbone_arch[-2] + backbone_arch[-1] + 1),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'start_level': fpn_start_level,
                'with_ln': fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': max_seq_len * max_buffer_len_factor,
                'fpn_strides': self.fpn_strides,
                'regression_range': self.reg_range
            }
        )

        # classification and regression heads
        assert self.num_classes > 0
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9
        self.reg_params = {}
        
        # icarl
        self.compute_means = True if cl_cfg['name'] == 'icarl' else False
        self.exemplar_means = []
        self.memory = {}
        self.adv_lambda = cl_cfg['adv_lambda']
        self.type_sampling = cl_cfg['type_sampling']
        self.n_known = 0
        self.dist_loss = nn.BCEWithLogitsLoss()
        
        # bic
        self.list_bias_layers = []
        self.list_splits = []
        
        self.cl_name = cl_cfg['name']
        
        # l2p
        prompt_pool = cl_cfg['prompt_pool']
        prompt_length = cl_cfg['length']
        top_k = cl_cfg['topk']
        pool_size = cl_cfg['pool_size']
        if prompt_pool:
            # embed_len = 62
            # embed_len += prompt_length * top_k
            embed_dim = cl_cfg['embed_dim']
            # self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.prompt_pool = prompt_pool
        self.use_prompt_mask = True
        if prompt_length is not None and pool_size is not None and prompt_pool: 
            self.prompt = Prompt(length=prompt_length, embed_dim=embed_dim, embedding_key='mean', prompt_init='uniform',
                    prompt_pool=prompt_pool, prompt_key=True, pool_size=pool_size, top_k=top_k, batchwise_prompt=True,
                    prompt_key_init='uniform',)
        # ssl
        self.narration_ssl = cl_cfg["narration_ssl"]
        self.narration_dim = cl_cfg["narration_dim"]
        if self.narration_ssl:
            feature_dim = 384
            self.narration_encoder = nn.Linear(cl_cfg['narration_dim'], feature_dim)
            self.memory_bank = MemoryBank(cl_cfg['memory_size'], feature_dim)
        self.ssl_factor = cl_cfg["ssl_factor"]
            
        # adapter
        self.num_emas = 1
        self.ema_decay = 0.999
        self.use_adapter = cl_cfg['use_adapter']
        if self.use_adapter:
            self.adapt_blocks = cl_cfg['adapt_blocks']
            self.num_freeze_epochs = 10
            self.setup_adpat()

    @property
    def device(self):
        try:
            return int(os.environ["LOCAL_RANK"])
        except:
            return torch.device("cuda:0")

    def setup_adpat(self):
        if getattr(self, "pets_emas", None) is None:
            self.pets_emas = nn.ModuleList([])
            self.pets = self.create_pets()
       
        if len(self.pets_emas) < self.num_emas:
            idx = len(self.pets_emas)
            ema = ModelEmaV2(self.pets, decay=self.ema_decay)
            self.pets_emas.append(ema)
        self.attach_pets(self.pets)
    
    def attach_pets(self, pets):
        for i, b in enumerate(self.adapt_blocks):
            self.backbone.branch[b].attach_adapter(attn=pets[i])
    
    def create_pets(self):

        n = len(self.adapt_blocks)
        embed_dim = 1024

        kwargs = {'down_sample': 5, 'mode': 'parallel', 'scale': 'null'}
        kwargs["embed_dim"] = embed_dim
        pets = nn.ModuleList([])
        for _ in range(n):
            pets.append(Adapter(**kwargs))
            kwargs["embed_dim"] = kwargs["embed_dim"] // 2
        return pets
    
    def pre_train_epoch(self, task_id=0, current_epoch=0):
        # if task_id == 0 or self.num_freeze_epochs < 1:
        #     return
        
        # if current_epoch == 0:
        #     freeze(self.pets)

        # if current_epoch == self.num_freeze_epochs:
        unfreeze(self.pets)
            
    def post_train_step(self):
        for idx, ema in enumerate(reversed(self.pets_emas)):
            if idx == 0:  # the last one
                ema.update(self.pets)
            else:
                ema.update(self.pets_emas[idx - 1])
    
    def forward(self, video_list, task_id=-1, ensemble=False, hidden_state=False, is_training=True, prev_out_cls_logits=None, get_emb=False, val_qilDatasetList=None):
        # video_list:  <class 'list'> 1
        # video_list[0] <class 'dict'>

        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        src_vid, src_vid_mask = self.preprocessing(video_list, is_training)
        src_txt, src_txt_mask, src_narration, src_narration_mask = self.query_preprocessing(video_list)
        
        if hasattr(self, 'prompt'):
            x = src_txt.permute(0, 2, 1)
            if is_training:
                start = task_id * self.prompt.top_k
                end = (task_id + 1) * self.prompt.top_k
                single_prompt_mask = torch.arange(start, end).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                if end > self.prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None
            res = self.prompt(x, prompt_mask=prompt_mask, cls_features=None)
            self.total_prompt_len = res['total_prompt_len']
            x = res['prompted_embedding']
            # x = x + self.pos_embed
            src_txt = x.permute(0, 2, 1)
            feats = [x['query_feats'] for x in video_list] 
            feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
            max_len = src_txt.shape[-1]
            src_txt_mask = torch.arange(max_len)[None, :] < feats_lens[:, None]
            src_txt_mask = src_txt_mask.unsqueeze(1).to(self.device)
            reduce_sim = res['reduce_sim']
        else:
            res=dict()
            reduce_sim = None
        
        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(src_vid, src_vid_mask, src_txt, src_txt_mask)
        # print("len(feats): ",len(feats))
        # feats:  <class 'tuple'> 6
        # 0 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 2560])
        # 1 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 1280])
        # 2 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 640])
        # 3 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 320])
        # 4 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 160])
        # 5 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 80])

        fpn_feats, fpn_masks = self.neck(feats, masks)
        
        if self.training and self.narration_ssl:
            narration_feats = self.narration_encoder(src_narration.permute(0, 2, 1))
            narration_feats = narration_feats.permute(0, 2, 1)
            narration_output = narration_feats * src_narration_mask[1]
            narration_mask_un_sum = torch.sum(src_narration_mask[1], dim=2, dtype=torch.float)
            narration_mask_un_sum[narration_mask_un_sum == 0.] = 1.
            narration_out = torch.sum(narration_output, dim=2) / narration_mask_un_sum
            narration_feats = F.normalize(narration_out, dim=1)
            video_feats = []
            for feat, mask in zip(fpn_feats, fpn_masks):
                visual_output = feat * mask
                video_mask_un_sum = torch.sum(mask, dim=2, dtype=torch.float)
                video_mask_un_sum[video_mask_un_sum == 0.] = 1.
                video_out = torch.sum(visual_output, dim=2) / video_mask_un_sum
                video_feats.append(video_out)
            video_feats = torch.stack(video_feats)
            video_feats = torch.mean(video_feats, dim=0)
            video_feats = F.normalize(video_feats, dim=1)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        assert self.num_classes > 0
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        
        # bic
        if self.n_known > 0 and self.cl_name == 'bic':
            list_fpn_feats = []
            for f_feat in out_cls_logits:
                list_out = []
                init_val = 0
                final_val = 0
                for i, val_lim in enumerate(self.list_splits):
                    x_old_classes = f_feat[:, init_val:val_lim]
                    init_val = val_lim
                    x_old_classes = self.list_bias_layers[i](x_old_classes)
                    list_out.append(x_old_classes)
                x = torch.cat(list_out, dim = 1)
                list_fpn_feats.append(x)
            out_cls_logits = list_fpn_feats

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]

        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        
        if not is_training and self.use_adapter:
            for ema in self.pets_emas:
                self.attach_pets(ema.module)
                # use_text
                # if self.use_cross_modal:
                feats, masks = self.backbone(src_vid, src_vid_mask, src_txt, src_txt_mask)     # List [b,c,ti].  List [b,1,ti]
                # else:
                #     # forward the network (backbone -> neck -> heads)
                #     feats, masks = self.backbone(batched_inputs, batched_masks)     # List [b,c,ti].  List [b,1,ti]
                fpn_feats, fpn_masks = self.neck(feats, masks)
                # points: List[T x 4] with length = # fpn levels
                points = self.point_generator(fpn_feats)    # 4 means [t, reg_range_left, reg_range_right, fpn_stride]
                # out_offset: List[B, 2, T_i]
                ema_out_offsets = self.reg_head(fpn_feats, fpn_masks)
                # out_cls: List[B, #cls + 1, T_i]       
                ema_out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
                ensemble_out_cls_logits = []
                ensemble_out_offsets = []
                for cls_logit, ema_cls_logit, offset, ema_offset in zip(out_cls_logits, ema_out_cls_logits, out_offsets, ema_out_offsets):
                    ema_cls_logit = ema_cls_logit.permute(0, 2, 1)
                    ema_offset = ema_offset.permute(0, 2, 1)
                    _cls_logit = torch.stack([cls_logit, ema_cls_logit], dim=-1).mean(dim=-1)
                    _offset = torch.stack([offset, ema_offset], dim=-1).mean(dim=-1)
                    ensemble_out_cls_logits.append(_cls_logit)
                    ensemble_out_offsets.append(_offset)
                out_cls_logits = ensemble_out_cls_logits
                out_offsets = ensemble_out_offsets
            self.attach_pets(self.pets)
            
        if get_emb:
            return out_cls_logits, out_offsets, fpn_masks

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]

            assert video_list[0]['one_hot_labels'] is not None, "GT action labels does not exist"
            gt_labels = [x['one_hot_labels'].to(self.device) for x in video_list]
            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels, self.num_classes)

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets,
                prev_out_cls_logits=prev_out_cls_logits,
                reduce_sim=reduce_sim
            )
            
            if self.narration_ssl and src_narration_mask[0].sum() > 0:
                src_narration_mask_0 = src_narration_mask[0].to(torch.bool)
                self.memory_bank.update(narration_feats[src_narration_mask_0])
                ssl_factor = self.ssl_factor
                ssl_loss = self.masked_contrastive_loss(narration_feats, video_feats, src_narration_mask_0)
                losses["final_loss"] += ssl_factor * ssl_loss
                losses["ssl_loss"] = ssl_factor * ssl_loss
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets, self.num_classes,
                val_qilDatasetList
            )
            
            if ensemble:
                return video_list, points, fpn_masks, out_cls_logits, out_offsets

            return results

    def add_samples_to_mem(self, cilsettask, data, m): 
        self.memory = {**self.memory, **data}
        for class_id, videos in self.memory.items():
            random.shuffle(videos)
            if m != 'ALL':
                self.memory[class_id] = videos[:m]
            else:
                self.memory[class_id] = videos
                    
        for class_id, videos in self.memory.items():
            print('Memory... Class: {}, num videos: {}'.format(class_id, len(videos)))
    
    @torch.no_grad()
    def classify(self, x, cilsettask):
       
        batch_size = 1
        fpn_levels = 7

        if self.compute_means:
            print("Computing mean of exemplars...")
            exemplar_means = [[] for _ in range(fpn_levels)] # fpn level
            for class_id, videos in self.memory.items():
                data_class = {class_id:videos}
                class_loader = cilsettask.get_dataloader(data_class, sample_frame = True)
                features = []
                # Extract feature for each exemplar in P_y
                for video_list in class_loader:
                    batched_inputs, batched_masks = self.preprocessing(video_list, is_training=False)
                    # use_text
                    # if self.use_cross_modal:
                    src_txt, src_txt_mask, src_narration, src_narration_mask = self.query_preprocessing(video_list)
                    feats, masks = self.backbone(batched_inputs, batched_masks, src_txt, src_txt_mask)     # List [b,c,ti].  List [b,1,ti]
                    # else:
                    #     # forward the network (backbone -> neck -> heads)
                    #     feats, masks = self.backbone(batched_inputs, batched_masks)     # List [b,c,ti].  List [b,1,ti]
                    fpn_feats, fpn_masks = self.neck(feats, masks)
                    feature = [feat / feat.norm() for feat in fpn_feats] # Normalize
                    if len(features) == 0:
                        features = [[feat] for feat in feature]
                    else:
                        for i in range(len(feature)):
                            features[i].append(feature[i])
                for i in range(len(features)):
                    features_i = torch.stack(features[i], dim=0)
                    mu_y = features_i.mean(0).squeeze()
                    mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
                    exemplar_means[i].append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False

        exemplar_means = self.exemplar_means
        means_list = []
        for i in range(fpn_levels):
            means = torch.stack(exemplar_means[i], dim = 0) # (n_classes, ti, feature_size)
            means = torch.stack([means] * batch_size) # (batch_size, n_classes, ti, feature_size)
            means = means.permute(0, 2, 3, 1) # (batch_size, ti, feature_size, n_classes)
            means_list.append(means)

        batched_inputs, batched_masks = self.preprocessing([x], is_training=False)
        # use_text
        # if self.use_cross_modal:
        src_txt, src_txt_mask, src_narration, src_narration_mask = self.query_preprocessing([x])
        feats, masks = self.backbone(batched_inputs, batched_masks, src_txt, src_txt_mask)     # List [b,c,ti].  List [b,1,ti]
        # else:
        #     # forward the network (backbone -> neck -> heads)
        #     feats, masks = self.backbone(batched_inputs, batched_masks)     # List [b,c,ti].  List [b,1,ti]
        fpn_feats, fpn_masks = self.neck(feats, masks)
        feature = fpn_feats
        dists_list = []
        preds_list = []
        for i in range(fpn_levels): 
            feature_i= feature[i].data / feature[i].data.norm()    # Normalize
            feature_i = feature_i.unsqueeze(3) # (batch_size, feature_size, ti, 1)
            feature_i = feature_i.expand_as(means_list[i]) # (batch_size, feature_size, ti, n_classes)
            feature[i].data = feature_i

            dists = (feature[i] - means_list[i]).pow(2).sum(1).squeeze(0) #(batch_size, ti, n_classes)
            if len(dists.size()) == 2:
                dists = dists.unsqueeze(0)
            _, preds = dists.min(2)
            dists_list.append(dists)
            preds_list.append(preds)

        return dists_list
    
    @torch.no_grad()
    def query_preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['query_feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        # batch input shape B, T, C
        batch_shape = [len(feats), feats[0].shape[0], max_len]
        batched_inputs = feats[0].new_full(batch_shape, padding_val)
        for feat, pad_feat in zip(feats, batched_inputs):
            pad_feat[..., :feat.shape[-1]].copy_(feat)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)
        
        if self.training and self.narration_ssl:
            n_feats = [x['narration_feats'] for x in video_list]
            n_feats_lens = torch.as_tensor([feat.shape[-1] for feat in n_feats])
            n_max_len = n_feats_lens.max(0).values.item()
            
            n_batch_shape = [len(n_feats), n_feats[0].shape[0], n_max_len]
            n_batched_inputs = n_feats[0].new_full(n_batch_shape, padding_val)
            for feat, pad_feat in zip(n_feats, n_batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            n_batched_inputs = n_batched_inputs.to(self.device)
            n_batched_masks_0 = [x['narration_mask'] for x in video_list]
            n_batched_masks_1 = torch.arange(n_max_len)[None, :] < n_feats_lens[:, None]
            n_batched_masks_0 = torch.Tensor(n_batched_masks_0)
            n_batched_masks_0 = n_batched_masks_0.to(self.device)
            n_batched_masks_1 = n_batched_masks_1.unsqueeze(1).to(self.device)
            return batched_inputs, batched_masks, n_batched_inputs, (n_batched_masks_0, n_batched_masks_1)
        
        return batched_inputs, batched_masks, None, None

    @torch.no_grad()
    def preprocessing(self, video_list, is_training=True, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if is_training:
            assert max_len <= self.max_seq_len, (
                "Input length must be smaller than max_seq_len during training", max_len, self.max_seq_len)
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels, num_classes):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)

        gt_cls, gt_offset = [], []
        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            assert len(gt_segment) == len(gt_label), (gt_segment, gt_label)
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label, num_classes
            )
            # "cls_targets: " #points, num_classes
            # "reg_targets: " #points, 2
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label, num_classes):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x 2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius

            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)

            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]

        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # limit the regression range for each location and inside the center radius
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))

        # if there are still more than one ground-truths for one point
        # pick the ground-truth with the shortest duration for the point (easiest to regress)
        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        # make sure that each point can only map with at most one ground-truth
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        # gt_label_one_hot = F.one_hot(gt_label, num_classes).to(reg_targets.dtype)
        gt_label_one_hot = gt_label.to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)

        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def masked_contrastive_loss(self, text_embeddings, video_embeddings, mask, temperature=0.07):
        text_embeddings = text_embeddings[mask]
        video_embeddings = video_embeddings[mask]
        batch_size = text_embeddings.size(0)
        # Compute positive logits
        positive_logits = torch.einsum('nc,nc->n', [text_embeddings, video_embeddings]).unsqueeze(-1)
        # Compute negative logits with memory bank
        memory_embeddings = self.memory_bank.get_all()
        negative_logits_text = torch.matmul(text_embeddings, memory_embeddings.T)
        negative_logits_video = torch.matmul(video_embeddings, memory_embeddings.T)
        
        # Concatenate positive and negative logits
        logits_text = torch.cat([positive_logits, negative_logits_text], dim=1) / temperature
        logits_video = torch.cat([positive_logits, negative_logits_video], dim=1) / temperature
        
        # Labels for positive pairs
        labels = torch.zeros(batch_size, dtype=torch.long).cuda()
        
        # Contrastive loss
        loss_text = F.cross_entropy(logits_text, labels)
        loss_video = F.cross_entropy(logits_video, labels)
        return (loss_text + loss_video) / 2
    
    def losses(
            self, fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets,
            prev_out_cls_logits=None, stage_id=0,
            reduce_sim=None
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        num_classes = gt_target.shape[-1]

        # optional label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        
        if self.n_known > 0 and self.cl_name == 'l2p':
            pull_constraint_coeff = 0.1
            final_loss = final_loss - pull_constraint_coeff * reduce_sim
        
        if self.n_known > 0 and self.cl_name == 'bic':
            len_f = len(out_cls_logits)
            dist_loss = 0
            dist_factor = 0.01
            n_classes = self.cls_head.cls_head.conv.out_channels
            alpha = self.n_known / n_classes
            T = 2
            for i in range(len_f):
                out_cls_logits_i = out_cls_logits[i]
                prev_out_cls_logits_i = prev_out_cls_logits[i]
                prev_out_cls_logits_i = torch.from_numpy(prev_out_cls_logits_i).to(out_cls_logits_i.device)
                logp = F.log_softmax(out_cls_logits_i[0,:,:self.n_known]/T, dim=1)
                loss_soft_target = -torch.mean(torch.sum(prev_out_cls_logits_i[:, :self.n_known] * logp, dim=1))
                dist_loss += dist_factor * alpha * loss_soft_target
            final_loss += dist_loss       
            return {'cls_loss'   : cls_loss,
                    'reg_loss'   : reg_loss,
                    'dist_loss'  : dist_loss,
                    'final_loss' : final_loss}
            
        if self.n_known > 0 and self.cl_name == 'icarl':
            len_f = len(out_cls_logits)
            dist_loss = 0
            dist_factor = 0.01
            for i in range(len_f):
                out_cls_logits_i = out_cls_logits[i]
                prev_out_cls_logits_i = prev_out_cls_logits[i]
                prev_out_cls_logits_i = torch.from_numpy(prev_out_cls_logits_i).to(out_cls_logits_i.device)
                dist_loss += dist_factor * sum(self.dist_loss(out_cls_logits_i[0,:,y], prev_out_cls_logits_i[:,y]) for y in range(self.n_known))
            final_loss += dist_loss       
            return {'cls_loss'   : cls_loss,
                    'reg_loss'   : reg_loss,
                    'dist_loss'  : dist_loss,
                    'final_loss' : final_loss}
              
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'final_loss' : final_loss}

    @torch.no_grad()
    def inference(
            self,
            video_list,
            points, fpn_masks,
            out_cls_logits, out_offsets, num_classes,
            cilsettask=None
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]
        vid_vl = [x for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vl, vidx, fps, vlen, stride, nframes) in enumerate(
                zip(vid_vl, vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):  
             # final validate
            cls_preds_per_vid = None
            if cilsettask is not None and self.compute_means:
                cls_preds_per_vid = self.classify(vl, cilsettask)
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, num_classes,
                cls_preds_per_vid=cls_preds_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocessing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
            self,
            points,
            fpn_masks,
            out_cls_logits,
            out_offsets,
            num_classes,
            cls_preds_per_vid=None
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for idx, (cls_i, offsets_i, pts_i, mask_i) in enumerate(zip(
                out_cls_logits, out_offsets, points, fpn_masks)
        ):
            if cls_preds_per_vid is not None:
                pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

                cls_preds_per_vid_i = cls_preds_per_vid[idx].flatten()
                dist_thresh = cls_preds_per_vid_i.mean()
                keep_idxs1 = (cls_preds_per_vid_i < dist_thresh)
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
                
                num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
                _, idxs = cls_preds_per_vid_i.sort(descending=False)
                if idxs[:num_topk].max() > pred_prob.shape[0]:
                    # out of index
                    pred_prob = pred_prob.clone()
                    topk_idxs = topk_idxs.clone()
                else:
                    pred_prob = pred_prob[idxs[:num_topk]].clone()
                    topk_idxs = topk_idxs[idxs[:num_topk]].clone()
            else:
                # sigmoid normalization for output logits
                pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs = torch.div(
                topk_idxs, num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments': segs_all,
                   'scores': scores_all,
                   'labels': cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen
            # 4: repack the results
            processed_results.append(
                {'video_id': vidx,
                 'segments': segs,
                 'scores': scores,
                 'labels': labels}
            )

        return processed_results
