import torch
from torch import nn
from torch.nn import functional as F

""" 
nn.py provides classes for deformable convolution built on PyTorch functionality.

gLN and cLN layers are copied from the SpeechBrain framework:
https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/lobes/models/conv_tasnet.html
See licence here: https://github.com/speechbrain/speechbrain/blob/develop/LICENSE
Copyright SpeechBrain 2022.

The reset_paramters functions were adapted from the PyTorch ConvNd classes:
https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d
See licence here: https://github.com/pytorch/pytorch/blob/master/LICENSE
Copyright 2022, PyTorch Contributors.

The remainder of this module is original code belonging to the dc1d project.
Author: William Ravenscroft, August 2022
Copyright William Ravenscroft 2022.
"""

# Generic
import math, time
from turtle import forward
from typing import Optional, Tuple

# PyTorch
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _reverse_repeat_tuple
from typing import Callable

# dc1d
""" 
ops.py provides operation functions for defomrable convolution

Author: William Ravenscroft, August 2022
Copyright William Ravenscroft 2022
"""

# PyTorch
from multiprocessing.dummy import Pool
import torch
import torch.multiprocessing as mp
# from torch.cuda.amp import autocast
from functools import partial

def full_seq_linterpolate(
    x, 
    offsets,
    kernel_size, 
    dilation,
    stride,
    dilated_positions=None,
    device="cpu",
    _test=False
    ):
    """
    Full sequence linear interpolation function for 1D deformable convolution. This should only be used for short sequence lengths else the user will be likely to run into memory issues.
    Args:
        x (Tensor): Input Data Tensor of shape batch size x channels x length
        offsets (Tensor): Deforming offset Tensor of shape batch size x offset groups x number of offset positions x kernel size
        kernel_size (int): Value of convolution kernel size
        dilation (int): Value of convolution kernel dilation factor
        stride (int): Value convolution kernel stride
        dilated_positions (Tensor): Allows user to save computation by using precomputed dilation offset positions. If not these can be computed from dilation kernel_size for each function call
        device: Device to operate function on. Default: "cpu".
    """
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, dilation*kernel_size-dilation,kernel_size,device=device) # kernel_size
    max_t0 = (offsets.shape[-2]-1)*stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2],device=device).unsqueeze(-1) # out_length x 1
    dilated_offsets_repeated = dilated_positions+offsets
    T = t0s + dilated_offsets_repeated # batch_size x groups x out_length x kernel_size

    if _test:
        print("x:",x.shape) # batch_size x in_channels x input_length
        print("offsets:",offsets.shape) # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:",t0s.shape) # out_lengths x 1
        print("dilated positions:",dilated_positions.shape) # kernel_size
        print("dilated_offsets_repeated:",dilated_offsets_repeated.shape)
    
    
    max_U = x.shape[-1]-1
    U = torch.linspace(0,max_U,max_U+1,device=device).repeat(1,1,1,1,1) # 1 x 1 x 1 x 1 x_length
    abs_sub = 1-torch.abs(U-T.unsqueeze(-1)) # batch_size x groups x out_length x kernel_size x_length
    _zeros = torch.zeros(abs_sub.shape,device=device)
    G = torch.max(_zeros, abs_sub) # batch_size x groups x out_length x kernel_size x_length
    
    if _test:
        print("T:",T.shape) # batch_size x groups x out_length x kernel_size
        print("U:",U.shape); 
        print("abs_sub:", abs_sub.shape)
        print("G:",G.shape)
    
    mx = torch.multiply(G.moveaxis((0,1),(2,3)),x)
    x_offset = torch.sum(mx, axis=-1).moveaxis((0,1),(-2,-1))   # batch_size x channels x output_length x kernel size

    if _test:
        print("mx:",mx.shape)
        print("x_offset:", x_offset.shape)
        print(
            "Desired shape:",
            (batch_size, x.shape[1], offsets.shape[-2], kernel_size),
            "(batch_size, in_channels, output_length, kernel_size)")
        assert x_offset.shape == (batch_size, x.shape[1],offsets.shape[-2], kernel_size)
    return x_offset

def _interpolate(i,x,t0s,T,kernel_rfield,x_offset,device):
    t0 = int(t0s[i,0].item())
    max_U = int(t0+kernel_rfield-1)
    U = torch.linspace(t0,max_U,kernel_rfield,device=device) #  kernel_size*max_dilation_factor
    abs_sub = 1-torch.abs(U.repeat(1,1,T.shape[-1],1)-T[:,:,i,:].unsqueeze(-1)) # batch_size x groups x kernel_size
    _zeros = torch.zeros(abs_sub.shape,device=device)
    G = torch.max(_zeros, abs_sub) # batch_size x channels x out_length x kernel_size x input_length
    mx = torch.multiply(G,x[:,:,t0:max_U+1].unsqueeze(-2))
    x_offset[:,:,i,:,] = torch.sum(mx, axis=-1)   # batch_size x channels x output_length x kernel size

def kernel_width_linterpolate(
    x, 
    offsets,
    kernel_size, 
    dilation,
    stride,
    dilated_positions=None,
    device="cpu",
    _test=False,
    _multiprocess=False,
    _max_memory=True
):
    assert x.device == offsets.device, "x and offsets must be on same device"
    kernel_rfield=dilation*(kernel_size-1)+1
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, kernel_rfield-1,kernel_size,device=device) # kernel_size
    
    max_t0 = (offsets.shape[-2]-1)*stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2],device=device).unsqueeze(-1) # out_length x 1
    dilated_offsets_repeated = dilated_positions+offsets
    
    T = t0s + dilated_offsets_repeated # batch_size x channels x out_length x kernel_size
    T = torch.max(T, t0s)
    T = torch.min(T, t0s+torch.max(dilated_positions))


    if _test:
        print("x:",x.shape) # batch_size x in_channels x input_length
        print("offsets:",offsets.shape) # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:",t0s.shape) # out_lengths x 1
        print("dilated positions:",dilated_positions.shape) # kernel_size
        print("dilated_offsets_repeated:",dilated_offsets_repeated.shape)
        print("T:",T.shape) # batch_size x groups x out_length x kernel_rfield

    if _max_memory:
        U = t0s+torch.linspace(0,kernel_rfield-1,kernel_rfield,device=device).repeat(1,1,1,1) # 1 x 1 x 1 x length x kernel_rfield
        if _test:
            print("U:",U.shape)

        abs_sub = 1-torch.abs(U.unsqueeze(-1)-T.unsqueeze(-2)) # batch_size x groups x out_length x kernel_size x_length
        if _test:
            print("abs_sub:", abs_sub.shape)

        _zeros = torch.zeros(abs_sub.shape,device=device)
        x = x.unfold(dimension=2, size=kernel_rfield, step=stride).unsqueeze(-1)
        if _test:
            print("x unfolded:",x.shape)

        G = torch.max(_zeros, abs_sub) # batch_size x groups x out_length x kernel_rfield x kernel_size
        if _test:
            print("G:",G.shape)

        mx = torch.multiply(G,x)
        x_offset = torch.sum(mx, axis=-2)  # batch_size x channels x output_length x kernel size
        return x_offset

    elif not _multiprocess: 
        x_offset = torch.zeros((x.shape[0], x.shape[1], offsets.shape[-2], kernel_size),device=x.device)
        for i in range(t0s.shape[0]):
            t0 = int(t0s[i,0].item())
            max_U = int(t0+kernel_rfield-1)
            U = torch.linspace(t0,max_U,kernel_rfield,device=device) #  kernel_size*max_dilation_factor
            abs_sub = 1-torch.abs(U.repeat(1,1,T.shape[-1],1)-T[:,:,i,:].unsqueeze(-1)) # batch_size x groups x kernel_size
            _zeros = torch.zeros(abs_sub.shape,device=device)
            G = torch.max(_zeros, abs_sub) # batch_size x channels x out_length x kernel_size x input_length
            mx = torch.multiply(G,x[:,:,t0:max_U+1].unsqueeze(-2))
            x_offset[:,:,i,:,] = torch.sum(mx, axis=-1)   # batch_size x channels x output_length x kernel size
        return x_offset

    else:
        x_offset = torch.zeros((x.shape[0], x.shape[1], offsets.shape[-2], kernel_size),device=x.device)
        T.share_memory_()
        x.share_memory_()
        t0s.share_memory_()
        x_offset.share_memory_()
        with mp.Pool() as p:
            p.map(
                partial(_interpolate,t0s=t0s,T=T,x=x,x_offset=x_offset,kernel_rfield=kernel_rfield,device=x.device),
                range(t0s.shape[0])
                )
        return x_offset

def efficient_linterpolate(
    x, 
    offsets,
    kernel_size, 
    dilation,
    stride,
    dilated_positions=None,
    device="cpu",
    _test=False,
    unconstrained=False
):  

    assert x.device == offsets.device, "x and offsets must be on same device"
    kernel_rfield=dilation*(kernel_size-1)+1
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, kernel_rfield-1,kernel_size,device=offsets.device,dtype=offsets.dtype) # kernel_size

    max_t0 = (offsets.shape[-2]-1)*stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2],device=offsets.device,dtype=offsets.dtype).unsqueeze(-1) # out_length x 1
    dilated_offsets_repeated = dilated_positions+offsets
    
    T = t0s + dilated_offsets_repeated # batch_size x channels x out_length x kernel_size
    if not unconstrained:
        T = torch.max(T, t0s)
        T = torch.min(T, t0s+torch.max(dilated_positions))
    else:
        T = torch.clamp(T, 0.0, float(x.shape[-1]))

    if _test:
        print("x:",x.shape) # batch_size x in_channels x input_length
        print("offsets:",offsets.shape) # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:",t0s.shape) # out_lengths x 1
        print("dilated positions:",dilated_positions.shape) # kernel_size
        print("dilated_offsets_repeated:",dilated_offsets_repeated.shape)
        print("T:",T.shape) # batch_size x groups x out_length x kernel_rfield

    with torch.no_grad():
        U = torch.floor(T).to(torch.long) # 1 x 1 x length x kernel_rfield
        U = torch.clamp(U,min=0,max=x.shape[2]-2)

        if _test:
            print("U:",U.shape)

        U = torch.stack([U,U+1],dim=-1)
        if U.shape[1] < x.shape[1]:
            U=U.repeat(1,x.shape[1],1,1,1)
        if _test:
            print("U:", U.shape)

    x=x.unsqueeze(-1).repeat(1,1,1,U.shape[-1])
    x = torch.stack([x.gather(index=U[:,:,:,i,:],dim=-2) for i in range(U.shape[-2])],dim=-1)
    
    G = torch.max(torch.zeros(U.shape,device=device), 1-torch.abs(U-T.unsqueeze(-1))) # batch_size x groups x out_length x kernel_rfield x kernel_size
    
    if _test:
        print("G:",G.shape)

    mx = torch.multiply(G,x.moveaxis(-2,-1))
    
    return torch.sum(mx, axis=-1) # .float()  # batch_size x channels x output_length x kernel size

class DeformConv1d(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = "valid",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
        device: str = "cpu",
        interpolation_function: Callable = efficient_linterpolate,
        unconstrained: str = None, # default None to maintain backwards compatibility
        *args,
        **kwargs
        ) -> None:
        """
        1D Deformable convolution kernel layer
        Args:
            in_channels (int): Value of convolution kernel size
            out_channels (int): Value of convolution kernel dilation factor
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int): See torch.nn.Conv1d for details. Default "valid". Still experimental beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int) = 1
            bias (bool) = True
            padding_mode: See torch.nn.Conv1d for details. Default "reflect". Still experimental beware of unexpected behaviour.
            device: Device to operate function on. Default: torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """

        self.device = device
        self.interpolation_function = interpolation_function
        
        super(DeformConv1d, self).__init__(*args,**kwargs)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0]
            if padding == 'same':
                for d, k, i in zip([dilation], [kernel_size], range( 0, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight = Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size)
        )

        self.dilated_positions = torch.linspace(0,
            dilation*kernel_size-dilation,
            kernel_size,
            ) # automatically store dilation offsets

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        if not unconstrained==None:
            self.unconstrained=unconstrained

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self, 
        input: Tensor, 
        offsets: Tensor, 
        mask: Optional[Tensor] = None # TODO
        ) -> Tensor:
        """
        Forward pass of 1D deformable convolution layer
        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            offset (Tensor[batch_size, offset_groups, output length, kernel_size]):
                offsets to be applied for each position in the convolution kernel. Offset groups can be 1 or such that (in_channels%offset_groups == 0) is satisfied.
            mask (Tensor[batch_size, offset_groups, kernel_width, 1, out_width]): To be implemented
        Returns:
            output (Tensor[batch_size, in_channels, length]): output tensor
        """
        
        if self.padding_mode != 'zeros':
            input = F.pad(
                input, 
                self._reversed_padding_repeated_twice, 
                mode=self.padding_mode
                )
        if not self.device == offsets.device: # naive assumption
            self.device = offsets.device
        if self.dilated_positions.device != self.device:
            self.dilated_positions = self.dilated_positions.to(offsets.device)

        if "unconstrained" in self.__dict__.keys():
            input = self.interpolation_function(
                input, 
                kernel_size=self.kernel_size, 
                dilation=self.dilation,
                offsets=offsets, 
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device,
                unconstrained=self.unconstrained
                )
        else:
            input = self.interpolation_function(
                input, 
                kernel_size=self.kernel_size, 
                dilation=self.dilation,
                offsets=offsets, 
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device
                ) 
        input = input.flatten(-2,-1)
        output=F.conv1d(input, 
            self.weight, 
            self.bias, 
            stride=self.kernel_size, 
            groups=self.groups
            )
        
        return output

class PackedDeformConv1d(DeformConv1d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = "valid",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
        offset_groups: int = 1,
        device: str = "cpu",
        interpolation_function: Callable = efficient_linterpolate,
        unconstrained: str = None, # default None to maintain backwards compatibility
        *args,
        **kwargs
        ) -> None:
        """
        Packed 1D Deformable convolution class. Depthwise-Separable convolution is used to compute offsets.
        Args:
            in_channels (int): Value of convolution kernel size
            out_channels (int): Value of convolution kernel dilation factor
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int): See torch.nn.Conv1d for details. Default "valid". Still experimental beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int): 1 or in_channels
            bias (bool): Whether to use bias. Default = True
            padding_mode (str): See torch.nn.Conv1d for details. Default "reflect". Still experimental beware of unexpected behaviour.
            offset_groups (int): 1 or in_channels
            device: Device to operate function on. Default: torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """
        assert offset_groups in [1,in_channels], "offset_groups only implemented for offset_groups in {1,in_channels}"
        
        super(PackedDeformConv1d,self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            interpolation_function = interpolation_function,
            unconstrained=unconstrained,
            *args,
            **kwargs
            )
        self.offset_groups = offset_groups

        self.offset_dconv = nn.Conv1d(in_channels,in_channels,kernel_size,stride=1,groups=in_channels,padding=padding,padding_mode=padding_mode,bias=False)
        self.odc_norm = gLN(in_channels)
        self.odc_prelu = nn.PReLU()
        
        self.offset_pconv = nn.Conv1d(in_channels,kernel_size*offset_groups,1,stride=1,bias=False)
        self.offset_pconv_tal = nn.Conv1d(2,kernel_size*offset_groups,1,stride=1,bias=False)   # for tal task
        self.odp_norm = gLN(kernel_size*offset_groups)
        self.odp_prelu = nn.PReLU()

        self.device=device
        self.to(device)
    
    def forward(self, input, offsets=None, with_offsets=False):
        """
        Forward pass of 1D deformable convolution layer
        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            
        Returns:
            output (Tensor[batch_size, in_channels, length]): output tensor
        """
        self.device = offsets.device # naive assumption to fix errors
        if offsets is not None:
            if len(offsets.shape) == 3:
                offsets = self.offset_pconv_tal(offsets)
                offsets = self.odp_norm(self.odp_prelu(offsets).moveaxis(1,2)).moveaxis(2,1) # batch_size x (kernel_size*offset_groups) x length
                offsets = offsets.unsqueeze(0).chunk(self.offset_groups,dim=2)# batch_size x offset_groups x length x kernel_size
                offsets = torch.vstack(offsets).moveaxis((0,2),(1,3))# batch_size x offset_groups x length x kernel_size
        elif offsets is None:
            offsets = self.offset_dconv(input)
            offsets = self.odc_norm(self.odc_prelu(offsets).moveaxis(1,2)).moveaxis(2,1)
            assert str(input.device) == str(self.device), f"Input is on {input.device} but self is on {self.device}"
            assert str(input.device) == str(offsets.device), f"Input is on {input.device} but self is on {self.device}"
            offsets = self.offset_pconv(offsets)

        if with_offsets:
            return super().forward(input,offsets), offsets
        else:
            return super().forward(input,offsets)
EPS=1e-9

class gLN(nn.Module):
    """Global Layer Normalization (gLN).

    Copyright SpeechBrain 2022

    Arguments
    ---------
    channel_size : int
        Number of channels in the third dimension.

    Example
    -------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = GlobalLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(gLN, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()


    def forward(self, y):
        """
        Arguments
        ---------
        y : Tensor
            Tensor shape [M, K, N]. M is batch size, N is channel size, and K is length.

        Returns
        -------
        gLN_y : Tensor
            Tensor shape [M, K. N]
        """
        mean = y.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True
        )  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2))
            .mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class cLN(nn.Module):
    """Channel-wise Layer Normalization (cLN).

    Arguments
    ---------
    channel_size : int
        Number of channels in the normalization dimension (the third dimension).

    Example
    -------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = ChannelwiseLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(cLN, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()


    def forward(self, y):
        """
        Args:
            y: [M, K, N], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, K, N]
        """
        mean = torch.mean(y, dim=2, keepdim=True)  # [M, K, 1]
        var = torch.var(y, dim=2, keepdim=True, unbiased=False)  # [M, K, 1]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y
   







class CxAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CxAM, self).__init__()
        self.key_conv = nn.Conv1d(in_channels, out_channels//reduction, 1)
        self.query_conv = nn.Conv1d(in_channels, out_channels//reduction, 1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x):
        batch_size, C, T = x.size()

        proj_query = self.query_conv(x).permute(0, 2, 1)   # B x T x C'

        proj_key = self.key_conv(x)  # B x C' x T

        R = torch.bmm(proj_query, proj_key)     # [b,t,t]    
        # 先进行全局平均池化, 此时 R 的shape为 B x N x 1 x 1, 再进行view, R 的shape为 B x 1 x W x H
        attention = F.softmax(R, dim=-1)        # [b,t,t]

        proj_value = self.value_conv(x).permute(0,2,1)     # [b,t,c]

        out = torch.bmm(attention, proj_value)  # [b,t,c]

        return out.permute(0, 2, 1)

class CnAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CnAM, self).__init__()
        # 原文中对应的P, Z, S
        self.query_conv = nn.Conv1d(in_channels, out_channels // reduction, 1)
        self.key_conv = nn.Conv1d(in_channels, out_channels // reduction, 1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, 1)

    # CnAM使用了FPN中的F5和CEM输出的特征图F
    def forward(self, x, init):
        batch_size, C, T = x.size()

        proj_query = self.query_conv(init).permute(0, 2, 1)   # B x T x C'

        proj_key = self.key_conv(init)  # B x C' x T

        R = torch.bmm(proj_query, proj_key)     # [b,t,t]    
        # 先进行全局平均池化, 此时 R 的shape为 B x N x 1 x 1, 再进行view, R 的shape为 B x 1 x W x H
        attention = F.softmax(R, dim=-1)        # [b,t,t]

        proj_value = self.value_conv(x).permute(0,2,1)     # [b,t,c]

        out = torch.bmm(attention, proj_value)  # [b,t,c]

        return out.permute(0, 2, 1)

        return out

class DenseBlock(nn.Module):
    def __init__(self, input_num, num1, num2, rate, drop_out):
        super(DenseBlock, self).__init__()

        # C: 2048 --> 512 --> 256
        self.conv1x1 = nn.Conv1d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=num1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dilaconv = nn.Conv1d(in_channels=num1, out_channels=num2, kernel_size=3, padding=1 * rate, dilation=rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.ConvGN(self.conv1x1(x))
        x = self.relu1(x)
        x = self.dilaconv(x)
        x = self.relu2(x)
        x = self.drop(x)
        return x


class DenseAPP(nn.Module):
    def __init__(self, num_channels=2048):
        super(DenseAPP, self).__init__()
        self.drop_out = 0.1
        self.channels1 = 512
        self.channels2 = 256
        self.num_channels = num_channels
        self.aspp3 = DenseBlock(self.num_channels, num1=self.channels1, num2=self.channels2, rate=3,
                                drop_out=self.drop_out)
        self.aspp6 = DenseBlock(self.num_channels + self.channels2 * 1, num1=self.channels1, num2=self.channels2,
                                rate=6,
                                drop_out=self.drop_out)
        self.aspp12 = DenseBlock(self.num_channels + self.channels2 * 2, num1=self.channels1, num2=self.channels2,
                                 rate=12,
                                 drop_out=self.drop_out)
        self.aspp18 = DenseBlock(self.num_channels + self.channels2 * 3, num1=self.channels1, num2=self.channels2,
                                 rate=18,
                                 drop_out=self.drop_out)
        self.aspp24 = DenseBlock(self.num_channels + self.channels2 * 4, num1=self.channels1, num2=self.channels2,
                                 rate=24,
                                 drop_out=self.drop_out)
        self.conv1x1 = nn.Conv1d(in_channels=5*self.channels2, out_channels=num_channels, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=num_channels)

    def forward(self, feature):
        aspp3 = self.aspp3(feature)
        feature = torch.cat([aspp3, feature], dim=1)
        aspp6 = self.aspp6(feature)
        feature = torch.cat([aspp6, feature], dim=1)
        aspp12 = self.aspp12(feature)
        feature = torch.cat([aspp12, feature], dim=1)
        aspp18 = self.aspp18(feature)
        feature = torch.cat([aspp18, feature], dim=1)
        aspp24 = self.aspp24(feature)

        x = torch.cat([aspp3, aspp6, aspp12, aspp18, aspp24], dim=1)
        out = self.ConvGN(self.conv1x1(x))
        return out


class ACConv(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.denseapp = DenseAPP(d_in)
        self.CxAM = CxAM(in_channels=d_in, out_channels=d_out)
        self.CnAM = CnAM(in_channels=d_in, out_channels=d_out) 

    def forward(self, x, mask):
        out_mask = mask.to(x.dtype)

        out = self.denseapp(x)          # [b,c,t]

        # ==== add cxam anam 
        # cxam = self.CxAM(out)
        # cnam = self.CnAM(out, x)
        # out = cxam + cnam
        # ==== add cxam anam 

        out = out * out_mask
        return out, out_mask.bool()


def calc_ious(input_offsets, target_offsets, eps=1e-8):
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)
    return iouk

def calc_cls_scores(input_preds, gt_targets, eps=1e-8):
    # [#, n_cls]. [#, n_cls]
    cls_idx = gt_targets.max(-1)[1]    # [numpos]
    gt_one_label = F.one_hot(cls_idx, input_preds.shape[-1])
    cls_scores = input_preds[gt_one_label!=0]   # [numpos]
    return cls_scores
