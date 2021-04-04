from PIL import Image
import requests
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import torch.nn.functional as F

#torch.set_grad_enabled(False)
import mmcv
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import auto_fp16
from scipy.optimize import linear_sum_assignment



class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
"""
class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 norm_groups=32,
                 inplace=True):
        super(ConvModule, self).__init__()
        self.inplace = inplace

        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False)

        # build normalization layers
        self.norm = nn.GroupNorm(num_channels=out_channels, num_groups=norm_groups)

        # build activation layer
        self.activate = nn.ReLU(inplace=inplace)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        kaiming_init(self.conv, nonlinearity='relu')
        constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)
        return x
"""
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 num_classes,
                 conv_cfg=None,
                 norm_cfg=None):
        super(MF, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        print(self.norm_cfg)
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = ConvModule(
                    self.in_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i==3 else self.in_channels
                    one_conv = ConvModule(
                        chn,
                        self.out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue

                one_conv = ConvModule(
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            ConvModule(
                self.out_channels,
                self.num_classes,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)


    def forward(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level + 1)
        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
                y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([input_feat.shape[0], 1, -1, -1])
                x = x.expand([input_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)
            feature_next = self.convs_all_levels[i](input_p)
            feature_add_all_level += feature_next
        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred

class SetCriterion(nn.Module):

    def __init__(self, num_classes, N, weight_dict, eos_coef):

        super().__init__()
        self.num_classes = num_classes
        self.N = N
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1).cuda()
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    @torch.no_grad()
    def match(self, output, target):
        num_queries = output["pred_cls"].shape[1]

        out_cls = output["pred_cls"][0]
        out_mask = output["pred_mask"]
        tgt_cls = target["cls"][0]
        tgt_mask = target["mask"][0]

        tgt_mask = tgt_mask.flatten(1, 2)
        out_mask = out_mask.flatten(1, 2)

        cost_cls = -out_cls[:,tgt_cls]

        a = 2*torch.mm(out_mask, tgt_mask.t())
        b1 = out_mask.sum(1).expand(a.shape[1],a.shape[0]).t()
        b2 = tgt_mask.sum(1).expand(a.shape[0],a.shape[1])
        cost_mask = 1-((a+1)/(b1+b2+1))
        
        C = 5*cost_mask+cost_cls
        C = C.view(num_queries,-1).cpu()
        i, j = linear_sum_assignment(C)
        return torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)


    def dice_loss(self, pred, target, num):
        pred = pred.flatten(1, 2)
        target = target.flatten(1, 2)
        numerator = 2 * (pred * target).sum(1)
        denominator = pred.sum(1) + target.sum(1)
        loss = 1-(numerator + 1) / (denominator + 1)
        return loss.sum()/max(num,1)

    def cos_distance(self, A):
        prod = torch.mm(A, A.t())
        norm = torch.norm(A,p=2,dim=1).unsqueeze(0)
        cos = prod.div(torch.mm(norm.t(),norm))
        return cos
    
    
    def forward(self, output, target):
        seed = output["raw_seed"][0]
        cos = self.cos_distance(seed)
        print(cos)
        cos = torch.exp(cos)
        loss_contrastive = -torch.log(1/(cos.sum(0))).sum()/self.N
        print(loss_contrastive)
        
        
        target["mask"] = target["mask"].to(output["pred_mask"]) 
        output["pred_mask"] = F.interpolate(output["pred_mask"][:, None], size=target["mask"][0].shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        src, tgt = self.match(output,target)
        pred_cls = output["pred_cls"][0]
        target_cls = torch.full(pred_cls.shape[:1], self.num_classes, dtype=torch.int64, device=pred_cls.device)
        target_cls[src] = target["cls"][0][tgt]
        print(pred_cls.argmax(1))
        print(target_cls)
        loss_cls = F.cross_entropy(pred_cls, target_cls, self.empty_weight)

        pred_mask = output["pred_mask"][src]
        target_mask = target["mask"][0][tgt]
        loss_mask = self.dice_loss(pred_mask, target_mask, target_mask.shape[0])
        
        #loss_contrastive =
        #loss_mask_invalid = self.dice_loss(invalid_mask, empty_mask, empty_mask.shape[0])

        losses = {"loss_cls":loss_cls,"loss_mask":loss_mask, "loss_contrastive":loss_contrastive}
        print(losses)
        return losses

class detr_solo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, N = 100, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50(pretrained=True)
        self.neck = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, start_level=0, num_outs=5)
        self.in_channels = 256
        self.out_channels = 256
        self.mask_feature = MF(in_channels=256,
            out_channels=256,
            start_level=0,
            end_level=3,
            num_classes=256,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        #self.linear_seg = nn.Linear(hidden_dim, 256)
        self.linear_seg = MLP(hidden_dim, hidden_dim, 256, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(N, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x4)
        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        
        src = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        tgt = self.query_pos.unsqueeze(1)
        h = self.transformer(src,tgt).transpose(0, 1)
        # finally project transformer outputs to class labels and bounding boxes
        #print("output")
        #print(h)
        feature_map = self.neck([x1,x2,x3,x4])

        feature_map = self.mask_feature(feature_map[self.mask_feature.start_level:self.mask_feature.end_level + 1])
        #print(feature_map.shape)

        cls = self.linear_class(h)

        seg_kernel = self.linear_seg(h).unsqueeze(0)
        seg_kernel = seg_kernel.permute(2,3,0,1)
        
        seg_preds = F.conv2d(feature_map, seg_kernel, stride=1).squeeze(0).sigmoid()

        return {'pred_cls': cls, 
                'pred_mask': seg_preds,
                'raw_seed': h}









#model = detr_solo(num_classes=91)
#print(type(model))

#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#im = Image.open(requests.get(url, stream=True).raw)

# standard PyTorch mean-std input image normalization
#transform = T.Compose([
#    T.Resize(800),
#    T.ToTensor(),
#    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#])


#img = transform(im).unsqueeze(0)
#img = img.repeat(2,1,1,1)

#outputs = model(img)



