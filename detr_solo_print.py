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
from mmcv.runner import auto_fp16
from scipy.optimize import linear_sum_assignment

class FPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

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


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict, eos_coef):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    @torch.no_grad()
    def match(self, output, target):
        #print(target)
        num_queries = output["pred_cls"].shape[1]

        out_cls = output["pred_cls"][0]
        out_mask = output["pred_mask"]
        print("out")
        print(out_cls.shape)
        print(out_mask.shape)
        tgt_cls = target["cls"][0]
        tgt_mask = target["mask"][0]

        print("tgt")
        print(tgt_cls.shape)
        print(tgt_mask.shape)
        #print(out_mask[:, None].shape)
        #out_mask = F.interpolate(out_mask[:, None], size=tgt_mask.shape[-2:],
        #                        mode="bilinear", align_corners=False).squeeze(1)

        tgt_mask = tgt_mask.flatten(1, 2)
        out_mask = out_mask.flatten(1, 2)

        print("mask")
        print(out_mask.shape)
        print(tgt_mask.shape)
        cost_cls = -out_cls[:,tgt_cls]



        print("cost")
        print(cost_cls.shape)
        #print(out_mask.shape)
        #print()


        #a = out_mask*tgt_mask[:1,:]
        print(out_mask)
        print(tgt_mask)
        a = 2*torch.mm(out_mask, tgt_mask.t())
        b1 = out_mask.sum(1).expand(a.shape[1],a.shape[0]).t()
        b2 = tgt_mask.sum(1).expand(a.shape[0],a.shape[1])
        print(a.shape)
        print(b1.shape)
        print(b2.shape)
        #b = out_mask*out_mask.t()+tgt_mask*tgt_mask.t()
        cost_mask = 1-((a+1)/(b1+b2+1))

        
        C = cost_mask
        C = C.view(num_queries,-1).cpu()
        print(C.shape)
        i, j = linear_sum_assignment(C)
        print(out_mask[i].sum)
        print(a[i,j])
        print(b1[i,j])
        print(b2[i,j])
        print(C[i,j])
        return torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)


    def dice_loss(self, pred, target, num):
        print("dice_loss")
        #print(pred)
        #print(target)
        pred = pred.flatten(1, 2)
        target = target.flatten(1, 2)
        #pred = pred.flatten(1)
        print(pred)
        print(target)
        numerator = 2 * (pred * target).sum(1)
        print(numerator)
        denominator = pred.sum(1) + target.sum(1)
        print(denominator)
        loss = 1-(numerator + 1) / (denominator + 1)
        print(loss)
        return loss.sum()/num


    def forward(self, output, target):
        #print(target)
        #print(target[0]['boxes'].device)
        target["mask"] = target["mask"].to(output["pred_mask"]) 
        output["pred_mask"] = F.interpolate(output["pred_mask"][:, None], size=target["mask"][0].shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        src, tgt = self.match(output,target)
        print(src)
        print(target["cls"].shape)
        print(target["mask"].shape)
        #loss_cls
        pred_cls = output["pred_cls"][0]
        #target_class_o = target["cls"][src]
        print(pred_cls)
        target_cls = torch.full(pred_cls.shape[:1], self.num_classes, dtype=torch.int64, device=pred_cls.device)
        print(target_cls)
        target_cls[src] = target["cls"][0][tgt]
        print("cls")
        print(target_cls.shape)
        print(pred_cls.shape)
        loss_cls = F.cross_entropy(pred_cls, target_cls, self.empty_weight)
        print(loss_cls)
        #loss_seg

        pred_mask = output["pred_mask"][src]
        target_mask = target["mask"][0][tgt]
        #print(pred_mask)
        #print(target_mask.shape)
        #print(src)
        loss_mask = self.dice_loss(pred_mask, target_mask, target_mask.shape[0])

        losses = {"loss_cls":loss_cls,"loss_mask":loss_mask}
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
    def __init__(self, num_classes, hidden_dim=256, nheads=1,
                 num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        self.neck = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, start_level=0, num_outs=5)
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_seg = nn.Linear(hidden_dim, 256)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

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
        print("x1.shape")
        print(x1.shape)
        x2 = self.backbone.layer2(x1)
        print("x2.shape")
        print(x2.shape)
        x3 = self.backbone.layer3(x2)
        print("x3.shape")
        print(x3.shape)
        x4 = self.backbone.layer4(x3)
        print("x4.shape")
        print(x4.shape)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x4)
        print("h.shape")
        print(h.shape)
        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        
        src = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        tgt = self.query_pos.unsqueeze(1)

        print("src.shape")
        print(src.shape)
        print("tgt.shape")
        print(tgt.shape)
        h = self.transformer(src,tgt).transpose(0, 1)
        print("h.shape")
        print(h.shape)
        # finally project transformer outputs to class labels and bounding boxes


        feature_map = self.neck([x1,x2,x3,x4])
        print("feature_map.shape")
        print(feature_map[0].shape)

        cls = self.linear_class(h)
        print("cls.shape")
        print(cls.shape)

        seg_kernel = self.linear_seg(h).unsqueeze(0)
        seg_kernel = seg_kernel.permute(2,3,0,1)
        
        seg_preds = F.conv2d(feature_map[0], seg_kernel, stride=1).squeeze(0).sigmoid()

        return {'pred_cls': cls, 
                'pred_mask': seg_preds}









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


