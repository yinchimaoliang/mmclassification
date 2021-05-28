import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS, build_backbone
from .cls_head import ClsHead


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes, in_channels, *args, **kwargs):
        super(LinearClsHead, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label, **kwargs):
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses


@HEADS.register_module()
class MulLabelLinearClsHead(LinearClsHead):

    def __init__(self,
                 wei_net_backbone,
                 pool_kernel,
                 fc_in_channels,
                 num_expert,
                 num_classes,
                 in_channels,
                 mul_label_ind=None,
                 sigma=1,
                 loss_step=1000,
                 final_label_ind=0,
                 *args,
                 **kwargs):
        super(MulLabelLinearClsHead, self).__init__(num_classes, in_channels,
                                                    *args, **kwargs)

        self.wei_net_backbone = build_backbone(wei_net_backbone)
        self.wei_net_pool = nn.AvgPool2d(kernel_size=pool_kernel)
        self.wei_net_fc = nn.Linear(fc_in_channels, num_expert)
        self.wei_net_softmax = nn.Softmax(dim=1)
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.final_label_ind = final_label_ind
        self.mul_label_ind = mul_label_ind
        self.iter_num = 0
        self.sigma = sigma
        self.loss_step = loss_step
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def forward_train(self, x, gt_label, img):
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, img)
        return losses

    def loss(self, cls_score, gt_label, img):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score,
                                        gt_label[..., self.final_label_ind])
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        feature_map = self.wei_net_pool(self.wei_net_backbone(img))
        wei_expert = self.wei_net_softmax(
            self.wei_net_fc(feature_map.view(feature_map.shape[0], -1)))

        loss_single_label = self.compute_loss(
            cls_score,
            gt_label[..., self.final_label_ind],
            avg_factor=num_samples)
        loss_mul_labels = []
        for label_ind in self.mul_label_ind:
            loss_mul_labels.append(
                self.compute_loss(
                    cls_score,
                    gt_label[..., label_ind],
                    avg_factor=num_samples))
        loss_mul_label = torch.mean(wei_expert * torch.stack(loss_mul_labels))
        iter_num_sig = self.sigma * (torch.sigmoid(
            torch.tensor(self.iter_num // self.loss_step).float()) - 1 / 2) * 2
        iter_num_sig = iter_num_sig.type_as(cls_score)
        losses['loss'] = (1 / (1 + iter_num_sig)) * loss_single_label + (
            iter_num_sig / (1 + iter_num_sig)) * loss_mul_label
        self.iter_num += 1
        return losses
