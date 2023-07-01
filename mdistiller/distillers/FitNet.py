import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes


class FitNet(Distiller):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, child1, child2, cfg):
        super(FitNet, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT  # 硬损失权重
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT  # 软损失权重
        self.hint_layer = cfg.FITNET.HINT_LAYER  # 选取 hint层
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )  # 得到教师网络和学生网络各个特征图的尺寸
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )  # 回归器

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())  # 获取可学习参数

    def get_extra_parameters(self):  # 统计参数个数
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)  # 学生网络前向
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)  # 教师网络前向

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)  # 硬损失
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        loss_feat = self.feat_loss_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict
