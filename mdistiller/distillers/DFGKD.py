import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student/temperature, dim = 1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim = 1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def kd_loss_tc(logits_child1, logits_teacher, temperature):
    log_pred_child1 = F.log_softmax(logits_child1 / temperature, dim = 1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim = 1)
    loss_kd_tc = F.kl_div(log_pred_child1, pred_teacher, reduction="none").sum(1).mean()
    loss_kd_tc *= temperature**2
    return loss_kd_tc

def kd_loss_sc(logits_child2, logits_student, temperature):
    log_pred_child2 = F.log_softmax(logits_child2 / temperature, dim = 1)
    pred_student = F.softmax(logits_student / temperature, dim = 1)
    loss_kd_sc = F.kl_div(log_pred_child2, pred_student, reduction="none").sum(1).mean()
    loss_kd_sc *= temperature**2
    return loss_kd_sc   

def tech_dif(child1, child2, temperature):
    return F.kl_div(F.softmax(child1/temperature, dim = 1), F.softmax(child2/temperature, dim = 1))

class DFGKD(Distiller):
    """Using our method to transfer the knowledge of teacher"""

    def __init__(self, student, teacher, child1, child2, cfg):
        super(DFGKD,self).__init__(student, teacher, child1, child2)
        self.child1 = child1
        self.child2 = child2
        self.temperature = cfg.TECH_KD.TEMPERATURE
        self.ce_loss_weight = cfg.TECH_KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.TECH_KD.LOSS.KD_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.process_weight = cfg.DFGKD.PROCESS_WEIGHT
        #self.feat_loss_weight = 2
        #增加hint
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        #设置学生与教师之间的回归器
        self.conv_reg1 = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
        #设置child与教师之间的回归器
        self.conv_reg2 = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
    def forward_train(self, image, target, **kwargs):
        logits_student, feat_student = self.student(image)
        logits_child1, feat_child1 = self.child1(image)
        logits_child2, feat_child2 = self.child2(image)
        # #self.temperature = self.temperature - ((kwargs["epoch"] - 1)/240)*(self.temperature - 1)
        with torch.no_grad():
            logits_teacher, feat_teacher =self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature)
        
    

        #给child增加硬损失
        loss_kd_tc = self.ce_loss_weight * F.cross_entropy(logits_child1, target) + self.kd_loss_weight * kd_loss_tc(logits_child1, logits_teacher, self.temperature)
        # loss_kd_tc2 = self.ce_loss_weight * F.cross_entropy(logits_child2, target) + self.kd_loss_weight * kd_loss_tc(logits_child2, logits_teacher, self.temperature)
        
        #loss_kd_sc = self.kd_loss_weight * F.cross_entropy(logits_child2, target) + self.kd_loss_weight * kd_loss_sc(logits_child2, logits_student, self.temperature) 
        #loss_kd_cs = self.kd_loss_weight * F.cross_entropy(logits_student, target) + self.kd_loss_weight * kd_loss_sc(logits_student, logits_child2, self.temperature)
        loss_kd_cs = self.process_weight*(self.kd_loss_weight * F.cross_entropy(logits_student, target) + self.kd_loss_weight * kd_loss_sc(logits_student, logits_child1, self.temperature))
        loss_kd_sc = self.kd_loss_weight * F.cross_entropy(logits_child1, target) + self.kd_loss_weight * kd_loss_sc(logits_child1, logits_student, self.temperature)
        #loss_kd_cc = self.kd_loss_weight * F.cross_entropy(logits_child2, target) + self.kd_loss_weight * kd_loss_sc(logits_child2, logits_child1, self.temperature)
        #loss_kd_cc += self.kd_loss_weight * F.cross_entropy(logits_child1, target) + self.kd_loss_weight * kd_loss_sc(logits_child1, logits_child2, self.temperature)

        #tech_dif = F.kl_div(F.softmax(logits_child2/self.temperature, dim = 1), F.softmax(logits_child1/self.temperature, dim = 1))


        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_kd_tc": loss_kd_tc,
            "loss_kd_cs": loss_kd_cs,
            "loss_kd_sc": loss_kd_sc,
 
        }

        return logits_student, losses_dict

         

         
