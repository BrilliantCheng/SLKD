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

class Tech_KD(Distiller):
    """Using our method to transfer the knowledge of teacher"""

    def __init__(self, student, teacher, child1, child2, cfg):
        super(Tech_KD,self).__init__(student, teacher, child1, child2)
        # self.child1 = child1
        # self.child2 = child2
        self.temperature = cfg.TECH_KD.TEMPERATURE
        self.ce_loss_weight = cfg.TECH_KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.TECH_KD.LOSS.KD_WEIGHT
        self.process_weight = cfg.TECH_KD.PROCESS_WEIGHT
        self.ce_loss_weight_cs = cfg.TECH_KD.LOSS.CE_WEIGHT_CS
        self.kd_loss_weight_cs = cfg.TECH_KD.LOSS.KD_WEIGHT_CS


    def forward_train(self, image, target, **kwargs):
        logits_student, feat_student = self.student(image)
        logits_child1, feat_child1 = self.child1(image)
        logits_child2, feat_child2 = self.child2(image)

        logits_child = 0.5*logits_child1 + 0.5*logits_child2

        with torch.no_grad():
            logits_teacher, feat_teacher =self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature)
        
    

        #给child增加硬损失
        loss_kd_tc1 = self.ce_loss_weight * F.cross_entropy(logits_child1, target) + self.kd_loss_weight * kd_loss(logits_child1, logits_teacher, self.temperature)
        loss_kd_tc2 = self.ce_loss_weight * F.cross_entropy(logits_child2, target) + self.kd_loss_weight * kd_loss(logits_child2, logits_teacher, self.temperature)
        loss_kd_tc = loss_kd_tc1 + loss_kd_tc2

      


        #增加
        loss_kd_cs = self.process_weight * (self.ce_loss_weight_cs * F.cross_entropy(logits_student, target) + self.kd_loss_weight_cs * kd_loss(logits_student, logits_child, self.temperature))






        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_kd_tc": loss_kd_tc,
            "loss_kd_cs": loss_kd_cs,
        }


        return logits_student, losses_dict
         
