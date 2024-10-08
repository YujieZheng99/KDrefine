import torch
import torch.nn as nn
import torch.nn.functional as F


# def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
#     # logits_student/teacher = {tensor:(64, 100)};target = {tensor:(64, )}(每张图片的类别1-100)
#     gt_mask = _get_gt_mask(logits_student, target)  # gt_mask = {tensor:(64, 100)(gt的位置,bool类型)
#     other_mask = _get_other_mask(logits_student, target)  # other_mask = {tensor:(64, 100)}(非gt的位置，bool类型)
#     pred_student = F.softmax(logits_student / temperature, dim=1)  # pred_student = {tensor:(64, 100)}(过softmax)
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)  # pred_teacher = {tensor:(64, 100)}(过softmax)
#     pred_student = cat_mask(pred_student, gt_mask, other_mask)  # pred_student = {tensor:(64, 2)}(student的正确得分和错误得分)
#     pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)  # pred_teacher = {tensor:(64, 2)}(teacher的正确得分和错误得分)
#     log_pred_student = torch.log(pred_student)  # log_pred_student = {tensor:(64, 2)}
#     tckd_loss = (
#         F.kl_div(log_pred_student, pred_teacher, size_average=False)
#         * (temperature**2)
#         / target.shape[0]
#     )
#     pred_teacher_part2 = F.softmax(
#         logits_teacher / temperature - 1000.0 * gt_mask, dim=1
#     )  # pred_teacher_part2 = {tensor(64, 100)}(非标签的logits过一遍softmax)
#     log_pred_student_part2 = F.log_softmax(
#         logits_student / temperature - 1000.0 * gt_mask, dim=1
#     )  # pred_student_part2 = {tensor(64, 100)}(非标签的logits过一遍softmax再过一个log)
#     nckd_loss = (
#         F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
#         * (temperature**2)
#         / target.shape[0]
#     )
#     return alpha * tckd_loss + beta * nckd_loss

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    # logits_student/teacher = {tensor:(64, 100)};target = {tensor:(64, )}(每张图片的类别1-100)
    gt_mask = _get_gt_mask(logits_student, target)  # gt_mask = {tensor:(64, 100)(gt的位置,bool类型)
    other_mask = _get_other_mask(logits_student, target)  # other_mask = {tensor:(64, 100)}(非gt的位置，bool类型)
    pred_student = F.softmax(logits_student / temperature, dim=1)  # pred_student = {tensor:(64, 100)}(过softmax)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)  # pred_teacher = {tensor:(64, 100)}(过softmax)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)  # pred_student = {tensor:(64, 2)}(student的正确得分和错误得分)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)  # pred_teacher = {tensor:(64, 2)}(teacher的正确得分和错误得分)
    pt = pred_teacher[:, 0]
    log_pred_student = torch.log(pred_student)  # log_pred_student = {tensor:(64, 2)}
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature**2)
            / target.shape[0]
        )
    tckd_loss = tckd_loss * (temperature**2) / target.shape[0]

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )  # pred_teacher_part2 = {tensor(64, 100)}(非标签的logits过一遍softmax)
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )  # pred_student_part2 = {tensor(64, 100)}(非标签的logits过一遍softmax再过一个log)

    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )

    return nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKDLoss(nn.Module):
    def __init__(self):
        super(DKDLoss, self).__init__()
        self.alpha = 1.0
        self.beta = 8.0
        self.temperature = 4.0
        self.warmup = 20

    def forward(self, logits_student, logits_teacher, target, epoch):
        loss_dkd = min(epoch / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        return loss_dkd
