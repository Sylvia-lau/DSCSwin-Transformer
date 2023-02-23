import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import  Counter
import numpy as np
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class SoftTargetFocalCB(nn.Module):

    def __init__(self,num_class, labels, loss_type="focal", beta=0.9999, gamma=2.0):
        super(SoftTargetFocalCB, self).__init__()
        self.num_class = num_class
        dict_label = Counter(labels)
        samples_per_cls = []
        for i in range(num_class) :
            if i not in dict_label.keys() :
                samples_per_cls.append(0)
            else :
                samples_per_cls.append(dict_label[i])
        self.samples_per_cls = samples_per_cls
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def focal_loss(self, labels, logits, alpha) :
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if self.gamma == 0.0 :
            modulator = 1.0
        else :
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 +
                                                                                         torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss

    def forward(self, input, target) :
        input, target = input.to('cpu'), target.to('cpu')
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.num_class

        labels_one_hot = target.float()
        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_class)

        if self.loss_type == "focal" :
            cb_loss = self.focal_loss(labels_one_hot, input, weights)
        elif self.loss_type == "sigmoid" :
            cb_loss = F.binary_cross_entropy_with_logits(input=input, target=labels_one_hot, weights=weights)
        elif self.loss_type == "softmax" :
            pred = input.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        # cb_loss = torch.from_numpy(cb_loss)
        return cb_loss