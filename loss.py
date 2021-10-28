import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # dist.addmm_(1, -2, x, y.t())
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    https://github.com/yxgeee/MMT/blob/057e1ea5d3054c9d7e5fa72c727298d8e4c5f668/mmt/loss/triplet.py#L65
	'''

	def __init__(self, margin, normalize_feature=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

	def forward(self, emb, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb = F.normalize(emb)
		mat_dist = euclidean_dist(emb, emb)
		# mat_dist = cosine_dist(emb, emb)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
		assert dist_an.size(0)==dist_ap.size(0)
		y = torch.ones_like(dist_ap)
		loss = self.margin_loss(dist_an, dist_ap, y)
		prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss, prec

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

def CrossEntropy(inputs, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(inputs, target)

def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    target = F.softmax(target, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss

def act_loss(x_list, y_list):
    if(len(x_list)!=len(y_list)):
        raise ValueError("len(x_list)!=len(y_list)")

    loss = 0

    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        x = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
        y = F.normalize(y.pow(2).mean(1).view(y.size(0), -1))
        loss += (x-y).pow(2).mean()

    return loss

# overlapped activation penalty 
def oap_loss(feat_maps_list): #(b, c, h, w)
    p = 0.3
    z_list = []

    for feat in feat_maps_list:
        # feat = F.normalize(feat.pow(2).mean(1).view(feat.size(0), -1)) #(b, h*w)
        feat = feat.pow(2).mean(1).view(feat.size(0), -1)

        sorted, _ = torch.sort(feat, dim=1, descending=True)
        k = sorted[:, int(feat.size(1)*p)].unsqueeze(1) #(b, 1)
        k = k.repeat(1, feat.size(1)) #(b, h*w)

        z = F.sigmoid(feat-k)
        # z = F.sigmoid(feat)
        z_list.append(z)

    # loss = (z_list[0]*z_list[1]*z_list[2]*z_list[3]).sum() / feat.size(0)
    loss = (z_list[0]*z_list[1]).sum() / feat.size(0)
        
    return loss