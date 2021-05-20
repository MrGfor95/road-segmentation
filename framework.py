import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import cv2
import numpy as np
from metrics import Evaluator
from loss import dice_bce_loss,edge_loss,DualTaskLoss
class MyFrame():
    def __init__(self, net,lr=2e-4, evalmode = False,batchsize=1):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.log_vars = nn.Parameter(torch.zeros((2))).cuda()
        self.segloss = dice_bce_loss()
        self.edgeloss=edge_loss()
        # self.Dualloss=DualTaskLoss()
        # self.edgeattention=ImageBasedCrossEntropyLoss2d(1)
        self.old_lr = lr
        # self.confusion_matrix = np.zeros((2,) * 2)
        self.batchsize=batchsize
        # self.evaluator = Evaluator(2)
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None,edge_batch=None,img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.edge = edge_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = torch.Tensor(img).cuda()
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self, volatile=False):
        self.img = self.img.cuda()
        if self.mask is not None:
            self.mask = self.mask.cuda()
            self.edge=self.edge.cuda()
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred,prededge = self.net.forward(self.img)
        # pred=F.log_softmax(pred, dim=1)
        # loss1 = self.segloss(self.mask, pred)
        precision1 = torch.exp(-self.log_vars[0]).cuda()
        loss1 = torch.sum(precision1 * (self.segloss(self.mask, pred)) ** 2. + self.log_vars[0], -1)
        # loss2 = self.edgeloss(prededge,self.edge)
        precision2 = torch.exp(-self.log_vars[1]).cuda()
        loss2= torch.sum(precision2 * (self.edgeloss(prededge,self.edge)) ** 2. + self.log_vars[1], -1)
        # loss3 = self.Dualloss(pred,self.mask.long(),128)
        # loss4 = self.edge_attention(pred,self.mask,self.labeledge)
        loss = loss1 + loss2
        # loss=loss1+loss2*20
        loss.backward()
        self.optimizer.step()
        return loss.data[0],loss1,loss2

    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        #print >> mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr)
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr


    # def getedge(self):
    #     mask=self.mask.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
    #     canny = np.zeros((self.batchsize, 1, 1024, 1024))
    #     for i in range(self.batchsize):
    #         canny[i] = cv2.Canny(mask[i][0], 10, 100)
    #     canny = torch.from_numpy(canny).cuda().float()
    #     return canny

    # def edge_attention(self,input, target, edge):
    #     n, c, h, w = input.size()
    #     filler = torch.ones_like(target) * 255
    #     return self.edgeattention(input,torch.where(edge.max(1)[0] > 0.8, target, filler).long())


# class ImageBasedCrossEntropyLoss2d(nn.Module):
#
#     def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
#                  norm=False, upper_bound=1.0):
#         super(ImageBasedCrossEntropyLoss2d, self).__init__()
#         self.num_classes = classes
#         self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
#         self.norm = norm
#         self.upper_bound = upper_bound
#         self.batch_weights = False
#
#     def calculateWeights(self, target):
#         hist = np.histogram(target.flatten(), range(
#             self.num_classes + 1), normed=True)[0]
#         if self.norm:
#             hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
#         else:
#             hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
#         return hist
#
#     def forward(self, inputs, targets):
#         target_cpu = targets.data.cpu().numpy()
#         if self.batch_weights:
#             weights = self.calculateWeights(target_cpu)
#             self.nll_loss.weight = torch.Tensor(weights).cuda()
#
#         loss = 0.0
#         for i in range(0, inputs.shape[0]):
#             if not self.batch_weights:
#                 weights = self.calculateWeights(target_cpu[i])
#                 self.nll_loss.weight = torch.Tensor(weights).cuda()
#
#             loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
#                                   targets[i].unsqueeze(0))
#         return loss
