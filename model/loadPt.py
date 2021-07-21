import torch
pth=torch.load('/home/zhangyn/zzz/deblur-task/rfdb-icip/lam-distillation-EDVR/pretrained_models/model_best.pt')
print(type(pth))
print(pth.keys())
print(pth['scam.gamma'])
print(pth['lam.gamma'])