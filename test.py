
import torch
from torch import nn
import random
import torch.nn.functional as F
import numpy as np
# a = torch.tensor([1,2,3,4], dtype = torch.float32)
# c = F.softmax(a, dim = 0)
# b = a.numpy()
# b = b.tolist()
# print(c)

# actions = "AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_UA AIR_UB BACK_JUMP BACK_STEP CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD DASH FOR_JUMP FORWARD_WALK JUMP NEUTRAL STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD THROW_A THROW_B"
# action_list = list(actions.split(" "))
# print(0.99*0.999**400)
#
# t = torch.tensor([[1,2,3],[2,56,6]])
# print(torch.argmax(t,1))
# print(random.choice(range(3)))


for i in range(30):
    if (i+1) % 2 == 0:
        print(i)
print(0.4*0.999**1000)