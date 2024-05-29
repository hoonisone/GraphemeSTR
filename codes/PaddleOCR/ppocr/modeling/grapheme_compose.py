from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

import math
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

class GraphemeCompose(nn.Layer):
    def __init__(self,
                 first_num,
                 second_num,
                 third_num,
                 char_num):
        super(GraphemeCompose, self).__init__()
        
        in_channel = first_num+second_num+third_num
        out_channel = len(char_num)
        
        self.linear1 = nn.Linear(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2D(out_channel)
        self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(out_channel, out_channel)

    def forward(self, f, targets=None):
        out = self.linear1(f)
        out = self.bn1(self.bn1(out))
        out = self.relu1(out)
        out = self.linear1(out)
        
        return out