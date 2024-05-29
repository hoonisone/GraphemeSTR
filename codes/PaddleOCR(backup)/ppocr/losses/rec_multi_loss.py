# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn

from .rec_ctc_loss import CTCLoss
from .rec_sar_loss import SARLoss
from .rec_nrtr_loss import NRTRLoss


class MultiLoss(nn.Layer):
    # Combine loss와 무슨 차이가 있을까?
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_funcs = {} # combined loss 처럼 여러 loss function을 묶는 것 같은데..
        self.loss_list = kwargs.pop('loss_config_list')
        
        # 내가 보는 config의 경우
        """ 
            loss_config_list:
              - CTCLoss:
              - SARLoss:
        """
        # 로 되어있음 
        # -가 붙어있으면 하나의 항목이고 Key:None으로 항목이 하나인 dict와 연결
        # 즉 loss_list는 [{CTCLoss:None}, {SARLoss:None}]가 된다.
                
        # dict.pop(key) => key에 해당하는 항목을 제거 하며 대응되는 value를 반환함
        
        
        self.weight_1 = kwargs.get('weight_1', 1.0)
        self.weight_2 = kwargs.get('weight_2', 1.0)
        #dict.get(key)는 값이 없는 경우 None을 반환함, None 대신 대체 값 사용 가능
        # forward부분을 보면 weight 1, 2가 순서대로 적용되는게 아니라 특정 loss마다 적용되는 weight이 있고 이를 그냥 순서대로 일컫는 것임
        
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                    # loss_config_list안에 있는 param를 밖으로 꺼내는 역할 수행
                    # 이런 과정을 거치는 이유는??
                    # kwargs안에 loss_config_list를 확인하면 param이 있을 것이다.
                    # 그렇지만 kwargs에 param을 또 다시 두어 사용하는 것은 loss_config_list말고도 직접 MultiLoss 호출 시 params를 지정할 수 있게 하기 위함이다.
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss
                # param에 맡게 loss를 생성하고 


    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0
        # batch [image, label_ctc, label_sar, length, valid_ratio]
        for name, loss_func in self.loss_funcs.items():
            if name == 'CTCLoss':
                loss = loss_func(predicts['ctc'],
                                 batch[:2] + batch[3:])['loss'] * self.weight_1
            elif name == 'SARLoss':
                loss = loss_func(predicts['sar'],
                                 batch[:1] + batch[2:])['loss'] * self.weight_2
            elif name == 'NRTRLoss':
                loss = loss_func(predicts['nrtr'],
                                 batch[:1] + batch[2:])['loss'] * self.weight_2
            else:
                # MultiLoss에서 지원하는 Loss는 CTCLoss, SARLoss, NRTRLoss가 전부이군
                raise NotImplementedError(
                    '{} is not supported in MultiLoss yet'.format(name))
            self.total_loss[name] = loss
            total_loss += loss
        self.total_loss['loss'] = total_loss
        # 결과적으로 로스가 여러개 이기 때문에 dict로 전달하며, total loss를 계산해서 넘겨주네
        return self.total_loss
    

class MultiLoss_Grapheme(nn.Layer):
    def __init__(self, handling_grapheme, **kwargs):
        super().__init__()
        self.handling_grapheme = handling_grapheme
        
        def extract_kwargs(kwargs, idx):
            import copy
            kwargs = copy.deepcopy(kwargs)
            kwargs["loss_config_list"][1]["SARLoss"]["ignore_index"] = kwargs["loss_config_list"][1]["SARLoss"]["ignore_index"][idx]
            return kwargs
        
        self.multiloss_dict = {
            grapheme:MultiLoss(**extract_kwargs(kwargs, i))
            for i, grapheme in enumerate(self.handling_grapheme)    
        }
        
    def forward(self, predicts, batch):
        def get_batch(batch, idx):
            return [
                batch[0],
                batch[idx+1]["label_ctc"],
                batch[idx+1]["label_sar"],
                batch[idx+1]["length"],
                batch[5]
            ]
            
        
        total_loss = {grapheme: self.multiloss_dict[grapheme](predicts[grapheme], get_batch(batch, i)) for i, grapheme in enumerate(self.handling_grapheme)}
        total_loss["loss"] = sum([total_loss[grapheme]["loss"] for grapheme in self.handling_grapheme])/len(self.handling_grapheme)
        
        return total_loss
