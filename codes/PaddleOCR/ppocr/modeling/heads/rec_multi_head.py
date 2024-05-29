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

import math
from re import I
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

from ppocr.modeling.necks.rnn import Im2Seq, EncoderWithRNN, EncoderWithFC, SequenceEncoder, EncoderWithSVTR
from .rec_ctc_head import CTCHead
from .rec_sar_head import SARHead
from .rec_nrtr_head import Transformer
import copy

class FCTranspose(nn.Layer):
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias_attr=False)

    def forward(self, x):
        if self.only_transpose:
            return x.transpose([0, 2, 1])
        else:
            return self.fc(x.transpose([0, 2, 1]))


class MultiHead(nn.Layer):
    """ 현재 head 종류는 3개 밖에 없어보임
        이때 SARHead와 NRTRHead는 둘 중 하나만 사용하는 것 같으며
        CTC loss는 무조건 사용인 듯
        또한 SARHead와 NRTRHead는 학습 할 때만 사용하는 것 같음
        보조 도구로 쓰이는 것이며
        실제 추론시에는 CTCLoss만 가지고 하는 듯 
    """
    
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop('head_list')

        self.gtc_head = 'sar'
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == 'SARHead':
                # sar head
                sar_args = self.head_list[idx][name]
                self.sar_head = eval(name)(in_channels=in_channels, \
                    out_channels=out_channels_list['SARLabelDecode'], **sar_args)
            elif name == 'NRTRHead':
                gtc_args = self.head_list[idx][name]
                max_text_length = gtc_args.get('max_text_length', 25)
                nrtr_dim = gtc_args.get('nrtr_dim', 256)
                num_decoder_layers = gtc_args.get('num_decoder_layers', 4)
                self.before_gtc = nn.Sequential(
                    nn.Flatten(2), FCTranspose(in_channels, nrtr_dim))
                self.gtc_head = Transformer(
                    d_model=nrtr_dim,
                    nhead=nrtr_dim // 32,
                    num_encoder_layers=-1,
                    beam_size=-1,
                    num_decoder_layers=num_decoder_layers,
                    max_len=max_text_length,
                    dim_feedforward=nrtr_dim * 4,
                    out_channels=out_channels_list['NRTRLabelDecode'])
            elif name == 'CTCHead':
                # ctc neck
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx][name]['Neck']
                encoder_type = neck_args.pop('name')
                
                # print(in_channels)
                # print(encoder_type)
                # print(neck_args)
                # exit()
                self.ctc_encoder = SequenceEncoder(in_channels=in_channels, \
                    encoder_type=encoder_type, **neck_args)
                # ctc head
                head_args = self.head_list[idx][name]['Head']

                self.ctc_head = eval(name)(in_channels=self.ctc_encoder.out_channels, \
                    out_channels=out_channels_list['CTCLabelDecode'], **head_args)
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiHead yet'.format(name))


    def forward(self, x, targets=None):
        
        ctc_encoder = self.ctc_encoder(x)
        # head_out["test"] = ctc_encoder
        
        # y = x[:, :64, :, :]
        # y = paddle.squeeze(y)
        # ctc_encoder = paddle.transpose(y, [0, 2, 1])
        if targets != None:
            ctc_input = [targets[name] for name in ["label_ctc", "label_sar", "length", "valid_ratio"]]
        else:
            ctc_input = None
        ctc_input = None ############################################################################################### 이래도 되네??
        ctc_out = self.ctc_head(ctc_encoder, ctc_input)
        
        # print("3")
        # print(ctc_out.shape)
        head_out = dict()
        head_out['ctc'] = ctc_out
        head_out['ctc_neck'] = ctc_encoder
        
        # eval mode
        if not self.training:
            return ctc_out
        
        if self.gtc_head == 'sar':
            # print("AA")
            # print("2")
            # print(targets)
            # print("2")
            # print(targets[1:])
            if targets != None:
                sar_input = [targets[name] for name in ["label_sar", "length", "valid_ratio"]]
            else:
                sar_input = None
            sar_out = self.sar_head(x, sar_input)
            head_out['sar'] = sar_out
        else:
            if targets != None:
                gtc_input = [targets[name] for name in ["label_sar", "length", "valid_ratio"]]
            else:
                sar_input = None
            gtc_out = self.gtc_head(self.before_gtc(x), gtc_input)
            head_out['nrtr'] = gtc_out
        # exit()
        return head_out

class MultiHead_Grapheme(nn.Layer):
    """ 현재 head 종류는 3개 밖에 없어보임
        이때 SARHead와 NRTRHead는 둘 중 하나만 사용하는 것 같으며
        CTC loss는 무조건 사용인 듯
        또한 SARHead와 NRTRHead는 학습 할 때만 사용하는 것 같음
        보조 도구로 쓰이는 것이며
        실제 추론시에는 CTCLoss만 가지고 하는 듯 
    """
    
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        
        self.handling_grapheme = kwargs["handling_grapheme"]
        # self.handling_grapheme = ["first", "second"]
        # out_channels_list = kwargs["out_channels_list_test"]
        
        # self.head_dict = {grapheme:MultiHead(in_channels, out_channels_list[grapheme], **copy.deepcopy(kwargs)) for grapheme in self.handling_grapheme}
        
        self.head_dict = dict()
        for i, grapheme in enumerate(self.handling_grapheme):
            channel = {k:v[i] for k, v in out_channels_list.items()}
            head = MultiHead(in_channels, channel, **copy.deepcopy(kwargs))
            setattr(self, f"{grapheme}_head", head) # 이렇게 직접 속성으로 있어야 nn.Layer로 탐색됨
            self.head_dict[grapheme] = head
        # self.head_dict = dict()
        # if "first" in self.handling_grapheme:
        #     self.first_head = MultiHead(in_channels, out_channels_list["first"], **copy.deepcopy(kwargs))
            
        #     self.head_dict["first"] = self.first_head
        # if "second" in self.handling_grapheme:
        #     self.second_head = MultiHead(in_channels, out_channels_list["second"], **copy.deepcopy(kwargs))
        #     self.head_dict["second"] = self.second_head
            
        # if "third" in self.handling_grapheme:
        #     self.third_head = MultiHead(in_channels, out_channels_list["third"], **copy.deepcopy(kwargs))
        #     self.head_dict["third"] = self.third_head
        # if "origin" in self.handling_grapheme:
        #     self.origin_head = MultiHead(in_channels, out_channels_list["origin"], **copy.deepcopy(kwargs))            
        #     self.head_dict["origin"] = self.origin_head
    
    def forward(self, x, targets=None):
        def get_targets(targets, grapheme):
            if targets == None:
                return None
            else:
                return {
                    ""
                    "label_ctc": targets[f"{grapheme}_label"]["label_ctc"],
                    "label_sar": targets[f"{grapheme}_label"]["label_sar"],
                    "length": targets[f"{grapheme}_label"]["length"],
                    "valid_ratio": targets["valid_ratio"]
                }
        head_out = dict()
        for grapheme in self.handling_grapheme:
            head_out[grapheme] = self.head_dict[grapheme](x, get_targets(targets, grapheme))        
        return head_out
        
        
class MultiHead_Grapheme2(nn.Layer):
    """ 현재 head 종류는 3개 밖에 없어보임
        이때 SARHead와 NRTRHead는 둘 중 하나만 사용하는 것 같으며
        CTC loss는 무조건 사용인 듯
        또한 SARHead와 NRTRHead는 학습 할 때만 사용하는 것 같음
        보조 도구로 쓰이는 것이며
        실제 추론시에는 CTCLoss만 가지고 하는 듯 
    """
    
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.handling_grapheme = kwargs["handling_grapheme_test"]
        self.handling_grapheme = ["first"]
        out_channels_list = kwargs["out_channels_list_test"]
        
        self.head_dict = dict()

        for i, grapheme in enumerate(self.handling_grapheme):
            # channels = {k:v[i] for k, v in out_channels_list.items()}
            channels = out_channels_list[grapheme]
            
            self.head_dict[grapheme] = MultiHead(in_channels, channels, **kwargs)
            # self.head_dict[grapheme] = MultiHead(in_channels, channels, **copy.deepcopy(kwargs))

    def forward(self, x, targets=None):
        # temp = targets["first_label"]
        # del targets["first_label"]
        # targets.update(temp)
        
        
        # def get_targets(targets, grapheme):
        #     if targets == None:
        #         return None
        #     else:
        #         return {
        #             ""
        #             "label_ctc": targets[f"{grapheme}_label"]["label_ctc"],
        #             "label_sar": targets[f"{grapheme}_label"]["label_sar"],
        #             "length": targets[f"{grapheme}_label"]["length"],
        #             "valid_ratio": targets["valid_ratio"]
        #         }
        
        head_out = dict()        
        for grapheme in self.handling_grapheme:
            head_out[grapheme] = self.head_dict[grapheme](x, targets)
            # head_out[grapheme] = self.head_dict[grapheme](x, targets = get_targets(targets, grapheme))
            # head_out[grapheme] = self.head_dict[grapheme](x, targets = None)

        

        return head_out
        # return head_out
