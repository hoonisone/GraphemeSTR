# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import unicode_literals

import copy

__all__ = ['build_post_process']

from .db_postprocess import DBPostProcess, DistillationDBPostProcess # @ 하나의 알고리즘에도 둘 이상의 프로세서 방식이 있구나
from .east_postprocess import EASTPostProcess
from .sast_postprocess import SASTPostProcess
from .fce_postprocess import FCEPostProcess
from .rec_postprocess import CTCLabelDecode, AttnLabelDecode, SRNLabelDecode, \
    DistillationCTCLabelDecode, NRTRLabelDecode, SARLabelDecode, \
    SEEDLabelDecode, PRENLabelDecode, ViTSTRLabelDecode, ABINetLabelDecode, \
    SPINLabelDecode, VLLabelDecode, RFLLabelDecode, SATRNLabelDecode, CTCLabelDecode_Grapheme
from .cls_postprocess import ClsPostProcess
from .pg_postprocess import PGPostProcess
from .vqa_token_ser_layoutlm_postprocess import VQASerTokenLayoutLMPostProcess, DistillationSerPostProcess
from .vqa_token_re_layoutlm_postprocess import VQAReTokenLayoutLMPostProcess, DistillationRePostProcess
from .table_postprocess import TableMasterLabelDecode, TableLabelDecode
from .picodet_postprocess import PicoDetPostProcess
from .ct_postprocess import CTPostProcess
from .drrg_postprocess import DRRGPostprocess
from .rec_postprocess import CANLabelDecode


def build_post_process(config, global_config=None):
    support_dict = [
        'DBPostProcess', 'EASTPostProcess', 'SASTPostProcess', 'FCEPostProcess',
        'CTCLabelDecode', 'AttnLabelDecode', 'ClsPostProcess', 'SRNLabelDecode',
        'PGPostProcess', 'DistillationCTCLabelDecode', 'TableLabelDecode',
        'DistillationDBPostProcess', 'NRTRLabelDecode', 'SARLabelDecode',
        'SEEDLabelDecode', 'VQASerTokenLayoutLMPostProcess',
        'VQAReTokenLayoutLMPostProcess', 'PRENLabelDecode',
        'DistillationSARLabelDecode', 'ViTSTRLabelDecode', 'ABINetLabelDecode',
        'TableMasterLabelDecode', 'SPINLabelDecode',
        'DistillationSerPostProcess', 'DistillationRePostProcess',
        'VLLabelDecode', 'PicoDetPostProcess', 'CTPostProcess',
        'RFLLabelDecode', 'DRRGPostprocess', 'CANLabelDecode',
        'SATRNLabelDecode', "CTCLabelDecode_Grapheme"
    ]
    # pre, post processing의 경우 설정에 민감하게 바뀐다.
    # 알고리즘이 다른 경우 입력 출력 형태가 다를 수 있어 pre, post에서 이를 맞춰주어야 한다.
    # 위에 보면 알고리즘 마다 각각 process들이 있는 것을 알 수 있다.

    if config['name'] == 'PSEPostProcess': # False
        from .pse_postprocess import PSEPostProcess
        support_dict.append('PSEPostProcess')

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    
    module_class = eval(module_name)(**config) # 결국은 eval로 객체를 만들어내고 있네, 그럼 사실 config에 모듈 별로 name만 봐도 어떤걸 쓸 건지 알 수 있겠다.
    return module_class
