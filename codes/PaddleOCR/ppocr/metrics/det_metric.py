# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['DetMetric', 'DetFCEMetric']

from .eval_det_iou import DetectionIoUEvaluator


class DetMetric(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        '''
       batch: a list produced by dataloaders. # 정답
           image: np.ndarray  of shape (N, C, H, W).
           ratio_list: np.ndarray  of shape(N,2) => witdh, height
           polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.  => K개의 텍스트에 대한 박스인 듯
           ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not. => 신뢰도인가??
       preds: a list of dict produced by post process # 예측
            points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
            confidences: np.ndarray of shape (N, K)
       '''
        ############################################## 근데 어떻게 추론 개수가 정확히 gt 개수와 똑같지?.... 앞에서 사전에 정리를 했나?..
        # 아마 polygon의 K와 preds의 K가 같은 값을 의미하는게 아닌 듯
       
        gt_polyons_batch = batch[2] # polygon
        ignore_tags_batch = batch[3] # ignore_tag
        # detection result를 assess할 땐 이것들만 필요하긴 하지
        
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch, # 예측과 정답을 매핑해서
                                                 ignore_tags_batch):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)] # 각 샘플 내에 여러 개의 텍스트가 있을 수 있지
            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': ''
            } for det_polyon in pred['points']] # prediction에 points 부분만 사용
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list) # 두 변수는 points, text 값을 갖는 딕셔너리
            self.results.append(result)

    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """

        metrics = self.evaluator.combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = []  # clear results


class DetFCEMetric(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        '''
       batch: a list produced by dataloaders.
           image: np.ndarray  of shape (N, C, H, W).
           ratio_list: np.ndarray  of shape(N,2)
           polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
           ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
       preds: a list of dict produced by post process
            points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
       '''
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]

        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch,
                                                 ignore_tags_batch):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': '',
                'score': score
            } for det_polyon, score in zip(pred['points'], pred['scores'])]

            for score_thr in self.results.keys():
                det_info_list_thr = [
                    det_info for det_info in det_info_list
                    if det_info['score'] >= score_thr
                ]
                result = self.evaluator.evaluate_image(gt_info_list,
                                                       det_info_list_thr)
                self.results[score_thr].append(result)

    def get_metric(self):
        """
        return metrics {'heman':0,
            'thr 0.3':'precision: 0 recall: 0 hmean: 0',
            'thr 0.4':'precision: 0 recall: 0 hmean: 0',
            'thr 0.5':'precision: 0 recall: 0 hmean: 0',
            'thr 0.6':'precision: 0 recall: 0 hmean: 0',
            'thr 0.7':'precision: 0 recall: 0 hmean: 0',
            'thr 0.8':'precision: 0 recall: 0 hmean: 0',
            'thr 0.9':'precision: 0 recall: 0 hmean: 0',
            }
        """
        metrics = {}
        metrics_list = []
        hmean = 0
        for score_thr in self.results.keys():
            metric = self.evaluator.combine_results(self.results[score_thr])
            metrics_list.append(metric)
            # for key, value in metric.items():
            #     metrics['{}_{}'.format(key, score_thr)] = value
            metric_str = 'precision:{:.5f} recall:{:.5f} hmean:{:.5f}'.format(
                metric['precision'], metric['recall'], metric['hmean'])
            metrics['thr {}'.format(score_thr)] = metric_str
            hmean = max(hmean, metric['hmean'])
        
        max_precision = 0
        pre_recall = 1
        AP = 0
        for v in metrics_list:
            precision = v["precision"]
            recall = v["recall"]
            delta_recall = pre_recall-recall
            AP += max_precision*delta_recall
            max_precision = max(max_precision, precision)
            pre_recall = recall
        print(f"AP={AP}")
            
        metrics['hmean'] = hmean

        self.reset()
        
        
        
        return metrics

    def reset(self):
        self.results = {0.1*th:[] for th in range(0, 10)}
        # self.results = {
        #     0.3: [],
        #     0.4: [],
        #     0.5: [],
        #     0.6: [],
        #     0.7: [],
        #     0.8: [],
        #     0.9: []
        # }  # clear results
