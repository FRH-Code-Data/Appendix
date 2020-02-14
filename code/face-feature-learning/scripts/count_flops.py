"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import torch

from model.common import models_backbones, models_landmarks
from ptflops import get_model_complexity_info


def main():
    """Runs flops counter"""
    parser = argparse.ArgumentParser(description='Evaluation script for Face Recognition in PyTorch')
    parser.add_argument('--embed_size', type=int, default=128, help='Size of the face embedding.')
    parser.add_argument('--model', choices=list(models_backbones.keys()) + list(models_landmarks.keys()), type=str,
                        default='rmnet')
    args = parser.parse_args()

    with torch.no_grad():
        if args.model in models_landmarks.keys():
            model = models_landmarks[args.model]()
        else:
            model = models_backbones[args.model](embedding_size=args.embed_size, feature=True)

        flops, params = get_model_complexity_info(model, model.get_input_res(),
                                                  as_strings=True, print_per_layer_stat=True)
        print('Flops:  {}'.format(flops))
        print('Params: {}'.format(params))


if __name__ == '__main__':
    main()
