from collections import OrderedDict

import torch

from transformers import T5ForConditionalGeneration


model = T5ForConditionalGeneration.from_pretrained('t5-small')
state_dict = model.state_dict()
double_state_dict = OrderedDict()
for k in state_dict:
    if k.startswith('decoder.'):
        if '.layer.0.' in k:
            double_state_dict[k] = state_dict[k]
        elif '.layer.1.' in k:
            double_state_dict[k] = state_dict[k]
            double_state_dict[k.replace('.layer.1.', '.layer.2.')] = state_dict[k]
        elif '.layer.2.' in k:
            double_state_dict[k.replace('.layer.2.', '.layer.3.')] = state_dict[k]
        else:
            double_state_dict[k] = state_dict[k]
    else:
            double_state_dict[k] = state_dict[k]

torch.save(double_state_dict, '/home/pugachev/github/T5_KG/pretrained_models/double_t5-small_state_dict')

print(1)