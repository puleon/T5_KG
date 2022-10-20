from base64 import decode
import torch

from transformers import AutoConfig, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration

from double_t5 import DoubleT5ForConditionalGeneration


tokenizer = AutoTokenizer.from_pretrained('t5-small')

# Do it to be able to use from_pretrained
# To be able to use from_pretrained you need pytorch_model.bin, config.json and state_dict
# config = AutoConfig.from_pretrained('t5-small')
# model = DoubleT5ForConditionalGeneration(config)
# model.save_pretrained('/home/pugachev/github/T5_KG/pretrained_models/double_t5-small_model')

# triplet_model = T5ForConditionalGeneration.from_pretrained("/home/pugachev/github/T5_KG/trained_models/t5_trex_pretrain_padtomaxlenF/checkpoint-128919")

decoder_input_ids_for_generation = tokenizer(["The British Information Commissioner 's Office invites Web users to locate its address using Google Maps .",
 "Tushar Gandhi said the Australian - born tycoon would be arrested if he visited Bombay or New Delhi again ."],
  padding=True, return_tensors="pt").input_ids

input_ids = tokenizer(["The British Information Commissioner.",
 "Tushar Gandhi."],
  padding=True, return_tensors="pt").input_ids


# outputs['decoder_hidden_states'] = (out_seq_len wo start or end token, num_layers+emb_layer (7), batch_size, 1, emb_dim) 
# triplet_outputs = triplet_model.generate(input_ids, return_dict_in_generate=True, output_hidden_states=True)
# print(tokenizer.decode(triplet_outputs[0][0], skip_special_tokens=True))

# triplet_outputs = triplet_outputs['decoder_hidden_states'] 

# decoder_hidden_states = []
# for j in range(len(triplet_outputs[0][0])):
#     decoder_hidden_states_per_batch = []
#     for i in range(len(triplet_outputs)):
#         decoder_hidden_states_per_batch.append(triplet_outputs[i][-1][j][0])
#     decoder_hidden_states.append(torch.stack(decoder_hidden_states_per_batch))
# decoder_hidden_states = torch.stack(decoder_hidden_states)

state_dict = torch.load('/home/pugachev/github/T5_KG/pretrained_models/double_t5-small_state_dict')
# 1st way
model = DoubleT5ForConditionalGeneration.from_pretrained('/home/pugachev/github/T5_KG/pretrained_models/double_t5-small_model',
state_dict=state_dict)
model.triplet_model = T5ForConditionalGeneration.from_pretrained("/home/pugachev/github/T5_KG/trained_models/t5_trex_pretrain_padtomaxlenF/checkpoint-128919")

# model.triplet_decoder_hidden_states = decoder_hidden_states

outputs = model.generate(input_ids, return_dict_in_generate=True, output_hidden_states=True,
 decoder_input_ids_for_generation=decoder_input_ids_for_generation)
print(tokenizer.decode(outputs[0][0], skip_special_tokens=True))

#2nd way (is wrong, why??)
# model = DoubleT5ForConditionalGeneration(config)
# model.load_state_dict(state_dict)


input_ids = tokenizer(["The <extra_id_0> walks in <extra_id_1> park", "The <extra_id_0> walks in <extra_id_1> park"], return_tensors="pt").input_ids
labels = tokenizer(["<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", "<extra_id_0> cute dog <extra_id_1> the <extra_id_2>"], return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=labels, decoder_input_ids_for_generation=decoder_input_ids_for_generation)
loss = outputs.loss
logits = outputs.logits
print(loss)
print(logits)