import argparse
import torch
import os
import numpy as np
from ds_cnn import DS_CNNnet

model = DS_CNNnet(3)
thing = torch.load('/Users/dsm/code/SeniorDesign/SeniorDesign2018/saved_models/ds_cnn_three_two_one')

model.load_state_dict(thing)
model.eval()

my_path = os.path.abspath(os.path.dirname(__file__))
path = '../_quant_weights/' + 'ds_cnn_three_two_one_' + 'weights.h'
weight_path = os.path.join(my_path, path)

open(weight_path, 'w+').close()
o = model.state_dict()
for v in model.state_dict():  
    var_name = str(v)
    var_values = model.state_dict()[v]
    min_value = var_values.min()
    max_value = var_values.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_value),abs(max_value)))))
    dec_bits = 7-int_bits
    # convert to [-128,128) or int8
    var_values = np.round(var_values*2**dec_bits)
    var_name = var_name.replace('/','_')
    var_name = var_name.replace(':','_')
    with open(weight_path,'a') as f:
      f.write('#define '+var_name+' {')
    
    transposed_wts = var_values.flatten()
    with open(weight_path,'a') as f:
      s = ",".join(str(int(x)) for x in transposed_wts)
      f.write( s )
      f.write('}\n')
    # convert back original range but quantized to 8-bits or 256 levels
    var_values = var_values/(2**dec_bits)
    # update the weights in tensorflow graph for quantizing the activations
    # var_values = sess.run(tf.assign(v,var_values))
    print(var_name+' number of wts/bias: '+str(var_values.shape)+\
            ' dec bits: '+str(dec_bits)+\
            ' max: ('+str(var_values.max())+','+str(max_value)+')'+\
            ' min: ('+str(var_values.min())+','+str(min_value)+')')



# bits = 8
# quant_method = 'linear'
# # quantize parameters
# state_dict = model.state_dict()
# state_dict_quant = OrderedDict()

# for k, v in state_dict.items():   
#     state_dict_quant[k] = v
#     bits = 32

#     if quant_method == 'linear':
#         sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=0)
#         v_quant  = quant.linear_quantize(v, sf, bits=bits)
#     elif quant_method == 'log':
#         v_quant = quant.log_minmax_quantize(v, bits=bits)
#     elif quant_method == 'minmax':
#         v_quant = quant.min_max_quantize(v, bits=bits)
#     else:
#         v_quant = quant.tanh_quantize(v, bits=bits)

#     state_dict_quant[k] = v_quant
#     model.load_state_dict(state_dict_quant)

# # quantize forward activation
# model_raw = quant.duplicate_model_with_quant(model, bits=8, overflow_rate=0,
#                                                 counter=20, type=quant_method)
# print(model_raw)




# # print sf
# print(model_raw)
# res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
#     args.type, args.quant_method, args.param_bits, args.bn_bits, args.fwd_bits, args.overflow_rate, acc1, acc5)
# print(res_str)
# with open('acc1_acc5.txt', 'a') as f:
#     f.write(res_str + '\n')