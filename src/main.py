import torch
import np
import time
from functools import reduce
from torch.autograd import Variable
from cnn import CNNnet
from ds_cnn import DS_CNNnet
from train import train, test
from handle_audio import AudioPreprocessor

audio_manager = AudioPreprocessor()

available_words = ['right', 'eight', 'cat', 
    'tree', 'bed', 'happy', 
    'go', 'dog', 'no', 
    'wow', 'nine', 'left', 
    'stop', 'three', 'sheila', 
    'one', 'bird', 'zero', 'seven', 
    'up', 'marvin', 'two', 'house', 
    'down', 'six', 'yes', 
    'on', 'five', 'off', 'four']

wanted_words = ['on', 'off', 'stop', 
    'down', 'left', 'right',
    'go', 'one', 'two', 'three']

# Make false if not using GPU
IS_CUDA = False

model = DS_CNNnet(len(wanted_words))

if IS_CUDA: 
    model = model.cuda()
    torch.backends.cudnn.benchmark=True

path_to_dataset = '/Users/dsm/Downloads/speech_commands_v0.01'
#path_to_dataset = '/home/utdesign/code/audio_files'

data, labelDictionary = audio_manager.extract_audio_files(path_to_dataset, wanted_words)

training_set, testing_set, validation_set = audio_manager.split_data_set(data, .80, .10, .10)

# needed to reshape/organize testing set: 
#! Not converting it to minibatch, just reorganizing the data
testing_set = audio_manager.feature_extraction(testing_set)
testing_list, testing_label_list = audio_manager.convert_to_minibatches(testing_set, 1)

# needed to reshape/organize validation set: 
validation_set = audio_manager.feature_extraction(validation_set)
validation_list, validation_label_list = audio_manager.convert_to_minibatches(validation_set, 1)

print('Training')
num_epochs = 100
for epoch_num in range(num_epochs):
    # Have to redefine so we don't over write "traning_set":
    train_set = training_set
    train_set = audio_manager.augment_data(train_set)
    train_batch = audio_manager.feature_extraction(train_set)
    
    mini_batch_list, mini_batch_label = audio_manager.convert_to_minibatches(train_batch, 64)
    
    for i, batch in enumerate(mini_batch_list):
        if IS_CUDA: 
            batch, label = (Variable(batch)).cuda(), (Variable(mini_batch_label[i])).cuda()
            train(model, batch, 64, label, 1e-4)
        else: 
            train(model, batch, 64, mini_batch_label[i], 1e-4)

    if epoch_num % 10 == 0: 
        print("Epoch #", epoch_num)
        if IS_CUDA:
            testing_list_cuda, testing_label_list_cuda = (Variable(testing_list)).cuda(), (Variable(testing_label_list)).cuda()
            test(model, testing_list_cuda, testing_label_list_cuda)
        else: 
            test(model, testing_list, testing_label_list)


saved_model_name = reduce((lambda x, y: y + '_' + x), wanted_words )

print("Final validation of model " + saved_model_name +  ":")
if IS_CUDA:
    validation_list_cuda, validation_label_list_cuda = (Variable(validation_list)).cuda(), (Variable(validation_label_list)).cuda()
    final_acc = test(model, testing_list_cuda, testing_label_list_cuda)
else: 
    final_acc = test(model, validation_list, validation_label_list)

time = localtime = time.asctime( time.localtime(time.time()) )
#Externally Record Accuracy when done
text_file = open(saved_model_name + ".txt", "w")
text_file.write("Accuracy: " + final_acc + '\n' + time)
text_file.close()

#Externally Save Model:
torch.save(model, '../saved_models/' + 'ds_cnn_' + saved_model_name)