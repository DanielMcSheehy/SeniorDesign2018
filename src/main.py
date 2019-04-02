import torch
import np
import time
import os
from functools import reduce
from torch.autograd import Variable
from cnn import CNNnet
from ds_cnn import DS_CNNnet
from train import train, test
from handle_audio import AudioPreprocessor
from sound_augmentation import load_background_audio

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

# wanted_words = ['on', 'off', 'stop', 
#     'down', 'left', 'right',
#     'go', 'one', 'two', 'three']

wanted_words = ['one', 'two', 'three']
wanted_words.append('silence')
wanted_words.append('unknown')

# Make false if not using GPU
IS_CUDA = False

model = DS_CNNnet(len(wanted_words)) 

if IS_CUDA: 
    model = model.cuda()
    torch.backends.cudnn.benchmark=True

path_to_dataset = '/Users/dsm/Downloads/speech_commands_v0.01'
#path_to_dataset = '/home/utdesign/code/audio_files'

data, labelDictionary = audio_manager.generate_dataset(path_to_dataset, wanted_words, available_words)

training_set, testing_set, validation_set = audio_manager.split_data_set(data, .80, .10, .10)

background_audio = load_background_audio()

# needed to reshape/organize testing set: 
#! Not converting it to minibatch, just reorganizing the data
testing_set = audio_manager.feature_extraction(testing_set)

testing_list, testing_label_list = audio_manager.convert_to_minibatches(testing_set, 1)

# needed to reshape/organize validation set: 
validation_set = audio_manager.feature_extraction(validation_set)
validation_list, validation_label_list = audio_manager.convert_to_minibatches(validation_set, 1)


num_epochs = 1000
learning_rate = 5e-4 # goes to 1e-4 after halfway being done with training

for epoch_num in range(num_epochs):
    print('Training on epoch #', epoch_num, ' time: ', time.asctime( time.localtime(time.time()) ))
    # Have to redefine so we don't over write "training_set":
    train_set = training_set
    # Shuffles data, adds background noise, shifting, and possibly reverb
    train_set = audio_manager.augment_data(train_set, background_audio)

    train_batch = audio_manager.feature_extraction(train_set)
    
    mini_batch_list, mini_batch_label = audio_manager.convert_to_minibatches(train_batch, 100)
    
    for i, batch in enumerate(mini_batch_list):
        if IS_CUDA: 
            batch, label = (Variable(batch)).cuda(), (Variable(mini_batch_label[i])).cuda()
            train(model, batch, 64, label, learning_rate)
        else: 
            train(model, batch, 64, mini_batch_label[i], learning_rate)

    if epoch_num % 3 == 0: 
        print("Testing Epoch #", epoch_num)
        if IS_CUDA:
            testing_list_cuda, testing_label_list_cuda = (Variable(testing_list)).cuda(), (Variable(testing_label_list)).cuda()
            test(model, testing_list_cuda, testing_label_list_cuda)
        else: 
            # Testing against pure testing data set
            test(model, testing_list, testing_label_list)

    # If halfway done through traning reduce learning rate:
    if round(num_epochs/(epoch_num + 1)) == 2: 
        learning_rate = 1e-4


saved_model_name = reduce((lambda x, y: y + '_' + x), wanted_words )

print("Final validation of model " + saved_model_name +  ":")
if IS_CUDA:
    validation_list_cuda, validation_label_list_cuda = (Variable(validation_list)).cuda(), (Variable(validation_label_list)).cuda()
    final_acc = test(model, validation_list_cuda, validation_label_list_cuda)
else: 
    final_acc = test(model, validation_list, validation_label_list)

time = localtime = time.asctime( time.localtime(time.time()) )
#Externally Record Accuracy when done
my_path = os.path.abspath(os.path.dirname(__file__))
path = '../logs/' + saved_model_name + '.txt'
log_path = os.path.join(my_path, path)

text_file = open(log_path, "w+")
text_file.write("Accuracy: " + str(final_acc) + '\n' + str(time))
text_file.close()

#Externally Save Model:
torch.save(model.state_dict(), '../saved_models/' + 'ds_cnn_' + saved_model_name)