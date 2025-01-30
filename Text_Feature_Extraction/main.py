from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from transformers import FlaxLlamaModel, GPT2ForSequenceClassification

from transformers import Trainer, TrainingArguments
import os
from helper import *
import torch 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import torch.nn as nn
import sys
import datetime

# check version of python 
print(sys.version)

# get device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if CUDA is available
if torch.cuda.is_available():
    # Print the CUDA version
    print("CUDA Version:", torch.version.cuda)
else:
    print("CUDA is not available.")



# Define the training configuration
# training_config = {
#     'model_name': "microsoft/deberta-base", # praneethvasarla/med-bert, microsoft/deberta-base,"Dr-BERT/DrBERT-4GB-CP-PubMedBERT" "FacebookAI/roberta-base", bert-base-uncased, emilyalsentzer/Bio_ClinicalBERT, dmis-lab/biobert-v1.1, ugaray96/biobert_ncbi_disease_ner
#     'epochs_num': 10 ,
#     'learning_rate': 1e-5,
#     'cv_fold': True,
#     'cv_k': 2,
#     'batch_size_list': [16],
#     'data': 'gpt', # expert, gpt, combined
# }


# bert-base-uncased, FacebookAI/roberta-base, microsoft/deberta-base, 
# emilyalsentzer/Bio_ClinicalBERT, dmis-lab/biobert-v1.1, ugaray96/biobert_ncbi_disease_ner
################### NOT WORKING 
# afmck/testing-llama-tiny"
# microsoft/DialogRPT-updown >> this is GPT2


# good result
training_config = {
    'model_name': "bert-base-uncased", 
    'epochs_num': 15 ,
    'learning_rate': 1e-5,
    'cv_fold': True,
    'cv_k': 4,
    'batch_size_list': [16],
    'data': 'gpt', # expert, gpt, combined
    'dec_scnario': 1,
    'num_trainable_layer': 5
}

# print the training configuration 
# print the configuration
print("\n\nConfiguration:")
for key, value in training_config.items():
    print(f"{key}: {value}")
print("\n\n")

# Set seed for torch
torch.manual_seed(42)
# Set seed for numpy
np.random.seed(42)
# Set seed for random
random.seed(42)



######### load data 
current_dir = os.path.dirname(os.path.abspath(__file__))
dec_scnario = training_config['dec_scnario']
subforlder = "original"

# Define the number of levels you want to move up
levels_to_go_up = 3  # Change this to the number of levels you need to go up
# Move up the directory structure
data_dir = current_dir
for _ in range(levels_to_go_up):
    data_dir = os.path.split(data_dir)[0]

data_dir = os.path.join(data_dir, 'data/wound_data')
data_dir = os.path.join(data_dir, subforlder)
data_loader = DataLoader(data_dir, dec_scnario)

if training_config['data'] == 'expert':
    X, y = data_loader.load_data()
elif training_config['data'] == 'gpt' and dec_scnario == 1:
    X, y = data_loader.load_data_GPT()
elif training_config['data'] == 'combined':
    X, y = data_loader.load_data_combined()


print_ = False
if print_:
    # get len of X and y
    print(f"len of X: {len(X)}")
    print(f"len of y: {len(y)}")

    # print the first 5 elements of X and y 
    for i in range(5):
        print(f"X[{i}]: {X[i]}")
        print(f"y[{i}]: {y[i]}")
        print("\n")


# Split and encode the data
data_preprocessor = DataPreprocessor()
if training_config['cv_fold']:
    cv_k = training_config['cv_k']
    encoded_folds = data_preprocessor.split_encode_kfold(X, y, n_splits=cv_k)
else:
    training_config['cv_k'] = 1
    encoded_folds = data_preprocessor.split_encode_kfold(X, y, n_splits=1)



# define a numpy array with size of (cv_k, epochs_num)
accuracy_mat_tr = np.zeros((training_config['cv_k'], training_config['epochs_num']))    
accuracy_mat_te = np.zeros((training_config['cv_k'], training_config['epochs_num']))



# get current directory and create subdirectory > saved_results 
current_dir = os.path.dirname(os.path.abspath(__file__))
subforlder = "saved_results"
# get current time 
current_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

# directory of saved_results 
saved_results_dir = os.path.join(current_dir, subforlder, current_time)

# Check and create the subfolder
try:
    if not os.path.exists(saved_results_dir):
        os.makedirs(saved_results_dir)
        print(f"Directory created at: {saved_results_dir}")
    else:
        print(f"Directory already exists at: {saved_results_dir}")
except Exception as e:
    print(f"An error occurred while creating the directory: {e}")

# save the configuration in the saved_results_dir
config_file_path = os.path.join(saved_results_dir, 'config.txt')
with open(config_file_path, 'w') as f:
    for key, value in training_config.items():
        f.write(f"{key}: {value}\n")

model_name = training_config['model_name']

# loop through different folds 
for fold, (X_train, X_test, y_train, y_test) in enumerate(encoded_folds):


    # check if in the model_name, there is roberta
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
    elif 'deberta' in model_name:
        tokenizer = DebertaTokenizer.from_pretrained(model_name)
        model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
    elif 'llama' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = FlaxLlamaModel.from_pretrained(model_name, num_labels=3)
    elif 'dialogRPT' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        

    

    # get number of layers in the model
    num_layers = model.config.num_hidden_layers
    num_trainable_layer = training_config['num_trainable_layer'] 

    trainable_layer_list = []
    for i in range(num_trainable_layer):
        trainable_layer_list.append('encoder.layer.%d'%(num_layers - i - 1))

    # freeze all layers acept trainable_layer_list and classifier 
    for name, param in model.named_parameters():
        if 'classifier' not in name and all([layer not in name for layer in trainable_layer_list]):
            param.requires_grad = False
        else:
            print(f"Parameter: {name}, Trainable: {param.requires_grad}")

    # Verify which parameters are trainable
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Trainable: {param.requires_grad}")

    # Tokenize the data
    # output of tokenizer can be used as input to the BERT model 
    X_train_tokens = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_test_tokens = tokenizer(X_test, padding=True, truncation=True,  max_length=512)


    # prepare data for training based on torch 
    train_dataset = Dataset(X_train_tokens, y_train)
    test_dataset = Dataset(X_test_tokens, y_test)



    # define the training arguments - set learning rate as 2e-5

    args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=training_config['epochs_num'],
        per_device_train_batch_size=training_config['batch_size_list'][0],
        learning_rate=training_config['learning_rate'],
    )



    # define the trainer with the callback
    train_accuracy_callback = TrainAccuracyCallback(model, train_dataset, test_dataset, tokenizer)


    model = model.to(device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[train_accuracy_callback],
        # loss_function=loss_function
    )

    # train the model
    trainer.train()

    # retrieve the vectors of accuracies
    train_accuracies = train_accuracy_callback.epoch_accuracies
    test_accuracies = train_accuracy_callback.test_accuracies

    

    # Add the train and test accuracies for each fold to the matrix
    accuracy_mat_tr[fold,:] = np.asarray(train_accuracies)
    accuracy_mat_te[fold,:] = np.asarray(test_accuracies)

    
    # Save the accuracy matrix to a file

    # print and save the accuracies
    print("\n\n\nfold = %d - Final Training Accuracy: %.2f"%(fold, train_accuracies[-1]))
    print("fold = %d - Final Test Accuracy: %.2f\n\n\n"%(fold, test_accuracies[-1]))


    # Get predictions on the test dataset
    predictions = trainer.predict(test_dataset)

    # Extract predicted labels from the output
    predicted_labels_te = np.argmax(predictions.predictions, axis=1)

    # calculate prediction label for tr 
    train_predictions = trainer.predict(train_dataset)
    predicted_labels_tr = np.argmax(train_predictions.predictions, axis=1)

    # Calculate the confusion matrix for tr and te 
    cm_te = confusion_matrix(y_test, predicted_labels_te)
    cm_tr = confusion_matrix(y_train, predicted_labels_tr)

    # subplot tr and te configuration matrix 
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm_tr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[0])
    axs[0].set_title("Confusion Matrix - Fold {}: Train Acc {:.2f}".format(fold, train_accuracies[-1]))
    axs[0].set_xlabel("Predicted Label")
    axs[0].set_ylabel("True Label")

    sns.heatmap(cm_te, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[1])
    axs[1].set_title("Confusion Matrix - Fold {}: Test Acc {:.2f}".format(fold, test_accuracies[-1]))
    axs[1].set_xlabel("Predicted Label")
    axs[1].set_ylabel("True Label")



    # Save the figure
    os.makedirs(saved_results_dir, exist_ok=True)  # Create the directory if it doesn't exist
    save_name = "confMat_{}.png".format(fold+1)
    file_path = os.path.join(saved_results_dir, save_name)
    plt.savefig(file_path)
    plt.close()


    # save the model
    model_path = os.path.join(saved_results_dir, f"model_{fold+1}")
    tokenizer_path = os.path.join(saved_results_dir, f"tokenizer{fold+1}")

    # if model_path does not exist, create it
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)
        

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

    u=1


    # print and save the vectors of accuracies
    # with open("accuracies_vectors.txt", "w") as f:5
    #     f.write("Train Accuracies: " + str(train_accuracies) + "\n")
    #     f.write("Test Accuracies: " + str(test_accuracies))

######### calculate the best training epoch for each fold

thr_check = False

if thr_check == True:

    acc_thr = .3
    strt_indx = 20
    last_indx = 50 - 1  # training_config['epochs_num']

    # Initialize a vector to store indices for each fold
    indices_for_each_fold = np.zeros(training_config['cv_k'], dtype=int)
    # Iterate over each fold
    for fold in range(training_config['cv_k']):
        # Calculate the difference between training and test accuracies
        accuracy_diff = accuracy_mat_tr[fold, strt_indx:] - accuracy_mat_te[fold, strt_indx:]

        # Find the index where the difference is greater than acc_thr
        index = np.where(accuracy_diff > acc_thr)[0]

        # If there are no such indices, set it to -1
        if len(index) == 0:
            indices_for_each_fold[fold] = accuracy_mat_tr.shape[1] - 1
        else:
            # Otherwise, take the first index
            indices_for_each_fold[fold] = strt_indx + index[0]

    # Print the resulting vector
    print("Indices for each fold:", indices_for_each_fold)
else:
    indices_for_each_fold = np.zeros(training_config['cv_k'], dtype=int)
    indices_for_each_fold[:] = training_config['epochs_num'] - 1


################ save training configuration as a text file in saved_results_dir 
with open(os.path.join(saved_results_dir, 'training_config.txt'), 'w') as f:
    for key, value in training_config.items():
        f.write(f"{key}: {value}\n")



#################  Plot train and test accuracies for each fold
for i in range(training_config['cv_k']):
    indx_fold_des = indices_for_each_fold[i]
    accuracy_tr_fold = accuracy_mat_tr[i, :indx_fold_des]
    accuracy_te_fold = accuracy_mat_te[i, :indx_fold_des]

    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_tr_fold, label='Traccuracy_mat_train Accuracies')
    plt.plot(accuracy_te_fold, label='Test Accuracies')

    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('%s - tr and te acc- fold = %d'%(model_name, i))

    # Add legend
    plt.legend()

    # plt.show()

    # Save the plot
    file_path = os.path.join(saved_results_dir, f'acc_{i+1}.png')
    plt.savefig(file_path)
    plt.close()



######### acuracy box plot 
mean_train_accuracies_fold = np.zeros(training_config['cv_k'])
mean_test_accuracies_fold = np.zeros(training_config['cv_k'])
for i in range(training_config['cv_k']):
    mean_train_accuracies_fold[i] = np.mean(accuracy_mat_tr[i, indices_for_each_fold[i]])
    mean_test_accuracies_fold[i] = np.mean(accuracy_mat_te[i, indices_for_each_fold[i]])

# Add the average accuracies to the title
mean_tot_tr = np.mean(mean_train_accuracies_fold)
mean_tot_te = np.mean(mean_test_accuracies_fold )
std_tot_tr = np.std(mean_train_accuracies_fold)
std_tot_te = np.std(mean_test_accuracies_fold)

accuracies = [mean_train_accuracies_fold , mean_test_accuracies_fold]

plt.figure(figsize=(10, 5))
plt.boxplot(accuracies, labels=['Train Accuracies', 'Test Accuracies'])
plt.ylabel('Accuracy')

# y range from 0 to 1
plt.ylim(0.4, 1)
plt.title('%s - tr and te acc \n TR: mean=%.2f std=%.2f -- TE: mean=%.2f std=%.2f'%(model_name, mean_tot_tr, std_tot_tr, mean_tot_te, std_tot_te))

# plt.show()

# Save the box plot
file_path = os.path.join(saved_results_dir, 'boxAcc.jpg')
plt.savefig(file_path)
plt.close()



u=1