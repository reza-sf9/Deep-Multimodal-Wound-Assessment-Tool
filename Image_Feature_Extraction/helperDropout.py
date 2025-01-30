## work on majority voting for the test 


import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import SwinForImageClassification, DeiTForImageClassification


from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import sys
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tabulate import tabulate
from matplotlib.ticker import MaxNLocator

class EvaluationMetrics:
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_precision(y_true, y_pred, average='binary'):
        return precision_score(y_true, y_pred, average=average)

    @staticmethod
    def calculate_recall(y_true, y_pred, average='binary'):
        return recall_score(y_true, y_pred, average=average)

    @staticmethod
    def calculate_f1_score(y_true, y_pred, average='binary'):
        return f1_score(y_true, y_pred, average=average)


    @staticmethod
    def calculate_roc_auc_score(y_true, y_score):
        return roc_auc_score(y_true, y_score)

    @staticmethod
    def calculate_auc_pr_curve(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def calculate_balanced_accuracy(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return (sensitivity + specificity) / 2.0

class Visualization:
    @staticmethod
    def plot_confusion_matrices(conf_matrices_all_folds, classes, subdirectory, filename_prefix):
        font_size_lbl =12
        font_size_num = 20

        # Iterate over folds
        for i, (train_conf_matrix, te_conf_matrix_w_aug, te_conf_matrix_no_aug, te_conf_matrix_majority_label, te_conf_matrix_majority_prob) in enumerate(conf_matrices_all_folds):
            fig, axs = plt.subplots(2, 3, figsize=(15, 7))  # Use a 2x3 grid for 5 plots

            # Plot train confusion matrix
            train_conf_matrix = np.squeeze(np.asarray(train_conf_matrix))
            sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap="Blues", ax=axs[0, 0],
                           annot_kws={"size": font_size_num})  # Accessing [0, 0] subplot
            axs[0, 0].set_title('tr w aug')
            #axs[0, 0].set_xlabel('Predicted', fontsize=font_size_lbl)
            axs[0, 0].set_ylabel('Actual', fontsize=font_size_lbl)
            axs[0, 0].set_xticklabels(classes, rotation=45, fontsize=font_size_lbl)
            axs[0, 0].set_yticklabels(classes, rotation=0, fontsize=font_size_lbl)

            # Plot te confusion matrix w aug 
            te_conf_matrix_w_aug = np.squeeze(np.asarray(te_conf_matrix_w_aug))
            sns.heatmap(te_conf_matrix_w_aug, annot=True, fmt='d', cmap="Blues", ax=axs[0, 1],
                          annot_kws={"size": font_size_num})  # Accessing [0, 1] subplot
            axs[0, 1].set_title('te w aug')
            #axs[0, 1].set_xlabel('Predicted', fontsize=font_size_lbl)
            #axs[0, 1].set_ylabel('Actual', fontsize=font_size_lbl)
            axs[0, 1].set_xticklabels(classes, rotation=45, fontsize=font_size_lbl)
            axs[0, 1].set_yticklabels(classes, rotation=0, fontsize=font_size_lbl)

            # Plot te confusion matrix no aug 
            te_conf_matrix_no_aug = np.squeeze(np.asarray(te_conf_matrix_no_aug))
            sns.heatmap(te_conf_matrix_no_aug, annot=True, fmt='d', cmap="Blues", ax=axs[0, 2],
                          annot_kws={"size": font_size_num})  # Accessing [0, 2] subplot
            axs[0, 2].set_title('te no aug')
            axs[0, 2].set_xlabel('Predicted', fontsize=font_size_lbl)
            #axs[0, 2].set_ylabel('Actual', fontsize=font_size_lbl)
            axs[0, 2].set_xticklabels(classes, rotation=45, fontsize=font_size_lbl)
            axs[0, 2].set_yticklabels(classes, rotation=0, fontsize=font_size_lbl)

            # Plot te confusion matrix majority label 
            te_conf_matrix_majority_label = np.squeeze(np.asarray(te_conf_matrix_majority_label))
            sns.heatmap(te_conf_matrix_majority_label, annot=True, fmt='d', cmap="Blues", ax=axs[1, 0],
                          annot_kws={"size": font_size_num})  # Accessing [1, 0] subplot
            axs[1, 0].set_title('te majority label')
            axs[1, 0].set_xlabel('Predicted', fontsize=font_size_lbl)
            axs[1, 0].set_ylabel('Actual', fontsize=font_size_lbl)
            axs[1, 0].set_xticklabels(classes, rotation=45, fontsize=font_size_lbl)
            axs[1, 0].set_yticklabels(classes, rotation=0, fontsize=font_size_lbl)

            # Plot te confusion matrix majority prob 
            te_conf_matrix_majority_prob = np.squeeze(np.asarray(te_conf_matrix_majority_prob))
            sns.heatmap(te_conf_matrix_majority_prob, annot=True, fmt='d', cmap="Blues", ax=axs[1, 1],
                          annot_kws={"size": font_size_num})  # Accessing [1, 1] subplot
            axs[1, 1].set_title('te majority prob')
            axs[1, 1].set_xlabel('Predicted', fontsize=font_size_lbl)
            #axs[1, 1].set_ylabel('Actual', fontsize=font_size_lbl)
            axs[1, 1].set_xticklabels(classes, rotation=45, fontsize=font_size_lbl)
            axs[1, 1].set_yticklabels(classes, rotation=0, fontsize=font_size_lbl)

            # super title 
            fig.suptitle(f'Fold {i + 1}', fontsize=16)

            # Remove the empty subplot in the bottom right corner
            fig.delaxes(axs[1, 2])

            plt.tight_layout()

            # Create the subdirectory if it doesn't exist
            if not os.path.exists(subdirectory):
                os.makedirs(subdirectory)

            # Save the plot as an image in the specified subdirectory with the given filename
            filename = f"{filename_prefix}_fold_{i + 1}.png"
            plt.savefig(os.path.join(subdirectory, filename))
            plt.close()


    @staticmethod
    def plot_class_counts_per_fold(class_counts_per_fold_train, class_counts_per_fold_val, subdirectory, filename):

        fnt_size_num = 20
        fnt_size_lbl = 15

        mat_tr = []
        mat_te = []

        # Iterate over each fold dictionary
        for fold_dict_tr, fold_dict_te in zip(class_counts_per_fold_train, class_counts_per_fold_val):
            # Extract the values for the current fold and convert them to a list
            fold_values_tr = list(fold_dict_tr.values())
            fold_values_te = list(fold_dict_te.values())

            # Append the values for the current fold to the list
            mat_tr.append(fold_values_tr)
            mat_te.append(fold_values_te)

        # Convert the list of lists into a NumPy array
        mat_tr = np.array(mat_tr)
        mat_te = np.array(mat_te)

        plt.figure(figsize=(16, 8))


        ############# tr #############
        plt.subplot(1, 2, 1)
        plt.title('Train Class Counts')
        plt.imshow(mat_tr, cmap='Blues', aspect='auto')
        plt.colorbar()
        plt.ylabel('Fold', fontsize=fnt_size_lbl)
        plt.xlabel('Class Index', fontsize=fnt_size_lbl)
        # Annotate each element of mat_tr with its value
        for i in range(mat_tr.shape[0]):
            for j in range(mat_tr.shape[1]):
                plt.text(j, i, str(mat_tr[i, j]), ha='center', va='center', color='black', fontsize=fnt_size_num)

        plt.xticks(np.arange(mat_tr.shape[1]), np.arange(mat_tr.shape[1]), fontsize=fnt_size_lbl)  
        plt.yticks(np.arange(mat_tr.shape[0]), np.arange(mat_tr.shape[0]), fontsize=fnt_size_lbl)  

        ############# te #############
        plt.subplot(1, 2, 2)
        plt.title('Validation Class Counts')
        plt.imshow(mat_te, cmap='Blues', aspect='auto')
        plt.colorbar()
        plt.ylabel('Fold', fontsize=fnt_size_lbl)
        plt.xlabel('Class Index', fontsize=fnt_size_lbl)
        for i in range(mat_te.shape[0]):
            for j in range(mat_te.shape[1]):
                plt.text(j, i, str(mat_te[i, j]), ha='center', va='center', color='black', fontsize=fnt_size_num)

        plt.xticks(np.arange(mat_te.shape[1]), np.arange(mat_te.shape[1]), fontsize=fnt_size_lbl)  
        plt.yticks(np.arange(mat_te.shape[0]), np.arange(mat_te.shape[0]), fontsize=fnt_size_lbl)  

        # Create the subdirectory if it doesn't exist
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        # Save the plot as an image in the specified subdirectory with the given filename
        plt.savefig(os.path.join(subdirectory, filename))
        plt.close()  # Close the plot to free memory



    @staticmethod
    def plot_boxplot(train_accuracies_all_folds, te_accuracies_all_folds_w_aug, te_accuracies_all_folds_no_aug, te_accuracies_all_folds_mjaority_label, te_accuracies_all_folds_mjaority_prob, NUM_FOLDS, subdirectory, filename ):
        
        font_size_lbl = 12
        
        train_accuracies_last_epoch = [train_accuracies_all_folds[i][-1] for i in range(NUM_FOLDS)]
        te_accuracies_last_epoch_w_aug = [te_accuracies_all_folds_w_aug[i][-1] for i in range(NUM_FOLDS)]
        te_accuracies_last_epoch_no_aug = [te_accuracies_all_folds_no_aug[i][-1] for i in range(NUM_FOLDS)]
        te_accuracies_last_epoch_majority_label = [te_accuracies_all_folds_mjaority_label[i][-1] for i in range(NUM_FOLDS)]
        te_accuracies_last_epoch_majority_prob = [te_accuracies_all_folds_mjaority_prob[i][-1] for i in range(NUM_FOLDS)]

        train_mean = np.mean(train_accuracies_last_epoch)
        train_std = np.std(train_accuracies_last_epoch)
        te_mean_w_aug = np.mean(te_accuracies_last_epoch_w_aug)
        te_std_w_aug = np.std(te_accuracies_last_epoch_w_aug)
        te_mean_no_aug = np.mean(te_accuracies_last_epoch_no_aug)
        te_std_no_aug = np.std(te_accuracies_last_epoch_no_aug)
        te_mean_majority_label = np.mean(te_accuracies_last_epoch_majority_label)
        te_std_majority_label = np.std(te_accuracies_last_epoch_majority_label)
        te_mean_majority_prob = np.mean(te_accuracies_last_epoch_majority_prob)
        te_std_majority_prob = np.std(te_accuracies_last_epoch_majority_prob)



        plt.figure(figsize=(8, 6))
        plt.boxplot([train_accuracies_last_epoch, te_accuracies_last_epoch_w_aug, te_accuracies_last_epoch_no_aug, te_accuracies_last_epoch_majority_label, te_accuracies_last_epoch_majority_prob])
        plt.xticks([1, 2, 3, 4, 5], ['tr_w_aug', 'te_w_aug', 'te_no_aug', 'te_majority_label', 'te_majority_prob'], rotation=15, fontsize= font_size_lbl)
        plt.ylabel('Accuracy', fontsize = font_size_lbl)
        
        # ylim should be set to the range of the data + 10 to leave some space above the highest bar and below the lowest bar
        # claculate the range of the data
        data_range = np.max([train_accuracies_last_epoch, te_accuracies_last_epoch_w_aug, te_accuracies_last_epoch_no_aug, te_accuracies_last_epoch_majority_label, te_accuracies_last_epoch_majority_prob]) - np.min([train_accuracies_last_epoch, te_accuracies_last_epoch_w_aug, te_accuracies_last_epoch_no_aug, te_accuracies_last_epoch_majority_label, te_accuracies_last_epoch_majority_prob])
        plt.ylim(np.min([train_accuracies_last_epoch, te_accuracies_last_epoch_w_aug, te_accuracies_last_epoch_no_aug, te_accuracies_last_epoch_majority_label, te_accuracies_last_epoch_majority_prob]) - 0.1 * data_range, np.max([train_accuracies_last_epoch, te_accuracies_last_epoch_w_aug, te_accuracies_last_epoch_no_aug, te_accuracies_last_epoch_majority_label, te_accuracies_last_epoch_majority_prob]) + 0.1 * data_range)

        
        plt.title(
            f'tr w aug: {train_mean:.2f} ± {train_std:.2f}\n'
            f'te w augUG: {te_mean_w_aug:.2f} ± {te_std_w_aug:.2f}\n'
            f'te no aug: {te_mean_no_aug:.2f} ± {te_std_no_aug:.2f}\n'
            f'te majority label: {te_mean_majority_label:.2f} ± {te_std_majority_label:.2f}\n'
            f'te majority prob: {te_mean_majority_prob:.2f} ± {te_std_majority_prob:.2f}',
          fontsize=8)  # Set the fontsize parameter to the desired font size

        
        # plt.show()
        # Create the subdirectory if it doesn't exist
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        # Save the plot as an image in the specified subdirectory with the given filename
        plt.savefig(os.path.join(subdirectory, filename))
        plt.close()

    @staticmethod
    def plot_train_val_metrics(train_metrics_all_folds, te_metrics_all_folds_w_aug, te_metrics_all_folds_no_aug, te_metrics_all_folds_mjaority_label, te_metrics_all_folds_mjaority_prob, metric_name, epoch_, NUM_FOLDS,
                               subdirectory, filename):
        
        font_size_lbl = 15
        font_size_legend = 12
        font_size_title = 7 

        train_metrics_last_epoch = [train_metrics_all_folds[i][-1] for i in range(NUM_FOLDS)]
        te_metrics_last_epoch_w_aug = [te_metrics_all_folds_w_aug[i][-1] for i in range(NUM_FOLDS)]
        te_metrics_last_epoch_no_aug = [te_metrics_all_folds_no_aug[i][-1] for i in range(NUM_FOLDS)]
        te_metrics_last_epoch_majority_label = [te_metrics_all_folds_mjaority_label[i][-1] for i in range(NUM_FOLDS)]
        te_metrics_last_epoch_majority_prob = [te_metrics_all_folds_mjaority_prob[i][-1] for i in range(NUM_FOLDS)]


        mean_train_metric = np.mean(train_metrics_last_epoch)
        std_train_metric = np.std(train_metrics_last_epoch)
        mean_te_metric_w_aug = np.mean(te_metrics_last_epoch_w_aug)
        std_te_metric_w_aug = np.std(te_metrics_last_epoch_w_aug)
        mean_te_metric_no_aug = np.mean(te_metrics_last_epoch_no_aug)
        std_te_metric_no_aug = np.std(te_metrics_last_epoch_no_aug)
        mean_te_metric_majority_label = np.mean(te_metrics_last_epoch_majority_label)
        std_te_metric_majority_label = np.std(te_metrics_last_epoch_majority_label)
        mean_te_metric_majority_prob = np.mean(te_metrics_last_epoch_majority_prob)
        std_te_metric_majority_prob = np.std(te_metrics_last_epoch_majority_prob)


        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        for fold in range(NUM_FOLDS):
            tr_metric_w_aug = train_metrics_all_folds[fold][-1]
            te_metric_w_aug = te_metrics_all_folds_w_aug[fold][-1]
            te_metric_no_aug = te_metrics_all_folds_no_aug[fold][-1]
            te_metric_majority_label = te_metrics_all_folds_mjaority_label[fold][-1]
            te_metric_majority_prob = te_metrics_all_folds_mjaority_prob[fold][-1]

            axs[fold // 2, fold % 2].plot(range(1, epoch_ + 1), train_metrics_all_folds[fold],
                                          label=f'Train {metric_name}', color='blue')
            axs[fold // 2, fold % 2].plot(range(1, epoch_ + 1), te_metrics_all_folds_w_aug[fold],
                                          label=f'Te {metric_name} W AUG', color='orange')
            axs[fold // 2, fold % 2].plot(range(1, epoch_ + 1), te_metrics_all_folds_no_aug[fold],
                                          label=f'Validation {metric_name} NO AUG', color='red')
            axs[fold // 2, fold % 2].plot(range(1, epoch_ + 1), te_metrics_all_folds_mjaority_label[fold],
                                          label=f'Validation {metric_name} Majority Label', color='green')
            axs[fold // 2, fold % 2].plot(range(1, epoch_ + 1), te_metrics_all_folds_mjaority_prob[fold],
                                          label=f'Validation {metric_name} Majority Prob', color='purple')

            axs[fold // 2, fold % 2].set_xlabel('Epoch', fontsize= font_size_lbl)
            axs[fold // 2, fold % 2].set_ylabel(metric_name, fontsize= font_size_lbl)
            axs[fold // 2, fold % 2].set_title((
                f'Fold {fold + 1}: tr_w_aug:{tr_metric_w_aug:.2f}%-'
                f'te_w_aug:{te_metric_w_aug:.2f}%-'
                f'te_no_aug:{te_metric_no_aug:.2f}%-'
                f'te_majority_label:{te_metric_majority_label:.2f}%-'
                f'te_majority_prob:{te_metric_majority_prob:.2f}%'), fontsize=font_size_title)
            



            
            axs[fold // 2, fold % 2].legend(fontsize = font_size_legend)

        # write supertitle in which the mean and std of the metrics are included (each metric in 1 line )

        fig.suptitle(
             f' tr w aug : {mean_train_metric:.2f} ± {std_train_metric:.2f}\n'
             f' te w aug : {mean_te_metric_w_aug:.2f} ± {std_te_metric_w_aug:.2f}\n'
             f' te  no aug: {mean_te_metric_no_aug:.2f} ± {std_te_metric_no_aug:.2f}\n'
             f' te  majority label: {mean_te_metric_majority_label:.2f} ± {std_te_metric_majority_label:.2f}\n'
             f' te  majority prob: {mean_te_metric_majority_prob:.2f} ± {std_te_metric_majority_prob:.2f}')

        # set xticks and yticks font size 
        for ax in axs.flat:  # Flatten the array if multidimensional
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            
            # Alternatively, if you want to set the fontsize directly
            ax.tick_params(axis='x', labelsize=font_size_lbl)
            ax.tick_params(axis='y', labelsize=font_size_lbl)

            

        plt.tight_layout()

        # Create the subdirectory if it doesn't exist
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        # Save the plot as an image in the specified subdirectory with the given filename
        plt.savefig(os.path.join(subdirectory, filename))
        plt.close()


class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute alpha weights
        if self.alpha is not None:
            alpha_weights = torch.tensor(self.alpha, dtype=torch.float32).to(targets.device)
            alpha_weights = alpha_weights[targets]
        else:
            alpha_weights = 1

        pt = torch.exp(-ce_loss)
        focal_loss = (alpha_weights * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def initialize_model(model_name, num_labels, label2id, id2label, unfreeze_layer=True, dropout=0.1):
    # check if includes vit

    if "vit" in model_name:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            hidden_dropout_prob=dropout
        )
    # check if model name starts with swin
    elif "swin" in model_name:
        model = SwinForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True, # because the labels of pretrained are 1000
            hidden_dropout_prob=dropout)
    
    elif "deit" in model_name:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True, # because the labels of pretrained are 1000
            hidden_dropout_prob=dropout
        )

    # freeze all layers except the classifier layer
    if unfreeze_layer==False: # False > only train the last layer, 
        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.classifier.parameters():
        #     param.requires_grad = True

        # Freeze the initial layers
        for param in model.base_model.parameters():
            param.requires_grad = False

        # Fine-tune only the last 4 layers
        for param in model.base_model.encoder.layers[-4:].parameters():
            param.requires_grad = True



    return model

def stratified_k_fold_split(ds, num_folds):
    skf = StratifiedKFold(n_splits=num_folds)
    fold_indices = []
    for train_indices, val_indices in skf.split(ds.imgs, ds.targets):
        fold_indices.append((train_indices, val_indices))
    return fold_indices




def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, adaptive_lr,
                 dict_val_name_ind, dict_val_lbl_ind, fold_val_lbl, fold_val_ind, tr_thr=[False, 0], majority_problability_enhanced=False):
    
    # tr metrics 
    tr_acc_h = []
    tr_precision_h = []
    tr_recall_h = []
    tr_f1_h = []
    tr_confMat_h = []

    # te with augmentation metrics
    te_acc_h_w_aug = []
    te_precision_h_w_aug = []
    te_recall_h_w_aug = []
    te_f1_h_w_aug = []
    te_confMat_h_w_aug = []

    # te no augmentation metrics
    te_acc_h_no_aug = []
    te_precision_h_no_aug = []
    te_recall_h_no_aug = []
    te_f1_h_no_aug = []
    te_confMat_h_no_aug = []

    # te majority label metrics
    te_acc_h_majority_lbl = []
    te_precision_h_majority_lbl = []
    te_recall_h_majority_lbl = []
    te_f1_h_majority_lbl = []
    te_confMat_h_majority_lbl = []

    # te majority prob metrics
    te_acc_h_majority_prob = []
    te_precision_h_majority_prob = []
    te_recall_h_majority_prob = []
    te_f1_h_majority_prob = []
    te_confMat_h_majority_prob = []


    if adaptive_lr:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=4, verbose=True)
    else:
        scheduler = None
    

    for epoch in range(num_epochs):
        model.train()
        
        # Print the learning rate
        print(f"\nEpoch {epoch+1}, Learning Rate: {optimizer.param_groups[0]['lr']}\n")

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), colour="yellow")
        train_predicted_labels = []
        train_true_labels = []
        

        for i, data in train_progress_bar:
            inputs, labels = data['pixel_values'], data['labels']

            inputs = inputs.to(device)
            labels = labels.to(device)


            check_transferred_to_gpu = False
            if check_transferred_to_gpu:
                model_is_cuda = next(model.parameters()).is_cuda
                print("\nModel on GPU: " + str(model_is_cuda))
                img_gpu_result = inputs.device.type == 'cuda'
                labels_gpu_result = labels.device.type == 'cuda'
                print(f"Data on GPU: {img_gpu_result}")
                print(f"Labels on GPU: {labels_gpu_result}")


            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.logits, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            running_loss += loss.item()

            train_progress_bar.set_description(f"Train Loss: {running_loss / (i+1):.4f}")
            train_progress_bar.set_postfix(train_accuracy=f'{100 * correct_train / total_train:.2f}%')

            train_predicted_labels.extend(predicted.cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())

            inputs = inputs.detach().cpu()
            labels = labels.detach().cpu()

        train_accuracy = correct_train / total_train
        train_precision = EvaluationMetrics.calculate_precision(train_true_labels, train_predicted_labels, average='macro')
        train_recall = EvaluationMetrics.calculate_recall(train_true_labels, train_predicted_labels, average='macro')
        train_f1_score = EvaluationMetrics.calculate_f1_score(train_true_labels, train_predicted_labels, average='macro')

        tr_acc_h.append(train_accuracy)
        tr_precision_h.append(train_precision)
        tr_recall_h.append(train_recall)
        tr_f1_h.append(train_f1_score)

        if epoch == num_epochs - 1:
            train_confusion_matrix = EvaluationMetrics.calculate_confusion_matrix(train_true_labels, train_predicted_labels)
            tr_confMat_h.append(train_confusion_matrix)

        model.eval()
        correct_val = 0
        total_val = 0
        val_progress_bar = tqdm(val_loader, desc="Validation", ncols=80, position=0, leave=True, colour="green")
        val_predicted_labels = []
        val_true_labels = []
        val_predicted_probs = []  # List to store predicted probabilities
        with torch.no_grad():
            for data in val_progress_bar:
                inputs, labels = data['pixel_values'], data['labels']

                # move to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.logits, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # get predicted probability of each class
                probs = F.softmax(outputs.logits, dim=1)
                # Append predicted probabilities to the list
                val_predicted_probs.extend(probs.cpu().numpy())


                val_predicted_labels.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

                inputs = inputs.detach().cpu()
                labels = labels.detach().cpu()

            val_accuracy = correct_val / total_val
            
            average = 'weighted'
            val_precision = EvaluationMetrics.calculate_precision(val_true_labels, val_predicted_labels, average=average)
            val_recall = EvaluationMetrics.calculate_recall(val_true_labels, val_predicted_labels, average=average)
            val_f1_score = EvaluationMetrics.calculate_f1_score(val_true_labels, val_predicted_labels, average=average)

            te_acc_h_w_aug.append(val_accuracy)
            te_precision_h_w_aug.append(val_precision)
            te_recall_h_w_aug.append(val_recall)
            te_f1_h_w_aug.append(val_f1_score)

            if epoch == num_epochs - 1:
                val_confusion_matrix = EvaluationMetrics.calculate_confusion_matrix(val_true_labels, val_predicted_labels)
                te_confMat_h_w_aug.append(val_confusion_matrix)


            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        
        
        # Step the scheduler based on validation loss
        if adaptive_lr:
            scheduler.step(val_accuracy)
        

        val_predicted_labels
        val_true_labels


        # info from augmentation 
        dict_val_name_ind
        dict_val_lbl_ind
        fold_val_lbl
        fold_val_ind

        keys_name = list(dict_val_name_ind.keys())

        list_ = []
        

        vec_retrieved_label = []
        vec_retrieved_pred_majority_lbl = []
        vec_retrieved_pred_majority_prob = []
        vec_no_te_aug = []
        vec_lbl_retrieved_all = []

        low_ind = 0
        for key in keys_name:
            aug_suffix_key = dict_val_name_ind[key]
            aug_lbl_key = dict_val_lbl_ind[key]
            vec_lbl_retrieved_all.append(aug_lbl_key)
            

            for i in range(len(aug_lbl_key)):
                str_ = "%s_ind%d_lbl%d"%(key, aug_suffix_key[i], aug_lbl_key[i])
                list_.append(str_)

            
            hihg_ind = low_ind + len(aug_lbl_key)
            vec_pred = val_predicted_labels[low_ind: hihg_ind]
            vec_true = val_true_labels[low_ind: hihg_ind]

            ################# MAJORITH LABEL 
            unique, counts = np.unique(vec_pred, return_counts=True)
            dict_count = dict(zip(unique, counts))
            max_key = max(dict_count, key=dict_count.get)
            # if no max key ? get first one (original image)
            if max_key not in vec_pred:
                max_key = vec_pred[0]
                print('no max key found, get the first one')

            # check if all vec_true are the same, if yes, define val_true 
            if len(set(vec_true)) == 1:
                val_true = vec_true[0]
            else:
                print("error happened, not all true labels are the same!")
                        
            vec_retrieved_pred_majority_lbl.append(max_key)
            vec_retrieved_label.append(val_true)

            ################ find MAJORITY PROBABILITY 
            val_predicted_probs = np.asarray(val_predicted_probs)
            pred_prob_key = val_predicted_probs[low_ind: hihg_ind, :]
            pred_prob_sum = np.sum(pred_prob_key, axis=0)

            # find the index of max prob 
            max_prob_ind = np.argmax(pred_prob_sum)
            vec_retrieved_pred_majority_prob.append(max_prob_ind)

            ################# NO TEST AUGMENTATION RESULT 
            vec_no_te_aug.append(vec_pred[0])

            # update low_ind
            low_ind = hihg_ind


        # Flatten the nested list into a single list
        vec_lbl_retrieved_flattened = [item for sublist in vec_lbl_retrieved_all for item in sublist]
        
        # check if vec_retrieved_label and labels are same 
        a = np.sum(np.abs(np.asarray(vec_lbl_retrieved_flattened) - np.asarray(val_true_labels)))
        if a>0:
            print("error happened, not all true labels are the same!")
        else:

            # calculate accuracy, percision, recall, f1 based on NO TEST AUGMENTATION
            val_accuracy_no_te_aug = accuracy_score(vec_retrieved_label, vec_no_te_aug)
            val_precision_no_te_aug = precision_score(vec_retrieved_label, vec_no_te_aug, average=average)
            val_recall_no_te_aug = recall_score(vec_retrieved_label, vec_no_te_aug, average=average)
            val_f1_score_no_te_aug = f1_score(vec_retrieved_label, vec_no_te_aug, average=average)

            # calculate accuracy, percision, recall, f1 based on MAJORITY LABEL VOTING 
            val_accuracy_majority_lbl = accuracy_score(vec_retrieved_label, vec_retrieved_pred_majority_lbl)
            val_precision_majority_lbl  = precision_score(vec_retrieved_label, vec_retrieved_pred_majority_lbl, average=average)
            val_recall_majority_lbl  = recall_score(vec_retrieved_label, vec_retrieved_pred_majority_lbl, average=average)
            val_f1_score_majority_lbl  = f1_score(vec_retrieved_label, vec_retrieved_pred_majority_lbl, average=average)

            # calculate accuracy, percision, recall, f1 based on MAJORITY PROBEBILITY VOTING
            val_accuracy_majority_prob = accuracy_score(vec_retrieved_label, vec_retrieved_pred_majority_prob)
            val_precision_majority_prob = precision_score(vec_retrieved_label, vec_retrieved_pred_majority_prob, average=average)
            val_recall_majority_prob = recall_score(vec_retrieved_label, vec_retrieved_pred_majority_prob, average=average)
            val_f1_score_majority_prob = f1_score(vec_retrieved_label, vec_retrieved_pred_majority_prob, average=average)


        # append the result of TE NO AUG
        te_acc_h_no_aug.append(val_accuracy_no_te_aug)
        te_precision_h_no_aug.append(val_precision_no_te_aug)
        te_recall_h_no_aug.append(val_recall_no_te_aug)
        te_f1_h_no_aug.append(val_f1_score_no_te_aug)

        # append the result of TE MAJORITY LABEL
        te_acc_h_majority_lbl.append(val_accuracy_majority_lbl)
        te_precision_h_majority_lbl.append(val_precision_majority_lbl)
        te_recall_h_majority_lbl.append(val_recall_majority_lbl)
        te_f1_h_majority_lbl.append(val_f1_score_majority_lbl)

        # append the result of TE MAJORITY PROB
        te_acc_h_majority_prob.append(val_accuracy_majority_prob)
        te_precision_h_majority_prob.append(val_precision_majority_prob)
        te_recall_h_majority_prob.append(val_recall_majority_prob)
        te_f1_h_majority_prob.append(val_f1_score_majority_prob)


        if epoch == num_epochs - 1:
            # conf mat for no test augmentation
            val_confusion_matrix_no_te_aug = confusion_matrix(vec_retrieved_label, vec_no_te_aug)
            te_confMat_h_no_aug.append(confusion_matrix(vec_retrieved_label, vec_no_te_aug))

            # conf mat for majority label
            val_confusion_matrix_majority_lbl = confusion_matrix(vec_retrieved_label, vec_retrieved_pred_majority_lbl)
            te_confMat_h_majority_lbl.append(val_confusion_matrix_majority_lbl)

            # conf mat for majority prob
            val_confusion_matrix_majority_prob = confusion_matrix(vec_retrieved_label, vec_retrieved_pred_majority_prob)
            te_confMat_h_majority_prob.append(val_confusion_matrix_majority_prob)

    
        data_tbl = [
            ["tr", "{:.2f}".format(train_accuracy), "{:.2f}".format(train_precision), "{:.2f}".format(train_recall), "{:.2f}".format(train_f1_score)],
            ["te w aug", "{:.2f}".format(val_accuracy), "{:.2f}".format(val_precision), "{:.2f}".format(val_recall), "{:.2f}".format(val_f1_score)],
            ['te no aug', "{:.2f}".format(val_accuracy_no_te_aug), "{:.2f}".format(val_precision_no_te_aug), "{:.2f}".format(val_recall_no_te_aug), "{:.2f}".format(val_f1_score_no_te_aug)],
            ["te lbl majority", "{:.2f}".format(val_accuracy_majority_lbl ), "{:.2f}".format(val_precision_majority_lbl ), "{:.2f}".format(val_recall_majority_lbl ), "{:.2f}".format(val_f1_score_majority_lbl )],
            ["te prob majority", "{:.2f}".format(val_accuracy_majority_prob ), "{:.2f}".format(val_precision_majority_prob ), "{:.2f}".format(val_recall_majority_prob ), "{:.2f}".format(val_f1_score_majority_prob )],
        ]

        print(tabulate(data_tbl, headers=["", "Accuracy", "Precision", "Recall", "F1 Score"]))

        if tr_thr[0]:
            if train_accuracy == tr_thr[1]:
                print("Early stopping - Train accuracy reached %.2f" % tr_thr[1])
                break

        
    return (
        tr_acc_h, te_acc_h_w_aug, te_acc_h_no_aug, te_acc_h_majority_lbl, te_acc_h_majority_prob,
        tr_precision_h, te_precision_h_w_aug, te_precision_h_no_aug, te_precision_h_majority_lbl, te_precision_h_majority_prob,
        tr_recall_h, te_recall_h_w_aug, te_recall_h_no_aug, te_recall_h_majority_lbl, te_recall_h_majority_prob,
        tr_f1_h, te_f1_h_w_aug, te_f1_h_no_aug, te_f1_h_majority_lbl, te_f1_h_majority_prob,
        tr_confMat_h, te_confMat_h_w_aug, te_confMat_h_no_aug, te_confMat_h_majority_lbl, te_confMat_h_majority_prob
    )

def load_data(data_dir):
    ds = ImageFolder(data_dir)
    return ds


def create_data_loaders(ds, train_indices, val_indices, batch_size, collator):
    train_ds_fold = torch.utils.data.Subset(ds, train_indices)
    val_ds_fold = torch.utils.data.Subset(ds, val_indices)


    train_loader_fold = DataLoader(train_ds_fold, batch_size=batch_size, collate_fn=collator, num_workers=2, shuffle=True, pin_memory=True)
    val_loader_fold = DataLoader(val_ds_fold, batch_size=batch_size, collate_fn=collator, num_workers=2, pin_memory=True)
    return train_loader_fold, val_loader_fold

def save_terminal_output_to_file(output_file):
    # Save terminal output to a file
    with open(output_file, 'w') as f:
        sys.stdout = f  # Redirect stdout to the file
        print("This output will be saved to the file.")
        print("You can write any text here and it will be saved to the file.")

def clear_terminal():
    # Clear the terminal
    subprocess.call('cls' if subprocess.os.name == 'nt' else 'clear', shell=True)

def save_config_to_file(config_file, config_train, current_time_dir):
    # Create the directory if it doesn't exist
    os.makedirs(current_time_dir, exist_ok=True)

    with open(os.path.join(current_time_dir, config_file), 'w') as f:
        for key, value in config_train.items():
            f.write(f'{key}: {value}\n')


def get_first_word_model_name(string):
    if '/' in string:
        # Split the string by '/' and take the second part
        first_part = string.split('/')[1]
    else:
        # If '/' is not present, take the whole string
        first_part = string

    # Split the first part by '-' and take the first part
    first_word = first_part.split('-')[0]
    return first_word



# Define function to extract image names and their corresponding indices from dataset
def extract_image_names_and_indices(dataset, indices):
    image_names = [os.path.basename(dataset.imgs[idx][0]) for idx in indices]
    image_indices = [idx for idx in indices]
    return image_names, image_indices