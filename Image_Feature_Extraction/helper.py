import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import SwinForImageClassification

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

        # num_classes = len(classes)

        # for i, (train_conf_matrix, val_conf_matrix) in enumerate(conf_matrices_all_folds):

        # Convert lists to NumPy arrays
        # train_conf_matrix = np.array(train_conf_matrix)
        # val_conf_matrix = np.array(val_conf_matrix)

        # fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # # Plot train confusion matrix
        # axs[0].set_title(f'Training Confusion Matrix - Fold {i + 1}')
        # axs[0].set_xlabel('Predicted')
        # axs[0].set_ylabel('Actual')
        # axs[0].set_xticks(np.arange(num_classes))
        # axs[0].set_yticks(np.arange(num_classes))
        # axs[0].set_xticklabels(classes, rotation=45)
        # axs[0].set_yticklabels(classes, rotation=0)
        # for y in range(train_conf_matrix.shape[0]):
        #     for x in range(train_conf_matrix.shape[1]):
        #         axs[0].text(x, y, str(train_conf_matrix[y, x]), ha='center', va='center', color='black')

        # # Plot validation confusion matrix
        # axs[1].set_title(f'Validation Confusion Matrix - Fold {i + 1}')
        # axs[1].set_xlabel('Predicted')
        # axs[1].set_ylabel('Actual')
        # axs[1].set_xticks(np.arange(num_classes))
        # axs[1].set_yticks(np.arange(num_classes))
        # axs[1].set_xticklabels(classes, rotation=45)
        # axs[1].set_yticklabels(classes, rotation=0)
        # for y in range(val_conf_matrix.shape[0]):
        #     for x in range(val_conf_matrix.shape[1]):
        #         axs[1].text(x, y, str(val_conf_matrix[y, x]), ha='center', va='center', color='black')

        # plt.tight_layout()

        # # Create the subdirectory if it doesn't exist
        # if not os.path.exists(subdirectory):
        #     os.makedirs(subdirectory)

        # # Save the plot as an image in the specified subdirectory with the given filename
        # filename = f"{filename_prefix}_fold_{i + 1}.png"
        # plt.savefig(os.path.join(subdirectory, filename))
        # plt.close()  # Close the plot to free memory


        # num_folds = len(conf_matrices_all_folds)
        # num_classes = len(classes)

        # for i, (train_conf_matrix, val_conf_matrix) in enumerate(conf_matrices_all_folds):
        #     # Convert lists to NumPy arrays
        #     train_conf_matrix = np.array(train_conf_matrix)
        #     val_conf_matrix = np.array(val_conf_matrix)

        #     fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        #     # Plot train confusion matrix
        #     axs[0].imshow(train_conf_matrix, cmap='Blues', aspect='auto')
        #     axs[0].set_title(f'Training Confusion Matrix - Fold {i + 1}')
        #     axs[0].set_xlabel('Predicted')
        #     axs[0].set_ylabel('Actual')
        #     axs[0].set_xticks(np.arange(num_classes))
        #     axs[0].set_yticks(np.arange(num_classes))
        #     axs[0].set_xticklabels(classes, rotation=45)
        #     axs[0].set_yticklabels(classes, rotation=0)
        #     for y in range(train_conf_matrix.shape[0]):
        #         for x in range(train_conf_matrix.shape[1]):
        #             axs[0].text(x, y, str(train_conf_matrix[y, x]), ha='center', va='center', color='black')

        #     # Plot validation confusion matrix
        #     axs[1].imshow(val_conf_matrix, cmap='Blues', aspect='auto')
        #     axs[1].set_title(f'Validation Confusion Matrix - Fold {i + 1}')
        #     axs[1].set_xlabel('Predicted')
        #     axs[1].set_ylabel('Actual')
        #     axs[1].set_xticks(np.arange(num_classes))
        #     axs[1].set_yticks(np.arange(num_classes))
        #     axs[1].set_xticklabels(classes, rotation=45)
        #     axs[1].set_yticklabels(classes, rotation=0)
        #     for y in range(val_conf_matrix.shape[0]):
        #         for x in range(val_conf_matrix.shape[1]):
        #             axs[1].text(x, y, str(val_conf_matrix[y, x]), ha='center', va='center', color='black')

        #     plt.tight_layout()

        #     # Create the subdirectory if it doesn't exist
        #     if not os.path.exists(subdirectory):
        #         os.makedirs(subdirectory)

        #     # Save the plot as an image in the specified subdirectory with the given filename
        #     filename = f"{filename_prefix}_fold_{i + 1}.png"
        #     plt.savefig(os.path.join(subdirectory, filename))

        for i, (train_conf_matrix, val_conf_matrix) in enumerate(conf_matrices_all_folds):
            fig, axs = plt.subplots(1, 2, figsize=(15, 7))

            # Plot train confusion matrix
            train_conf_matrix = np.squeeze(np.asarray(train_conf_matrix))
            sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap="Blues", ax=axs[0])
            axs[0].set_title(f'Training Confusion Matrix - Fold {i + 1}')
            axs[0].set_xlabel('Predicted')
            axs[0].set_ylabel('Actual')
            axs[0].set_xticklabels(classes, rotation=45)
            axs[0].set_yticklabels(classes, rotation=0)

            # Plot validation confusion matrix
            val_conf_matrix = np.squeeze(np.asarray(val_conf_matrix))
            sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap="Blues", ax=axs[1])
            axs[1].set_title(f'Validation Confusion Matrix - Fold {i + 1}')
            axs[1].set_xlabel('Predicted')
            axs[1].set_ylabel('Actual')
            axs[1].set_xticklabels(classes, rotation=45)
            axs[1].set_yticklabels(classes, rotation=0)

            plt.tight_layout()

            # Create the subdirectory if it doesn't exist
            if not os.path.exists(subdirectory):
                os.makedirs(subdirectory)

            # Save the plot as an image in the specified subdirectory with the given filename
            filename = f"{filename_prefix}_fold_{i + 1}.png"
            plt.savefig(os.path.join(subdirectory, filename))
            plt.close()



            
            # for i, (train_conf_matrix, val_conf_matrix) in enumerate(conf_matrices_all_folds):
            #     fig, axs = plt.subplots(1, 2, figsize=(15, 7))

            #     # Plot train confusion matrix
            #     train_conf_matrix = np.squeeze(np.asarray(train_conf_matrix))
            #     sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap="Blues", ax=axs[0])
            #     axs[0].set_title(f'Training Confusion Matrix - Fold {i + 1}')
            #     axs[0].set_xlabel('Predicted')
            #     axs[0].set_ylabel('Actual')
            #     axs[0].set_xticklabels(classes, rotation=45)
            #     axs[0].set_yticklabels(classes, rotation=0)

            #     # Plot validation confusion matrix
            #     val_conf_matrix = np.squeeze(np.asarray(val_conf_matrix))
            #     sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap="Blues", ax=axs[1])
            #     axs[1].set_title(f'Validation Confusion Matrix - Fold {i + 1}')
            #     axs[1].set_xlabel('Predicted')
            #     axs[1].set_ylabel('Actual')
            #     axs[1].set_xticklabels(classes, rotation=45)
            #     axs[1].set_yticklabels(classes, rotation=0)

            #     plt.tight_layout()

            #     # Create the subdirectory if it doesn't exist
            #     if not os.path.exists(subdirectory):
            #         os.makedirs(subdirectory)

            #     # Save the plot as an image in the specified subdirectory with the given filename
            #     filename = f"{filename_prefix}_fold_{i + 1}.png"
            #     plt.savefig(os.path.join(subdirectory, filename))
            #     plt.close()

    @staticmethod
    def plot_class_counts_per_fold(class_counts_per_fold_train, class_counts_per_fold_val, subdirectory, filename):

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

        plt.subplot(1, 2, 1)
        plt.title('Train Class Counts')
        plt.imshow(mat_tr, cmap='Blues', aspect='auto')
        plt.colorbar(label='Count')
        plt.ylabel('Fold')
        plt.xlabel('Class Index')
        # Annotate each element of mat_tr with its value
        for i in range(mat_tr.shape[0]):
            for j in range(mat_tr.shape[1]):
                plt.text(j, i, str(mat_tr[i, j]), ha='center', va='center', color='black')

        plt.subplot(1, 2, 2)
        plt.title('Validation Class Counts')
        plt.imshow(mat_te, cmap='Blues', aspect='auto')
        plt.colorbar(label='Count')
        plt.ylabel('Fold')
        plt.xlabel('Class Index')
        for i in range(mat_te.shape[0]):
            for j in range(mat_te.shape[1]):
                plt.text(j, i, str(mat_te[i, j]), ha='center', va='center', color='black')

        # plt.subplot(1, 2, 1)
        # plt.title('Train Class Counts')
        # sns.heatmap(mat_tr, annot=True, fmt='d', cmap='Blues')
        # plt.ylabel('Fold')
        # plt.xlabel('Class Index')

        # plt.subplot(1, 2, 2)
        # plt.title('Validation Class Counts')
        # sns.heatmap(mat_te, annot=True, fmt='d', cmap='Blues')
        # plt.ylabel('Fold')
        # plt.xlabel('Class Index')
        # plt.show()

        # Create the subdirectory if it doesn't exist
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        # Save the plot as an image in the specified subdirectory with the given filename
        plt.savefig(os.path.join(subdirectory, filename))
        plt.close()  # Close the plot to free memory

    @staticmethod
    def plot_boxplot(train_accuracies_all_folds, val_accuracies_all_folds, NUM_FOLDS, subdirectory, filename ):
        train_accuracies_last_epoch = [train_accuracies_all_folds[i][-1] for i in range(NUM_FOLDS)]
        val_accuracies_last_epoch = [val_accuracies_all_folds[i][-1] for i in range(NUM_FOLDS)]

        train_mean = np.mean(train_accuracies_last_epoch)
        train_std = np.std(train_accuracies_last_epoch)
        val_mean = np.mean(val_accuracies_last_epoch)
        val_std = np.std(val_accuracies_last_epoch)

        plt.figure(figsize=(8, 6))
        plt.boxplot([train_accuracies_last_epoch, val_accuracies_last_epoch])
        plt.xticks([1, 2], ['Train', 'Validation'])
        plt.ylabel('Accuracy')
        # ylim should be set to the range of the data + 10 to leave some space above the highest bar and below the lowest bar
        # claculate the range of the data
        data_range = max(max(train_accuracies_last_epoch), max(val_accuracies_last_epoch)) - min(min(train_accuracies_last_epoch), min(val_accuracies_last_epoch))
        lower_bound = min(min(train_accuracies_last_epoch), min(val_accuracies_last_epoch)) - 0.1 * data_range
        upper_bound = max(max(train_accuracies_last_epoch), max(val_accuracies_last_epoch)) + 0.1 * data_range
        plt.ylim(lower_bound, upper_bound)
        plt.title(f'Train Accuracy (Mean: {train_mean:.2f}, Std: {train_std:.2f})\nValidation Accuracy (Mean: {val_mean:.2f}, Std: {val_std:.2f})')
        # plt.show()
        # Create the subdirectory if it doesn't exist
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        # Save the plot as an image in the specified subdirectory with the given filename
        plt.savefig(os.path.join(subdirectory, filename))
        plt.close()

    @staticmethod
    def plot_train_val_metrics(train_metrics_all_folds, val_metrics_all_folds, metric_name, epoch_, NUM_FOLDS,
                               subdirectory, filename):
        train_metrics_last_epoch = [train_metrics_all_folds[i][-1] for i in range(NUM_FOLDS)]
        val_metrics_last_epoch = [val_metrics_all_folds[i][-1] for i in range(NUM_FOLDS)]

        mean_train_metric = np.mean(train_metrics_last_epoch)
        std_train_metric = np.std(train_metrics_last_epoch)
        mean_val_metric = np.mean(val_metrics_last_epoch)
        std_val_metric = np.std(val_metrics_last_epoch)

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        for fold in range(NUM_FOLDS):
            train_metric = train_metrics_all_folds[fold][-1]
            val_metric = val_metrics_all_folds[fold][-1]

            axs[fold // 2, fold % 2].plot(range(1, epoch_ + 1), train_metrics_all_folds[fold],
                                          label=f'Train {metric_name}', color='blue')
            axs[fold // 2, fold % 2].plot(range(1, epoch_ + 1), val_metrics_all_folds[fold],
                                          label=f'Validation {metric_name}', color='orange')

            axs[fold // 2, fold % 2].set_xlabel('Epoch')
            axs[fold // 2, fold % 2].set_ylabel(metric_name)
            axs[fold // 2, fold % 2].set_title(
                f'Fold {fold + 1}: Train {metric_name} {train_metric:.2f}%, Val {metric_name} {val_metric:.2f}%')
            axs[fold // 2, fold % 2].legend()

        fig.suptitle(
            f'Mean Train {metric_name}: {mean_train_metric:.2f}% (±{std_train_metric:.2f}%) | Mean Validation {metric_name}: {mean_val_metric:.2f}% (±{std_val_metric:.2f}%)')

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


def initialize_model(model_name, num_labels, label2id, id2label, unfreeze_layer=True):
    # check if includes vit

    if "vit" in model_name:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        )
    # check if model name starts with swin
    elif "swin" in model_name:
        model = SwinForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True # because the labels of pretrained are 1000
        )

    # freeze all layers except the classifier layer
    if unfreeze_layer==False:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    return model

def stratified_k_fold_split(ds, num_folds):
    skf = StratifiedKFold(n_splits=num_folds)
    fold_indices = []
    for train_indices, val_indices in skf.split(ds.imgs, ds.targets):
        fold_indices.append((train_indices, val_indices))
    return fold_indices




def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, dict_val_name_ind, dict_val_lbl_ind, fold_val_lbl):
    train_accuracy_history = []
    val_accuracy_history = []
    train_precision_history = []
    train_recall_history = []
    train_f1_score_history = []
    val_precision_history = []
    val_recall_history = []
    val_f1_score_history = []
    train_confusion_matrix_history = []
    val_confusion_matrix_history = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=4, verbose=True)
    

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

        train_accuracy = 100 * correct_train / total_train
        train_accuracy_history.append(train_accuracy)

        train_precision = EvaluationMetrics.calculate_precision(train_true_labels, train_predicted_labels, average='macro')
        train_recall = EvaluationMetrics.calculate_recall(train_true_labels, train_predicted_labels, average='macro')
        train_f1_score = EvaluationMetrics.calculate_f1_score(train_true_labels, train_predicted_labels, average='macro')

        train_precision_history.append(train_precision)
        train_recall_history.append(train_recall)
        train_f1_score_history.append(train_f1_score)

        if epoch == num_epochs - 1:
            train_confusion_matrix = EvaluationMetrics.calculate_confusion_matrix(train_true_labels, train_predicted_labels)
            train_confusion_matrix_history.append(train_confusion_matrix)

        model.eval()
        correct_val = 0
        total_val = 0
        val_progress_bar = tqdm(val_loader, desc="Validation", ncols=80, position=0, leave=True, colour="green")
        val_predicted_labels = []
        val_true_labels = []
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

                val_predicted_labels.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

                inputs = inputs.detach().cpu()
                labels = labels.detach().cpu()

            val_accuracy = 100 * correct_val / total_val
            val_accuracy_history.append(val_accuracy)
            dict_val_name_ind
            dict_val_lbl_ind
            fold_val_lbl

            a = np.sum(np.abs(np.asarray(fold_val_lbl) - np.asarray(val_true_labels)))


            average = 'weighted'
            val_precision = EvaluationMetrics.calculate_precision(val_true_labels, val_predicted_labels, average=average)
            val_recall = EvaluationMetrics.calculate_recall(val_true_labels, val_predicted_labels, average=average)
            val_f1_score = EvaluationMetrics.calculate_f1_score(val_true_labels, val_predicted_labels, average=average)

            val_precision_history.append(val_precision)
            val_recall_history.append(val_recall)
            val_f1_score_history.append(val_f1_score)

            if epoch == num_epochs - 1:
                val_confusion_matrix = EvaluationMetrics.calculate_confusion_matrix(val_true_labels, val_predicted_labels)
                val_confusion_matrix_history.append(val_confusion_matrix)


            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        
        
        # Step the scheduler based on validation loss
        scheduler.step(val_accuracy)
        

        
        print(
            f"    - Train       Accuracy: {train_accuracy:.2f}%    -  Precision: {train_precision:.4f}    -  Recall: {train_recall:.4f}    -  F1: {train_f1_score:.4f}")
        print(
            f"    - Validation  Accuracy: {val_accuracy:.2f}%    -  Precision: {val_precision:.4f}    -  Recall: {val_recall:.4f}    -  F1: {val_f1_score:.4f}")

    return (
        train_accuracy_history, val_accuracy_history,
        train_precision_history, val_precision_history,
        train_recall_history, val_recall_history,
        train_f1_score_history, val_f1_score_history,
        train_confusion_matrix_history, val_confusion_matrix_history
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