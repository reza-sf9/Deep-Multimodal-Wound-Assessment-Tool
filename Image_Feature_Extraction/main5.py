# add majority voting to main4.py 
# add train threshold 
# add majority prob enhanced (not implemented yet, only its parameter added)



## add main and write with command line script
## load model one time from the saved model and make sure it is working well
# claculate all metrics for each class labels

# print python version



import sys
print("python version = %s \n\n"%sys.version)


from pathlib import Path
from helperDropout import *
import os
import datetime
import copy
import transformers
from transformers import ViTImageProcessor
from sklearn.model_selection import train_test_split

# check if gpu is availabl (if device is cuda) else devide=cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" {device} is available")

# Constants
EPOCH_NUM = 100
LEARNING_RATE = 1e-4
ADAPTIVE_LR  = False
BATCH_SIZE = 16
NUM_FOLDS = 4
SEED = 42
# MODEL_NAME = ["google/vit-base-patch16-224-in21k", "microsoft/swinv2-tiny-patch4-window8-256"]
# MODEL_NAME = ["google/vit-base-patch16-224-in21k"] 
MODEL_NAME = ["facebook/deit-tiny-patch16-224"]
# MODEL_NAME = ["microsoft/swinv2-tiny-patch4-window8-256"] # "google/vit-base-patch16-224-in21k", "microsoft/swinv2-tiny-patch4-window8-256", "facebook/deit-base-distilled-patch16-224"
LOSS_TYPE = ["baseline"] # "FocalLoss", "CrossEntropyLoss", "baseline"
AUG_TYPE = ["NONE"]
UNFREEZE_NUM = [True] # False > only train the last layer, True > train all layers
IMAGE_TYPE = 'original' # 'original' or 'cropped1:35:11
fldr_name_orig = 'croppedPolicy1'
# FLDR_NAME_AUG = ['originalPolicy1', 
#'augPolicy1_260_240_278_from_both', 
#'augPolicy1_1950_1800_2085_from_cropped', 'augPolicy1_1950_1800_2085_from_both',
# augPolicy1_3900_3600_4170_from_cropped]
FLDR_NAME_AUG = ['augPolicy1_1950_1800_2085_from_cropped']
DROP_OUT  = 0.7
L2Reg = False  
LAMBDA_L2  = 0.01
tr_thr = [True, 0.85] 
majority_prob_enhanced = False 


config_train = {
    "EPOCH_NUM": EPOCH_NUM,
    "LEARNING_RATE": LEARNING_RATE,
    "ADAPTIVE_LR ": ADAPTIVE_LR ,
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_FOLDS": NUM_FOLDS,
    "SEED": SEED,
    "MODEL_NAME": MODEL_NAME,
    "LOSS_TYPE": LOSS_TYPE,
    "AUG_TYPE": AUG_TYPE,
    "UNFREEZE_NUM": UNFREEZE_NUM,
    "DEVICE": device,
    "IMAGE_TYPE": IMAGE_TYPE,
    "fldr_name_orig": fldr_name_orig,
    "FLDR_NAME_AUG": FLDR_NAME_AUG,
    "DROP_OUT": DROP_OUT,
    "L2Reg": L2Reg,
    "LAMBDA_L2 ": LAMBDA_L2,
    "tr_thr": tr_thr,
    "majority_prob_enhanced": majority_prob_enhanced
}

# print all items of configuration in a for loop 
print("\n\nConfiguration:")
max_key_length = max(len(key) for key in config_train.keys())
for key, value in config_train.items():
    print(f"{key.ljust(max_key_length)} : {value}")


# device = "cpu"

## set random seed64
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

####################### Load Data #######################
# Get data directory
current_dir = os.getcwd()  # Using current working directory instead of __file__
levels_to_go_up = 3
data_dir = current_dir
for _ in range(levels_to_go_up):
    data_dir = os.path.split(data_dir)[0]
data_dir = os.path.join(data_dir, 'data/wound_data/labeledFolder')



# get current time
now = datetime.datetime.now()
current_time = now.strftime("%Y_%m_%d__%H_%M_%S")
# if there is not saved_data foder, create it
saved_info_dir = os.path.join(current_dir, 'saved_info')
os.makedirs(saved_info_dir, exist_ok=True)

# in the saved_data folder, create a subdirectory with name of current_time
current_time_dir = os.path.join(saved_info_dir, current_time)
os.makedirs(current_time_dir, exist_ok=True)

cnt_save = 1
for fldr_name_aug in FLDR_NAME_AUG:
    # fldr_name = "augPolicy1_1950_1800_2085"
    data_dir_orig = os.path.join(data_dir, fldr_name_orig)
    data_dir_aug = os.path.join(data_dir, fldr_name_aug)

    # Initialize dataset for original images
    data_dir_orig = Path(data_dir_orig)
    ds_orig = load_data(data_dir_orig)

    # Initialize dataset for augmented images
    data_dir_aug = Path(data_dir_aug)
    ds_aug = load_data(data_dir_aug)

    # Initialize model
    label2id = {class_name: str(i) for i, class_name in enumerate(ds_orig.classes)}
    id2label = {str(i): class_name for i, class_name in enumerate(ds_orig.classes)}


    # Initialize stratified k-fold cross-validation
    len_data = len(ds_orig)
    indices = np.arange(len_data)
    y_ = np.asarray(ds_orig.targets)


    if NUM_FOLDS == 1:
        
        tr_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=y_, random_state=42)
        # create a list of tuples, each tuple contains the indices of the train and validation data
        fold_indices_orig = [(tr_indices, val_indices)]
    else:
        fold_indices_orig = stratified_k_fold_split(ds_orig, NUM_FOLDS)


    ####################


    # Extract image names and indices for each fold
    fold_indices_aug = []
    fold_dict_val_name_ind = []
    fold_dict_val_lbl_ind = []
    fold_val_lbl = []

    i = -1
    for tr_indices, val_indices in tqdm(fold_indices_orig, desc="Extracting image names and indices for each fold"):
        i = i + 1
        tr_image_names, tr_image_indices = extract_image_names_and_indices(ds_orig, tr_indices)
        val_image_names, val_image_indices = extract_image_names_and_indices(ds_orig, val_indices)

        # Get corresponding augmented indices for training images
        tr_aug_indices = []
        for tr_name in tr_image_names:
            tr_name_without_ext, _ = os.path.splitext(tr_name)
            # Loop over the enumerated ds_aug.imgs
            for idx, img_path in enumerate(ds_aug.imgs):
                # Check if the basename of the img_path starts with tr_name_without_ext
                img_name_idx = os.path.basename(img_path[0])
                # remove extension
                img_name_idx_1 = img_name_idx.split(".")[0]
                # remove character after the last underscore
                img_name_idx_2 = img_name_idx_1.rsplit('_', 1)[0]

                if img_name_idx_2 == tr_name_without_ext:
                    # If it does, extend tr_aug_indices with the current index
                    tr_aug_indices.append(idx)

        # Get corresponding augmented indices for validation images
        val_aug_indices = []
        dict_val_name_ind = {}
        dict_val_lbl_ind = {}
        val_lbl = []
        val_indx_uniq_gen = []
        indx_uniq=-1
        for val_name in val_image_names:
            indx_uniq +=1
            val_name_without_ext, _ = os.path.splitext(val_name)
            # Loop over the enumerated ds_aug.imgs

            all_idx_val = []
            all_idx_val_lbl = []
            for idx, img_path in enumerate(ds_aug.imgs):
                # Check if the basename of the img_path starts with val_name_without_ext
                img_name_idx = os.path.basename(img_path[0])
                # remove extension
                img_name_idx_1 = img_name_idx.split(".")[0]
                # remove character after the last underscore
                img_name_idx_2 = img_name_idx_1.rsplit('_', 1)[0]

                if img_name_idx_2 == val_name_without_ext:
                    # If it does, extend val_aug_indices with the current index
                    val_aug_indices.append(idx)
                    all_idx_val.append(idx)
                    all_idx_val_lbl.append(ds_aug.targets[idx])
                    val_lbl.append(ds_aug.targets[idx])
                    val_indx_uniq_gen.append(indx_uniq)

            dict_val_name_ind[val_name] = all_idx_val
            dict_val_lbl_ind[val_name] = all_idx_val_lbl

        # Append image names and indices to the list of fold_indices_aug
        fold_indices_aug.append((tr_aug_indices, val_aug_indices))
        fold_dict_val_name_ind.append(dict_val_name_ind)
        fold_dict_val_lbl_ind.append(dict_val_lbl_ind)
        fold_val_lbl.append(val_lbl)


    ####################

    chk = False
    if chk:
        # train 


        # validation 
        dict_lbl_val_fld0 = fold_dict_val_lbl_ind[0]
        dict_lbl_val_fld1 = fold_dict_val_lbl_ind[1]
        dict_lbl_val_fld2 = fold_dict_val_lbl_ind[2]
        dict_lbl_val_fld3 = fold_dict_val_lbl_ind[3]

        dict_name_val_fld0 = fold_dict_val_name_ind[0]
        dict_name_val_fld1 = fold_dict_val_name_ind[1]
        dict_name_val_fld2 = fold_dict_val_name_ind[2]
        dict_name_val_fld3 = fold_dict_val_name_ind[3]

        print("\n\n")
        print('fold0 - dict #lbl = %d - #name = %d' % (len(dict_lbl_val_fld0), len(dict_name_val_fld0)))
        print('fold1 - dict #lbl = %d - #name = %d' % (len(dict_lbl_val_fld1), len(dict_name_val_fld1)))
        print('fold2 - dict #lbl = %d - #name = %d' % (len(dict_lbl_val_fld2), len(dict_name_val_fld2)))
        print('fold3 - dict #lbl = %d - #name = %d' % (len(dict_lbl_val_fld3), len(dict_name_val_fld3)))

        lbl_val_fold0 = fold_val_lbl[0]
        lbl_val_fold1 = fold_val_lbl[1]
        lbl_val_fold2 = fold_val_lbl[2]
        lbl_val_fold3 = fold_val_lbl[3]

        print("\n\n")
        print('fold0 - #lbl = %d' % len(lbl_val_fold0))
        print('fold1 - #lbl = %d' % len(lbl_val_fold1))
        print('fold2 - #lbl = %d' % len(lbl_val_fold2))
        print('fold3 - #lbl = %d' % len(lbl_val_fold3))

        val_ind_augـfold0= fold_indices_aug[0][1]



    # Initialize model for the current fold
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    ############################### Training loop ###############################


    
    for model_name in MODEL_NAME:
        print("\n ############# Model: %s #############" % model_name)
        # Initialize feature extractor and collator
        # NOT SURE ABOUT THESE TWO LINES FOR SWIN MODEL (MAYBE NEED TO CHANGE)
        # feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        collator = ImageClassificationCollator(feature_extractor)
        # processor = AutoImageProcessor.from_pretrained(model_name)
        # collator = ImageClassificationCollator(processor)


        for aug_type in AUG_TYPE:
            print("\n ############# Augmentation: %s #############" % aug_type)
            for unfreeze_num in UNFREEZE_NUM:
                print("\n ############# Unfreeze Num: %d #############" % unfreeze_num)
                for loss_type in LOSS_TYPE:
                    print("\n ############# Loss Type: %s #############" % loss_type)
                    # Lists to store accuracy history for each fold

                    # TR
                    tr_acc_all_folds = []
                    tr_percision_all_folds = []
                    tr_recall_all_folds = []
                    tr_f1_all_folds = []
                    tr_confusion_matrix_all_folds = []

                    # TE W AUG
                    te_acc_all_folds_w_aug = []
                    te_percision_all_folds_w_aug = []
                    te_recall_all_folds_w_aug = []
                    te_f1_all_folds_w_aug = []
                    te_confusion_matrix_all_folds_w_aug  = []

                    # TE NO AUG
                    te_acc_all_folds_no_aug = []
                    te_percision_all_folds_no_aug = []
                    te_recall_all_folds_no_aug = []
                    te_f1_all_folds_no_aug = []
                    te_confusion_matrix_all_folds_no_aug  = []

                    # TE MAJORITY LABEL
                    te_acc_all_folds_majority_lbl = []
                    te_percision_all_folds_majority_lbl = []
                    te_recall_all_folds_majority_lbl = []
                    te_f1_all_folds_majority_lbl = []
                    te_confusion_matrix_all_folds_majority_lbl  = []

                    # TE MAJORITY PROB
                    te_acc_all_folds_majority_prob = []
                    te_percision_all_folds_majority_prob = []
                    te_recall_all_folds_majority_prob = []
                    te_f1_all_folds_majority_prob = []
                    te_confusion_matrix_all_folds_majority_prob  = []



                    tr_loaders_all_folds = []
                    val_loaders_all_folds = []

                    class_counts_per_fold_tr = []
                    class_counts_per_fold_val = []

                    for fold, (tr_indices, val_indices) in enumerate(fold_indices_aug, 1):
                        
                        dict_val_name_ind = fold_dict_val_name_ind[fold-1]
                        dict_val_lbl_ind = fold_dict_val_lbl_ind[fold-1]
                        val_lbl_fold = fold_val_lbl[fold-1]
                        fold_val_ind_aug = fold_indices_aug[fold-1][1]

                        print("\n ############# fold %d/%d #############\n" % (fold, NUM_FOLDS))

                        ########## Extract labels for the current fold
                        fold_labels_train = [ds_aug.targets[i] for i in tr_indices]
                        fold_labels_val = [ds_aug.targets[i] for i in val_indices]
                        # Calculate class counts for the current fold
                        fold_class_counts_train = {class_idx: fold_labels_train.count(class_idx) for class_idx in range(len(ds_aug.classes))}
                        fold_class_counts_val = {class_idx: fold_labels_val.count(class_idx) for class_idx in range(len(ds_aug.classes))}
                        # Append class counts to the list
                        class_counts_per_fold_tr.append(fold_class_counts_train)
                        class_counts_per_fold_val.append(fold_class_counts_val)

                        # find percentage of each class in the training
                        class_count = torch.tensor(np.asarray([fold_class_counts_train[i] for i in range(len(ds_aug.classes))])/len(tr_indices))

                        # Initialize model for the current fold
                        model = initialize_model(model_name, len(label2id), label2id, id2label, unfreeze_num, DROP_OUT)

                        # Move the model to the specified device (GPU)
                        model.to(device)

                        # Initialize optimizer and criterion
                        if L2Reg:
                            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA_L2 )
                        else:
                            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                        class_count = class_count.float().to(device)
                        if loss_type == "FocalLoss":
                            # print("\n ################ Focal Loss ################\n")
                            criterion = FocalLoss(alpha=class_count)
                        elif loss_type == "CrossEntropyLoss":
                            # print("\n ################ Cross Entropy Loss ################\n")
                            criterion = torch.nn.CrossEntropyLoss(weight=class_count)
                        elif loss_type == "baseline":
                            criterion = torch.nn.CrossEntropyLoss()

                        # Create data loaders for the current fold
                        train_loader_fold, val_loader_fold = create_data_loaders(ds_aug, tr_indices, val_indices, BATCH_SIZE, collator)
                        tr_loaders_all_folds.append(train_loader_fold)
                        val_loaders_all_folds.append(val_loader_fold)

                        # Train the model for the current fold
                        
                        tr_acc_h, te_acc_h_w_aug, te_acc_h_no_aug, te_acc_h_majority_lbl, te_acc_h_majority_prob, \
                        tr_precision_h, te_precision_h_w_aug, te_precision_h_no_aug, te_precision_h_majority_lbl, te_precision_h_majority_prob, \
                        tr_recall_h, te_recall_h_w_aug, te_recall_h_no_aug, te_recall_h_majority_lbl, te_recall_h_majority_prob, \
                        tr_f1_h, te_f1_h_w_aug, te_f1_h_no_aug, te_f1_h_majority_lbl, te_f1_h_majority_prob,\
                        tr_confMat_h, te_confMat_h_w_aug, te_confMat_h_no_aug, te_confMat_h_majority_lbl, te_confMat_h_majority_prob \
                        = train_model(
                            model, train_loader_fold, val_loader_fold, optimizer, criterion, EPOCH_NUM, device, ADAPTIVE_LR ,
                            dict_val_name_ind, dict_val_lbl_ind, val_lbl_fold, fold_val_ind_aug, tr_thr, majority_prob_enhanced
                        )


                        # accuracy, percision, recall, f1_score for each fold

                        ######### TR 
                        tr_acc_all_folds.append(tr_acc_h)
                        tr_percision_all_folds.append(tr_precision_h)
                        tr_recall_all_folds.append(tr_recall_h)
                        tr_f1_all_folds.append(tr_f1_h)
                        tr_confusion_matrix_all_folds.append(tr_confMat_h)
                        

                        ######### TE W AUG
                        te_acc_all_folds_w_aug.append(te_acc_h_w_aug)
                        te_percision_all_folds_w_aug.append(te_precision_h_w_aug)
                        te_recall_all_folds_w_aug.append(te_recall_h_w_aug)
                        te_f1_all_folds_w_aug.append(te_f1_h_w_aug)
                        te_confusion_matrix_all_folds_w_aug.append(te_confMat_h_w_aug)

                        ######### TE NO AUG
                        te_acc_all_folds_no_aug.append(te_acc_h_no_aug)
                        te_percision_all_folds_no_aug.append(te_precision_h_no_aug)
                        te_recall_all_folds_no_aug.append(te_recall_h_no_aug)
                        te_f1_all_folds_no_aug.append(te_f1_h_no_aug)
                        te_confusion_matrix_all_folds_no_aug.append(te_confMat_h_no_aug)

                        ######### TE MAJORITY LABEL
                        te_acc_all_folds_majority_lbl.append(te_acc_h_majority_lbl)
                        te_percision_all_folds_majority_lbl.append(te_precision_h_majority_lbl)
                        te_recall_all_folds_majority_lbl.append(te_recall_h_majority_lbl)
                        te_f1_all_folds_majority_lbl.append(te_f1_h_majority_lbl)
                        te_confusion_matrix_all_folds_majority_lbl.append(te_confMat_h_majority_lbl)

                        ######### TE MAJORITY PROB
                        te_acc_all_folds_majority_prob.append(te_acc_h_majority_prob)
                        te_percision_all_folds_majority_prob.append(te_precision_h_majority_prob)
                        te_recall_all_folds_majority_prob.append(te_recall_h_majority_prob)
                        te_f1_all_folds_majority_prob.append(te_f1_h_majority_prob)
                        te_confusion_matrix_all_folds_majority_prob.append(te_confMat_h_majority_prob)



                        # calculate the confusion matrix for the train and validation

                        # save model into subdirectory saved_model folder
                        # model_saved_path = os.path.join(current_time_dir, 'saved_model')
                        # model_saved_path = Path(os.path.join(model_saved_path, f'fold_{fold}'))
                        # model_saved_path.mkdir(parents=True, exist_ok=True)
                        # model.save_pretrained(model_saved_path)

                    print('Finished Training')

                    # save configuration in a text file
                    config_tr_save = copy.copy(config_train)
                    config_tr_save["MODEL_NAME"] = model_name
                    config_tr_save["LOSS_TYPE"] = loss_type
                    config_tr_save["AUG_TYPE"] = aug_type
                    config_tr_save["UNFREEZE_NUM"] = unfreeze_num
                    config_tr_save["FLDR_NAME_AUG"] = fldr_name_aug
                    config_tr_save["DROP_OUT"] = DROP_OUT
                    config_tr_save["L2Reg"] = L2Reg
                    if L2Reg:
                        config_tr_save["LAMBDA_L2 "] = LAMBDA_L2 
                    config_tr_save["ADAPTIVE_LR "] = ADAPTIVE_LR 


                    saveModelName = get_first_word_model_name(model_name)
                    str_fldr = str(cnt_save) + "_" + fldr_name_aug + "_" + saveModelName + "_aug" + aug_type + "_loss" + loss_type + "_unfreeze" + str(unfreeze_num)
                    cnt_save += 1

                    str_subdir_save = os.path.join(current_time_dir, str_fldr)

                    config_file = 'config.txt'
                    save_config_to_file(config_file, config_tr_save, str_subdir_save)


                    # clean the terminal
                    os.system('clear')

                    ############################## Visualization ##############################
                    print("\n\nVisuzalization")


                    # Plot class counts for each fold
                    Visualization.plot_class_counts_per_fold(class_counts_per_fold_tr, class_counts_per_fold_val, str_subdir_save, "class_counts_per_fold")

                    # Plot train and validation accuracies, precisions, recalls, and F1 scores for each fold
                    Visualization.plot_train_val_metrics(tr_acc_all_folds, te_acc_all_folds_w_aug, te_acc_all_folds_no_aug, te_acc_all_folds_majority_lbl, te_acc_all_folds_majority_prob,
                                                         "Accuracy", EPOCH_NUM, NUM_FOLDS, str_subdir_save, "acc")
                    Visualization.plot_train_val_metrics(tr_percision_all_folds, te_percision_all_folds_w_aug, te_percision_all_folds_no_aug, te_percision_all_folds_majority_lbl, te_percision_all_folds_majority_prob,
                                                         "Precision", EPOCH_NUM, NUM_FOLDS, str_subdir_save, "precision")
                    Visualization.plot_train_val_metrics(tr_recall_all_folds, te_recall_all_folds_w_aug, te_recall_all_folds_no_aug, te_recall_all_folds_majority_lbl, te_recall_all_folds_majority_prob,
                                                         "Recall", EPOCH_NUM, NUM_FOLDS, str_subdir_save, "recall")
                    Visualization.plot_train_val_metrics(tr_f1_all_folds, te_f1_all_folds_w_aug, te_f1_all_folds_no_aug, te_f1_all_folds_majority_lbl, te_f1_all_folds_majority_prob,
                                                         "F1 Score", EPOCH_NUM, NUM_FOLDS, str_subdir_save, "f1")

                    # Plot boxplot
                    Visualization.plot_boxplot(tr_acc_all_folds, te_acc_all_folds_w_aug, te_acc_all_folds_no_aug, te_acc_all_folds_majority_lbl, te_acc_all_folds_majority_prob, NUM_FOLDS, str_subdir_save, "boxplot")


                    # # Plot confusion matrix for train and validation for each fold
                    # Assuming tr_conf_mat_all_folds and val_conf_all_folds are lists containing confusion matrices for each fold
                    classes = ['Dec1', 'Dec2', 'Dec3']
                    Visualization.plot_confusion_matrices(list(zip(tr_confusion_matrix_all_folds, te_confusion_matrix_all_folds_w_aug, te_confusion_matrix_all_folds_no_aug,
                                                                    te_confusion_matrix_all_folds_majority_lbl, te_confusion_matrix_all_folds_majority_prob )),
                                                                      classes, str_subdir_save, 'conf_matrix')
                    



                    print("\n\nVisualization is finished")


