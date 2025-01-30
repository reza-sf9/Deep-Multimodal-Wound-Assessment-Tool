import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torchmetrics.utilities.data import to_categorical 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

class DataLoader:
    def __init__(self, data_path, dec_scnario=1):
        self.data_path = data_path
        self.dec_scnario = dec_scnario

    def load_data(self):
        # The data loading code from your original script goes here
        # Modify the code to use self.data_path wherever needed
        # This function should return X, y_avlbl, available_img_ids

        print("loading the data ... ")
        # 1. Get the current directory
        current_dir = self.data_path

        # 2. Create the relative path to the db.xlsx file in the data subdirectory

        # move one step up
        dir_up = os.path.split(current_dir)[0]

        xlsx_path = os.path.join(dir_up, 'db_info.xlsx')

        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(xlsx_path)

        # get information of the column with header Decision_Loretz and store it in a numpy array
        yExp1 = df['Decision_Dunn'].values
        yExp2 = df['Decision_Loretz'].values

        # remove non-number from yExp1 and yExp2
        yExp1 = [x for x in yExp1[:-1] if isinstance(x, (int, float))]
        yExp2 = [x for x in yExp2[:-1] if isinstance(x, (int, float))]

        if self.dec_scnario == 1:
            # find max value by comparing each element of yExp1 and yExp2
            y = np.maximum(yExp1, yExp2)

        elif self.dec_scnario == 2:
            y = yExp1
        elif self.dec_scnario == 3:
            y = yExp2
        
        # rmove nan from y 
        y = [x for x in y if not np.isnan(x)]

        # starts y from 0 
        y = np.array(y) - 1

        # convert y to int 
        y = y.astype(int)

        # convert to long tensor
        y = torch.tensor(y, dtype=torch.long)


        # get all headers column names
        headers = df.columns.values


        # Get the text data from the 'Loretz' column
        loretz_texts = df['Loretz Comments'].astype(str).values
        # remove nan from lorez_texts
        loretz_texts = [text for text in loretz_texts if text != 'nan']

        # Get the text data from the 'Dunn' column
        dunn_texts = df['Dunn Comments'].astype(str).values

        # create texts by combining loretz_texts and dunn_texts
        texts = []
        for i in range(len(loretz_texts)):
            texts.append(loretz_texts[i] + ' ' + dunn_texts[i])


        X = texts
        

        print("data is loaded successfully ... \n\n\n")

        return X, y
    

    def load_data_GPT(self):
        # The data loading code from your original script goes here
        # Modify the code to use self.data_path wherever needed
        # This function should return X, y_avlbl, available_img_ids

        print("loading the data ... ")
        # 1. Get the current directory
        current_dir = self.data_path

        # 2. Create the relative path to the db.xlsx file in the data subdirectory

        # move one step up
        dir_up = os.path.split(current_dir)[0]

        xlsx_path = os.path.join(dir_up, 'oneFolder/dbPolicy1_GPT.xlsx')

        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(xlsx_path)

        # get all headers column names
        headers = df.columns.values
        
        # get information of the column with header Decision_Loretz and store it in a numpy array
        y= df[headers[1]].values
        
        # rmove nan from y 
        y = [x for x in y if not np.isnan(x)]

        # convert y to int 
        # y = y.astype(int)

        # convert to long tensor
        y = torch.tensor(y, dtype=torch.long)


        


        # Get the text data from the 'Loretz' column
        GPT_text = df[headers[2]].astype(str).values

        # remove nan from lorez_texts
        GPT_text = [text for text in GPT_text if text != 'nan']


        X = GPT_text
        

        print("data is loaded successfully ... \n\n\n")

        return X, y
    

    def load_data_combined(self):
        # The data loading code from your original script goes here
        # Modify the code to use self.data_path wherever needed
        # This function should return X, y_avlbl, available_img_ids

        print("loading the data ... ")
        # 1. Get the current directory
        current_dir = self.data_path

        # 2. Create the relative path to the db.xlsx file in the data subdirectory

        # move one step up
        dir_up = os.path.split(current_dir)[0]

        ##################################### extract texts from GPT and expert data
        ############################# GPT

        xlsx_path_gpt = os.path.join(dir_up, 'oneFolder/dbPolicy1_GPT.xlsx')

        # Read the Excel file into a pandas DataFrame
        df_gpt = pd.read_excel(xlsx_path_gpt)

        # get all headers column names
        headers_gpt = df_gpt.columns.values
        
        # get information of the column with header Decision_Loretz and store it in a numpy array
        y_gpt= df_gpt[headers_gpt[1]].values
        
        # rmove nan from y 
        y_gpt = [x for x in y_gpt if not np.isnan(x)]

        # convert to long tensor
        y_gpt = torch.tensor(y_gpt, dtype=torch.long)


        # Get the text data from the 'Loretz' column
        GPT_text = df_gpt[headers_gpt[2]].astype(str).values
        # remove nan from lorez_texts
        GPT_text = [text for text in GPT_text if text != 'nan']


        ############################# expert 
        xlsx_path_expert = os.path.join(dir_up, 'db_info.xlsx')

        # Read the Excel file into a pandas DataFrame
        df_expert = pd.read_excel(xlsx_path_expert)

        # get all headers column names
        headers_expert = df_expert.columns.values

        # Get the text data from the 'Loretz' column
        loretz_texts = df_expert['Loretz Comments'].astype(str).values
        # remove nan from lorez_texts
        loretz_texts = [text for text in loretz_texts if text != 'nan']

        # Get the text data from the 'Dunn' column
        dunn_texts = df_expert['Dunn Comments'].astype(str).values

        # create texts by combining loretz_texts and dunn_texts
        texts_combined = []
        for i in range(len(loretz_texts)):
            texts_combined.append(loretz_texts[i] + ' ' + dunn_texts[i] + ' ' + GPT_text[i])


        ########################## extract labels 
        # get information of the column with header Decision_Loretz and store it in a numpy array
        yExp1 = df_expert['Decision_Dunn'].values
        yExp2 = df_expert['Decision_Loretz'].values

        # remove non-number from yExp1 and yExp2
        yExp1 = [x for x in yExp1[:-1] if isinstance(x, (int, float))]
        yExp2 = [x for x in yExp2[:-1] if isinstance(x, (int, float))]

        if self.dec_scnario == 1:
            # find max value by comparing each element of yExp1 and yExp2
            y = np.maximum(yExp1, yExp2)

        elif self.dec_scnario == 2:
            y = yExp1
        elif self.dec_scnario == 3:
            y = yExp2
        
        # rmove nan from y 
        y = [x for x in y if not np.isnan(x)]

        # starts y from 0 
        y = np.array(y) - 1

        # convert y to int 
        y = y.astype(int)

        # convert to long tensor
        y = torch.tensor(y, dtype=torch.long)


        X = texts_combined

        print("data is loaded successfully ... \n\n\n")

        return X, y

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_encode_tr_te(self, X, y):
        """
        Splits the data into training and test sets and encodes the labels.

        Parameters:
            X: Features dataset.
            y: Labels dataset.

        Returns:
            X_train, X_test, y_train, y_test: Split and encoded datasets.
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        # Convert labels to categorical
        y_train_encoded = to_categorical(y_train)
        y_test_encoded = to_categorical(y_test)

        return X_train, X_test, y_train_encoded, y_test_encoded


    # add K-fold cross validation to the tr-te data
    

    def split_encode_kfold(self, X, y, n_splits=3):
        """
        Splits the data into K folds for cross-validation and encodes the labels.

        Parameters:
            X: Features dataset.
            y: Labels dataset.
            n_splits: Number of folds for cross-validation (default is 5).

        Returns:
            List of tuples containing (X_train_fold, X_val_fold, y_train_fold, y_val_fold) for each fold.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        encoded_folds = []
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = [X[index] for index in train_idx]
            X_val_fold = [X[index] for index in val_idx]

            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # # Convert labels to categorical
            # y_train_encoded = to_categorical(y_train_fold)
            # y_val_encoded = to_categorical(y_val_fold)

            encoded_folds.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))

        return encoded_folds


# create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])





def compute_metrics(p):
    print(type(p))
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


####### have the accuracy on the training set
from transformers import TrainerCallback

class TrainAccuracyCallback(TrainerCallback):
    def __init__(self, model, train_dataset, test_dataset, tokenizer):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.epoch_accuracies = []
        self.test_accuracies = []  

    def on_epoch_end(self, args, state, control, **kwargs):
        # Calculate accuracy on the training set
        train_accuracy = self.calculate_accuracy_on_train_set()
        self.epoch_accuracies.append(train_accuracy)
        print(f"Epoch {state.epoch} - Training Accuracy: {train_accuracy}")

        # Calculate accuracy on the test set
        test_accuracy = self.calculate_accuracy_on_test_set()
        self.test_accuracies.append(test_accuracy)  # Add this line
        print(f"Epoch {state.epoch} - Test Accuracy: {test_accuracy}")

    def calculate_accuracy_on_train_set(self):
        # Set the model to evaluation mode
        self.model.eval()

        # Initialize variables to store correct predictions and total predictions
        correct_predictions = 0
        total_predictions = 0

        # Iterate through the training dataset
        for i in range(len(self.train_dataset)):
            # Convert token IDs to text
            input_text = self.tokenizer.decode(self.train_dataset[i]['input_ids'], skip_special_tokens=True)

            # Tokenize the input
            input_tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

            # Move the input to the GPU
            input_tokens = {key: value.to('cuda') for key, value in input_tokens.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**input_tokens)

            # Get the predicted class
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

            # Get the true class label (update key based on your dataset structure)
            true_class = self.train_dataset[i]['labels']

            # Update correct and total predictions
            correct_predictions += 1 if predicted_class == true_class else 0
            total_predictions += 1

        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        return accuracy

    def calculate_accuracy_on_test_set(self):
        # Set the model to evaluation mode
        self.model.eval()

        # Initialize variables to store correct predictions and total predictions
        correct_predictions = 0
        total_predictions = 0

        # Iterate through the test dataset
        for i in range(len(self.test_dataset)):
            # Convert token IDs to text
            input_text = self.tokenizer.decode(self.test_dataset[i]['input_ids'], skip_special_tokens=True)

            # Tokenize the input
            input_tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

            # Move the input to the GPU
            input_tokens = {key: value.to('cuda') for key, value in input_tokens.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**input_tokens)

            # Get the predicted class
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

            # Get the true class label (update key based on your dataset structure)
            true_class = self.test_dataset[i]['labels']

            # Update correct and total predictions
            correct_predictions += 1 if predicted_class == true_class else 0
            total_predictions += 1

        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        return accuracy
