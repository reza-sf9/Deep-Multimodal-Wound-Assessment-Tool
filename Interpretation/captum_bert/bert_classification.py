import os
import torch
import torch.version
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import LayerIntegratedGradients
import pandas as pd
import numpy as np 
from captum.attr import visualization as viz


############## load model and tokenizer ######
print("torch cuda version: "+ torch.version.cuda)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to your saved model, tokenizer, and config file
base_path = 'good_2024_08_15__10_27_54'
model_path = os.path.join(base_path, 'model_1')
tokenizer_path = os.path.join(base_path, 'tokenizer1')
config_path = os.path.join(base_path, 'config.txt')

# Load the saved model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Move the model to the specified device 10.2
model.to(device)
model.eval()

# Load the config.txt file and convert it into a dictionary
training_config = {}
with open(config_path, 'r') as f:
    for line in f:
        key, value = line.strip().split(':')
        key = key.strip()
        value = value.strip()
        
        # Attempt to convert to an appropriate type
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        elif value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        
        training_config[key] = value

# Print out the loaded training configuration to verify
print("Loaded Training Configuration:")
for key, value in training_config.items():
    print(f"{key}: {value}")


##################### helper functions ####################
def load_data(current_dir, dec_scnario):
    # The data loading code from your original script goes here
    # Modify the code to use self.data_path wherever needed
    # This function should return X, y_avlbl, available_img_ids

    print("loading the data ... ")
    # 1. Get the current directory

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

    if dec_scnario == 1:
        # find max value by comparing each element of yExp1 and yExp2
        y = np.maximum(yExp1, yExp2)
    elif dec_scnario == 2:
        y = yExp1
    elif dec_scnario == 3:
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


def load_data_GPT(current_dir, dec_scnario):
    # The data loading code from your original script goes here
    # Modify the code to use self.data_path wherever needed
    # This function should return X, y_avlbl, available_img_ids

    print("loading the data ... ")
    # 1. Get the current directory

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


def load_data_combined(current_dir, dec_scnario):
        # The data loading code from your original script goes here
        # Modify the code to use self.data_path wherever needed
        # This function should return X, y_avlbl, available_img_ids

        print("loading the data ... ")
        # 1. Get the current directory
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

        if dec_scnario == 1:
            # find max value by comparing each element of yExp1 and yExp2
            y = np.maximum(yExp1, yExp2)

        elif dec_scnario == 2:
            y = yExp1
        elif dec_scnario == 3:
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


# Define the prediction and forward functions for Captum
def predict(inputs, attention_mask=None):
    output = model(inputs, attention_mask=attention_mask)
    return output.logits

def forward_func(inputs, attention_mask=None):
    preds = predict(inputs, attention_mask=attention_mask)
    return torch.softmax(preds, dim=1)  # Probability distribution over the classes

# Define a function to summarize attributions
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


################################# Load your dataset
current_dir = os.path.dirname(os.path.abspath(''))
dec_scenario = training_config['dec_scnario']  # Assuming this is the decision scenario you want to use
subfolder = "original"

print("current_dir: " + current_dir)

# Adjust directory levels
levels_to_go_up = 2  # Adjust this number as necessary
data_dir = current_dir
for _ in range(levels_to_go_up):
    data_dir = os.path.split(data_dir)[0]

data_dir = os.path.join(data_dir, 'data/wound_data', subfolder)


# Load data based on the training configuration
if training_config['data'] == 'expert':
    X, y = load_data(data_dir, dec_scenario)
elif training_config['data'] == 'gpt' and dec_scenario == 1:
    X, y = load_data_GPT(data_dir, dec_scenario)
elif training_config['data'] == 'combined':
    X, y = load_data_combined(data_dir, dec_scenario)

# Example: Print the first data point to verify
print("Example sentence from dataset:", X[0])
print("Corresponding label:", y[0])

############## Analysze data using captum 
# Analyze the first sentence from the dataset using Captum
sentence = X[0]  # Replace 0 with the index of the sentence you want to analyze
target_lbl = y[0]  

# Tokenize input and create reference token
input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(device)
ref_input_ids = torch.tensor([[tokenizer.pad_token_id] * input_ids.size(1)], device=device)

attention_mask = torch.ones_like(input_ids)

# Compute Attributions Using Captum
lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)

# Pass the target_lbl as the target
attributions, delta = lig.attribute(
    inputs=input_ids,
    baselines=ref_input_ids,
    additional_forward_args=(attention_mask,),
    target=target_lbl.item(),  # Use the target_lbl here
    return_convergence_delta=True
)

# Summarize attributions and convert tokens
attributions_sum = summarize_attributions(attributions)
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())


################ Visualize atttribution 

# Create visualization
visualization_data = viz.VisualizationDataRecord(
                        attributions_sum,
                        torch.max(torch.softmax(predict(input_ids)[0], dim=0)),
                        torch.argmax(predict(input_ids)),
                        torch.argmax(predict(input_ids)),
                        str(torch.argmax(predict(input_ids))),
                        attributions_sum.sum(),       
                        all_tokens,
                        delta)

viz.visualize_text([visualization_data])

u=1