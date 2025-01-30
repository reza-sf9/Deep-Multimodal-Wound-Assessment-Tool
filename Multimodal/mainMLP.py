import tensorflow as tf
from keras import layers, Model
from transformers import BertTokenizer, TFBertModel
from helper import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os

### Load data
# Load the data
training_config = {
    'model_name': 'mobilenetv2',  # vgg16, resnet50, inceptionv3, efficientnetb0, mobilenetv2
    'epochs_num': 10,
    'jigsaw_active': False,
    'n_jigsaw': 2,
    'learning_rate': 0.00001,
    'cv_fold': True,
    'cv_k': 4,
    'image_type': 'cropped',  # original, cropped
    'batch_size_list': [16],
    'freeze_layer_list': [1],
    'data_agumentation': False,
    'thr_acc': .2
}
dec_scnario = 1 # dec_scnario = 1, 2, 3, 4


# load data
current_dir = os.path.dirname(os.path.abspath(__file__))

if training_config['image_type'] == 'cropped':
    subforlder= 'cropped'
elif training_config['image_type']  == 'original':
    subforlder= 'original'

# Define the number of levels you want to move up
levels_to_go_up = 2  # Change this to the number of levels you need to go up
# Move up the directory structure
data_dir = current_dir
for _ in range(levels_to_go_up):
    data_dir = os.path.split(data_dir)[0]
    # print(desired_dir)

data_dir = os.path.join(data_dir, 'data/wound_data')
data_dir = os.path.join(data_dir, subforlder)
data_loader = DataLoader(data_dir, dec_scnario)
X, texts, y_avlbl, available_img_ids = data_loader.load_data()

# plot histogram of y_avlbl
import matplotlib.pyplot as plt

plt.figure()
# Count occurrences of each treatment category
counts = [np.sum(y_avlbl == 0), np.sum(y_avlbl == 1), np.sum(y_avlbl == 2)]

# Create a bar plot
plt.bar([0, 1, 2], counts, color=['blue', 'orange', 'green'])

# Set xticks and labels
plt.xticks([0, 1, 2], ['current treatment', 'referral-non urgent', 'referral-urgent'])
# make xticks vertical
plt.ylabel('Number of data per class')
# Show the plot
plt.tight_layout()
# make all fonts bigger
plt.rc('font', size=15)

# xticks font size =10
plt.xticks(fontsize=12)

plt.show()





# Split the data into training and validation sets
train_images, val_images, train_texts, val_texts, train_labels, val_labels = train_test_split(X, texts, y_avlbl, test_size=0.2, random_state=42)

# Convert train and validation image lists to numpy arrays
train_images_np = np.array(train_images)
val_images_np = np.array(val_images)

# Ensure train and validation text lists are properly formatted
train_texts_processed = [str(text) for text in train_texts]
val_texts_processed = [str(text) for text in val_texts]

# Convert train and validation labels to numpy arrays
train_labels_np = np.array(train_labels)
val_labels_np = np.array(val_labels)

# Define constants
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NUM_CHANNELS = 3
MAX_SEQ_LENGTH = 128
VOCAB_SIZE = 30522  # For BERT-base-uncased
EMBEDDING_DIM = 768  # For BERT-base-uncased

# Define image model (MobileNetV2)
image_input = layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS), include_top=False, weights='imagenet')
image_features = base_model(image_input)
image_features = layers.GlobalAveragePooling2D()(image_features) ## this is my image embedding vector
image_model = Model(inputs=image_input, outputs=image_features)

# Define text model (BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_input = layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32)
text_model = TFBertModel.from_pretrained('bert-base-uncased')(text_input)[0][:, 0, :]  # Use CLS token
text_model = layers.Dense(128, activation='relu')(text_model)

# Concatenate modalities
concatenated = layers.Concatenate()([image_features, text_model])

# Define MLP classifier
mlp_output = layers.Dense(64, activation='relu')(concatenated)
mlp_output = layers.Dropout(0.5)(mlp_output)
mlp_output = layers.Dense(3, activation='softmax')(mlp_output)  # Adjust the number of output classes as per your problem

# Compile model
model = Model(inputs=[image_input, text_input], outputs=mlp_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Tokenize and pad text data
train_texts_tokenized = tokenizer(train_texts_processed, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='tf')
val_texts_tokenized = tokenizer(val_texts_processed, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='tf')

train_texts_np = train_texts_tokenized['input_ids']
val_texts_np = val_texts_tokenized['input_ids']

# Ensure that attention masks are also provided
train_attention_masks_np = train_texts_tokenized['attention_mask']
val_attention_masks_np = val_texts_tokenized['attention_mask']

# Train model
epoch_ = 50
batch_ = 16
model.fit([train_images_np, train_texts_np], train_labels_np, validation_data=([val_images_np, val_texts_np], val_labels_np), epochs=epoch_, batch_size=batch_)

# Evaluate model
val_loss, val_accuracy = model.evaluate([val_images_np, val_texts_np], val_labels_np)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)
