import tensorflow as tf
from keras import layers, Model
from sklearn.svm import SVC
from transformers import BertTokenizer, TFBertModel
import os
from helper import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

### load data
## load the data
# config the parameters
training_config = {
    'model_name': 'mobilenetv2', # vgg16, resnet50, inceptionv3, efficientnetb0, mobilenetv2
    'epochs_num': 10,
    'jigsaw_active': False,
    'n_jigsaw': 2,
    'learning_rate': 0.00001,
    'cv_fold': True,
    'cv_k': 4,
    'image_type': 'original', # original, cropped
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



print("\n\n Defining the model ... \n\n")
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
image_features = layers.GlobalAveragePooling2D()(image_features)
image_model = Model(inputs=image_input, outputs=image_features)

# Define text model (BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_input = layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32)
text_model = TFBertModel.from_pretrained('bert-base-uncased')(text_input)[0][:, 0, :]  # Use CLS token
text_model = layers.Dense(128, activation='relu')(text_model)

# Concatenate modalities
concatenated = layers.Concatenate()([image_features, text_model])

# Classifier
svm_classifier = SVC(kernel='linear')
# Compile model
model = Model(inputs=[image_input, text_input], outputs=concatenated)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


print("\n\nTraining the model ...\n\n ")

# Tokenize and pad text data
train_texts_tokenized = tokenizer(train_texts_processed, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='np')
val_texts_tokenized = tokenizer(val_texts_processed, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='np')

train_texts_np = train_texts_tokenized['input_ids']
val_texts_np = val_texts_tokenized['input_ids']

# Ensure that attention masks are also provided
train_attention_masks_np = train_texts_tokenized['attention_mask']
val_attention_masks_np = val_texts_tokenized['attention_mask']


# Train model
epoch_ = 20
batch_ = 16
model.fit([train_images_np, train_texts_np], train_labels_np, validation_data=([val_images_np, val_texts_np], val_labels_np), epochs=epoch_, batch_size=batch_)

# Extract concatenated features for SVM training
train_concatenated_features = model.predict([train_images_np, train_texts_np])
val_concatenated_features = model.predict([val_images_np, val_texts_np])

# Train SVM classifier
svm_classifier.fit(train_concatenated_features, train_labels)

# Evaluate SVM classifier
svm_score = svm_classifier.score(val_concatenated_features, val_labels)
print("SVM accuracy:", svm_score)
