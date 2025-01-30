import os
import numpy as np
import pandas as pd
from PIL import Image
import csv

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

        # get information of the column with header Image_ID and store it in a numpy array
        image_ids = df['Image_ID'].values

        # check all ids in image_ids and if it is int go to the subdirectory data/DrLoretzUMASS and based on image id read (id).jpg file and store it in a 3d numpy array
        # if it is a string and starts with Lei then go to the subdirectory data/Lei-Web and based on image id read (id).bmp file and store it in a 3d numpy array
        # if it is a string and starts with Web then go to the subdirectory data/Web and based on image id read (id).jpg file and store it in a 3d numpy array
        # all above images should be saved in a signle numpy array with shape (number of images, 224, 224, 3)

        # create an empty numpy array with shape (number of images, 224, 224, 3)
        X = np.empty((len(image_ids), 224, 224, 3))

        # in image_ids there is a string value "TOTAL" find and remove it and all elements after it
        image_ids = image_ids[:np.where(image_ids == 'TOTAL')[0][0]]

        # crate an empty numpy variable for image_id and name it available_img_ids, I want to populate it in the below loop with all available image ids its elements might be number or string
        available_img_ids = np.empty(len(image_ids), dtype=object)

        # create an empty numpy vector (int) for available lables - lenth is unkonen an I will populate it in the below loop
        y_avlbl = []

        # loop over all image ids

        cnt_ = -1
        texts = []
        for i in range(len(image_ids)):
            # print the current image id
            # print(image_ids[i])
            # get the image id
            image_id = image_ids[i]

            excluded_ids = [117, 119, 121, 502, 779, 897, 1461, 1588]
            # find these images 203, 354, 388, 1696

            # get the comments
            # print('i = %d'%i)
            loretz_comments = df['Loretz Comments'].astype(str).values[i]
            Dunn_comments = df['Dunn Comments'].astype(str).values[i]
            # check if Dunn_comments is not 'nan'
            if Dunn_comments != 'nan':
                note_i = loretz_comments + ' ' + Dunn_comments
            else:
                note_i = loretz_comments


            # check if image id is int and not equal to 117, 119, 121, 203, 354, 388, 502, 779, 897, 1461, 1588, 1696
            if isinstance(image_id, int) and image_id not in excluded_ids:

                # for image_id, retrieve Loretz_Comments

                texts.append(note_i)


                cnt_ += 1
                # if it is int then go to the subdirectory data/DrLoretzUMASS and based on image id read (id).jpg file and store it in a 3d numpy array
                image_path = os.path.join(current_dir, 'DrLoretzUMASS', str(image_id) + '.jpg')
                # load image from image_path
                img = Image.open(image_path)
                # Resize the image to 224x224 pixels
                img_resized = img.resize((224, 224))
                # Convert the Pillow Image object to a numpy array
                img_array = np.array(img_resized)
                # store image in X
                X[cnt_] = img_resized
                # store image id in available_img_ids
                available_img_ids[cnt_] = image_ids[i]
                # store label in y_available
                y_avlbl.append(y[i] - 1)








            # check if image id is string and starts with Lei
            elif isinstance(image_id, str) and image_id.startswith('Lei'):
                texts.append(note_i)

                cnt_ += 1
                # if it is string and starts with Lei then go to the subdirectory data/Lei-Web and based on image id read (id).bmp (remove Lei from image_id and use remainting.bmp)file and store it in a 3d numpy array
                image_path = os.path.join(current_dir, 'Lei-Web', str(image_id[3:]) + '.bmp')
                img = Image.open(image_path)
                # Resize the image to 224x224 pixels

                img_resized = img.resize((224, 224))
                # Convert the Pillow Image object to a numpy array
                img_array = np.array(img_resized)

                a = img_array[:, :, 0:3]

                # store image in X
                X[cnt_] = img_array[:, :, 0:3]
                # store image id in available_img_ids
                y_avlbl.append(y[i] - 1)


            # check if image id is string and starts with Web
            elif isinstance(image_id, str) and image_id.startswith('Web'):
                texts.append(note_i)
                cnt_ += 1
                # if it is string and starts with Web then go to the subdirectory data/Web and based on image id read (id).jpg file and store it in a 3d numpy array
                image_path = os.path.join(current_dir, 'Web', str(image_id[3:]) + '.jpg')
                # load image from image_path
                img = Image.open(image_path)
                # Resize the image to 224x224 pixels
                img_resized = img.resize((224, 224))
                # Convert the Pillow Image object to a numpy array
                img_array = np.array(img_resized)
                # store image in X
                X[cnt_] = img_resized
                # store image id in available_img_ids
                y_avlbl.append(y[i] - 1)


        ## RSF: end of write csv

        # convert lables to numpy array
        y_avlbl = np.array(y_avlbl)

        # remove all None elements from available_img_ids
        available_img_ids = available_img_ids[:cnt_ + 1]

        # in X only keep cnt_ first images
        X = X[:cnt_ + 1]

        print("data is loaded successfully ... \n\n\n")

        return X, texts, y_avlbl
