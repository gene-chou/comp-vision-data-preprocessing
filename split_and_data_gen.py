import os,sys
import numpy as np
import random
import cv2
import keras


def get_train_test_set(data_dir, split_pct=0.2, random_state=False):

    '''
    *splits data into training, testing sets with specific structure
    *example of what training_set, testing_set would look like:
        training_set = {'class_1': [2.png],
                        'class_2':[3.png, 4.png],
                        'class_3':[7.png, 8.png]
                       }
        testing_set = {'class_1': [1.png],
                       'class_2':[5.png],
                       'class_3':[6.png]
                      }
    *the reason for this structure is 
     1. I find this easier to check or balance data
     2. this structure is needed for balanced_data_generator() below,
        which balances each training batch with the same number of images
        from each class so that model does not become biased even if 
        classes sizes are disproportionate

    *example tree of data_dir:
    -data_dir
        -class_1 folder
            -1.png
            -2.png
        -class_2 folder
        -class_3 folder

    data_dir: directory for data containing classes
    split_pct: % of testing_set 
    random_state: random seed
    '''
   files_lst = []
   for root, dirs, files in os.walk(data_dir):
     files.sort() #sort before shuffle so shuffle gets same results 
     if not random_state: 
         random.seed(100)
     random.shuffle(files) #no need to reshuffle during split
     files_lst.append(files)
   files_lst.pop(0) #first lst is empty lst (result of os.walk; print to confirm)

   testing_set = {}
   training_set = {}
   tr_count, tt_count = 0, 0
   class_names = ['class_1', 'class_2', 'class_3'] #make sure same order as files_lst
   for i in range(len(files_lst)):
     split_idx = 0
     tr_lst=[]
     tt_lst=[]
     for x in files_lst[i]: #8-2 split for each class
         if split_idx < len(files_lst[i])*split_pct: #at least one file moved to testing_set even if few data
             tt_lst.append(x)
             tt_count+=1
         else:
             tr_lst.append(x)
             tr_count+=1
         split_idx+=1
     training_set[class_names[i]] = tr_lst
     testing_set[class_names[i]] = tt_lst
   print('training count: {}\ntesting count: {}'.format(tr_count, tt_count))
   return training_set, testing_set


def balanced_data_generator(data_ids, patch_size=(256, 256, 3), batch_size=32, shuffle=True):
    '''
    *balances each training batch with the same number of images
     from each class so that model does not become biased even if 
     classes sizes are disproportionate
    *preferably batch_size should be multiple of number of classes 
    *resize, augmentation...etc can be done here on the fly or in preprocessing
    *quick example of using this generator with keras:
        resnet = ResNet50(include_top=False, pooling='avg', input_shape=(128,128,3), weights='imagenet')
        out = keras.layers.Dense(units=num_of_classes, activation='softmax', name='resnet50_output')(resnet.output)
        model = keras.models.Model(inputs=resnet.input, outputs=out)
        model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer=Adam())
        train_datagen = balanced_data_generator(training_set)
        test_datagen = balanced_data_generator(testing_set)
        model.fit_generator(train_datagen, validation_data=test_datagen)

    data_ids: training_set or testing_set in the format of get_train_test_set() above; len = 4 if 4 classes to classify
    patch_size: the dimensions of the images
    '''
    
    batch = np.zeros(shape=(batch_size, *patch_size), dtype=np.int32)
    label = []
    b = 0
    i = 0
    idx = [0 for i in range(len(data_ids))] #2 entries for 2 classes
    
    while True:
        if idx[i] >= len(data_ids[ list(data_ids)[i] ]): #index out of bounds, reset 
            idx[i]=0
        #list(data_ids)[i] is the key/class name
        image_id = data_ids[ list(data_ids)[i] ][idx[i]] #e.g. image_id=='1.png'
       
        try:
            image = cv2.imread( '{}/{}/{}'.format(crop_im_dir, list(data_ids)[i], image_id) )[:,:,::-1] #check RGB/BGR
            #resize image here to patch_size as needed, if not resized in preprocessing

        except:
            continue

        batch[b] = np.array(image)[...,:3]
        label.append(class_name_lst.index(list(data_ids)[i]))
        b+=1
        idx[i]+=1
        if b % (batch_size/len(data_ids)) == 0: #each batch is balanced 
            i+=1

        if b == batch_size:
            #batch = augmentation.augment_images(batch) #augment if needed
            #batch = keras.applications.resnet50.preprocess_input(batch) #preprocess as needed
            label = keras.utils.to_categorical(label, len(data_ids)) #needed for binary as well 
            if shuffle:
                p = np.random.permutation(len(batch))
                yield batch[p], label[p]
            else:
                yield batch, label
            b = 0
            i = 0
            batch = np.zeros(shape=(batch_size, *patch_size), dtype=np.int32)
            label = []