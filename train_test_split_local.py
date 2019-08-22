import pandas as pd 
import numpy as np 
import os, sys
import shutil
from shutil import copyfile
import random
import glob


def training_testing_split_local(all_data_dir='classes_path', training_data_dir='classes_path/training_set',
                           testing_data_dir='classes_path/validation_set', testing_data_pct=0.2):
    '''
    *preprocesses any directory which contains subdirectories (that represent classes) into a training & testing set, based on a % of split
    in the same folder that contains the folders of classes; creates testing and training sets
    *splits data locally; use when want to flow from directory
    *example tree before split:
    -all_data_dir
        -1a(first class)
            -1.png
            -2.png
        -2a(second class)
            -3.png
            -4.png
        -3a(third class)
            -5.png
            -6.png
            -7.png
    *example tree after split:
    -all_data_dir
        -training_set
            -1a
                -2.png
            -2a
                -3.png
            -3a
                -5.png
                -7.png
        -testing_set
            -1a
                -1.png
            -2a
                -4.png
            -3a
                -6.png
    
    all_data_dir: folder that contains the folders of classes
    training_data_dir: name of training set folder 
    testing_data_dir: name of testing set folder 
    testing_data_pct: % of testing data, 0.2 means 80-20 split for training and testing
    '''
    training_set_lst = []
    testing_set_lst = []
    num_training_files = 0
    num_testing_files = 0
    category_lst = []
    
   # do not create dir or move files during this loop bc new files will be counted 
    for root, dirs, files in os.walk(all_data_dir): =
        category_name = os.path.basename(root) 
        if category_name in map(os.path.basename, [all_data_dir,training_data_dir,testing_data_dir,
                                                   '.ipynb_checkpoint','.ipynb_checkpoints','']): #skips dir not needed 
            continue #only the categories '1a','2a'...etc left 

        category_lst.append(category_name) 
        
        files.sort()
        random.seed(100)
        random.shuffle(files) #type(files) == list
        current_train = []
        current_test = []
        for index, file in enumerate(files):
            input_file = os.path.join(root, file) #absolute path, e.g. 'data/1a/a.png' 
            if index < (testing_data_pct*len(files)): #at least one file moved to testing_set even if few data bc index starts at 0
                current_test.append(input_file)
                num_testing_files += 1
            else:
                current_train.append(input_file)
                num_training_files += 1  
        training_set_lst.append(current_train) 
        testing_set_lst.append(current_test)
    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")

     # testing and training directories
    if not os.path.exists(training_data_dir):
        os.mkdir(training_data_dir)
    if not os.path.exists(testing_data_dir):
        os.mkdir(testing_data_dir)
        
    print('cat lst',category_lst)
    print('lst len',len(category_lst))
    for i in range(len(training_set_lst)):
        ind = int(category_lst[i][:-1])-1 #order not sequential so this matches class name to index; needs to be changed if classes name are not in the form of 1a, 2a, 3a...
        training_data_category_dir = os.path.join(training_data_dir, category_lst[ind]) 
        testing_data_category_dir = os.path.join(testing_data_dir, category_lst[ind])

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)
        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)
    
        for im in training_set_lst[ind]:
            try:
                shutil.move(im, training_data_category_dir)
            except Exception:
                pass
        for im in testing_set_lst[ind]:
            try:
                shutil.move(im, testing_data_category_dir)    
            except Exception:
                pass
      
    os.chdir(all_data_dir)
    for i in category_lst:
        shutil.rmtree(i)
    os.chdir('../')
       