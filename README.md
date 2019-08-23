# comp-vision-data-preprocessing
tools for data preprocessing, written specifically for computer vision projects but potentially applicable elsewhere

#### Files
`split_and_data_gen.py`&nbsp;  
After locating a directory with tree  
>     -data_dir  
>       -class_1 folder  
>         -1.png   
>         -2.png  
>       -class_2 folder  
>       -class_3 folder  

splits data into training, testing sets with structure  
>     training_set = {'class_1':[2.png],  
>                     'class_2':[3.png, 4.png],  
>                     'class_3':[7.png, 8.png]  
>                    }  
>     testing_set = {'class_1':[1.png],  
>                    'class_2':[5.png],  
>                    'class_3':[6.png]  
>                   }  
Then create balanced data generator--each training batch has the same number of images from each class so that model does not become biased even if classes sizes are disproportionate.

  
\
`train_test_split_local.py`  
Given directory containing folders of classes, splits data locally. Useful when want to flow from directory.  
Example tree before split  
>     -data_dir  
>       -class_1 folder  
>         -1.png   
>         -2.png  
>         -3.png
>       -class_2 folder  
>       -class_3 folder  
Example tree after split 
>     -data_dir  
>       -training_set
>           -class_1 folder  
>             -1.png   
>             -3.png  
>           -class_2 folder  
>           -class_3 folder  
>       -testing_set
>           -class_1 folder  
>            -2.png   
>           -class_2 folder  
>           -class_3 folder    

\
`resize_retain_spatial.py`  
Functions for resizing images while retaining spatial information; i.e. photo does not become "fatter" or "thinner". Essentially creates a black background of desired dimensions and pastes resized photo onto it. Note that if h,w proportion of resized im is very different from original, new im will have a lot of black edges.  
Supports numpy arrays and PIL images.  

\
`plot.py`  
Functions for plotting and visualizing images, confusion matrix. 
