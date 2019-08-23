import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as ipyw
from math import ceil
from matplotlib.animation import FuncAnimation
from sklearn.metrics import confusion_matrix, accuracy_score


def plot_image_array(image_array, ncols=5, size=1, path=False):
    """
    Plot array of images into grid.
     
    image_array: images of np.array (shape=(NxWxHxC))
    ncols: number of columns of the grid
    size: size of image
    path: set path to save the figure
    """
    nimg = len(image_array)
    nrows = int(ceil(nimg/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols*size,nrows*size))
    if nrows == 0:
        return
    elif ncols == 1:
        for r, ax in zip(np.arange(nrows), axes):
            nth=r
            if nth < nimg:
                ax.imshow(image_array[nth])
            ax.set_axis_off()
            
    elif nrows == 1:
        for c, ax in zip(np.arange(ncols), axes):
            nth=c
            if nth < nimg:
                ax.imshow(image_array[nth])
            ax.set_axis_off()
    else:
        for r, row in zip(np.arange(nrows), axes):
            for c, ax in zip(np.arange(ncols), row):
                nth=r*ncols+c
                if nth < nimg:
                    ax.imshow(image_array[nth])
                ax.set_axis_off()
    
    if path:
        plt.tight_layout()
        plt.savefig(path, dpi=300)
    plt.show()


def make_array_to_gif(filename, array, interval=175):
    """
    plot np.array of images into gif
    
    filename: name of file to save
    array: 4-D array to plot, shape=(number of images, width, height, channel)
    interval: time interval of each image, the higher the slower
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.tight_layout()
    
    def update(i):
        ax.imshow(array[i])
        ax.set_axis_off()
    
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(array), interval=175))
    anim.save('{}.gif'.format(filename), dpi=80, writer="pillow", savefig_kwargs={'pad_inches':0})
    

def plot_confusion_matrix(y_true, y_pred, labels, figsize=(16,16), filename=None, cell_fontsize=16, label_fontsize=20, title_fontsize=20,
                          linewidths=1, cbar=False, normalize_axis=1, x_rotate=0, y_rotate=0, dpi=500):
    """
    Generate matrix plot of confusion matrix as seaborn heatmap with accuracies.
    Plotted image can be saved to disk.
 
    y_true: true label of the data, with shape (nsamples,)
    y_pred: prediction of the data, with shape (nsamples,)
    labels: string array, name the order of class labels in the confusion matrix.  use `clf.classes_` if using scikit-learn models. with shape (nclass,).
    cell_fontsize:  Font size of cell
    label_fontsize: Font size of label
    linewidths: width of line between cells
    figsize: the size of the figure plotted.
    filename: filename of figure file to save
    cbar: Whether to show bar legend
    normalize_axis: which axis to normalize (precision or recall)
    x_rotate: rotation angle of x label
    y_rotate: rotation angle of y label
    dpi: dpi for saving figure
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=normalize_axis, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d / %d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
        
    cm = cm.astype("float")
    for i in range(cm.shape[0]):
        cm[i,:] = cm[i,:] / np.sum(cm[i,:])    
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Ground Truth'
    cm.columns.name = 'Prediction'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues", cbar=cbar,
                annot_kws={"size": cell_fontsize}, linewidths=linewidths, linecolor='w')
      
    ax.set_title("Confusion Matrix, Accuracy={:.3f}".format(acc), fontsize=title_fontsize)
    ax.xaxis.set_tick_params(labelsize=label_fontsize)   
    ax.yaxis.set_tick_params(labelsize=label_fontsize)  
    plt.tight_layout()
    plt.xticks(rotation=x_rotate)
    plt.yticks(rotation=y_rotate)
    plt.autoscale()
    if filename:
        plt.savefig(filename+".png", dpi=dpi, bbox_inches='tight')
    plt.show()