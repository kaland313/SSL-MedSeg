import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# the cmap below is based on cmap = plt.cm.Set1 (1st and 3rd colors)
my_cmap=np.array(
    [[0.        , 0.        , 0.        , 0.],
    [0.89411765, 0.10196078, 0.10980392, 1.],
    [0.30196078, 0.68627451, 0.29019608, 1.],
    [1.        , 0.49803922, 0.        , 1.]])

my_cmap = matplotlib.colors.ListedColormap(my_cmap)


def plot_acdc_prediction(img, preds, gt_mask, plot_title="", out_path=None):
    """[summary]

    Args:
        img ([type]): [description]
        preds ([type]): [description]
        gt_mask ([type]): [description]
        plot_title (str, optional): Title of the plot. Defaults to "".
        out_path ([type], optional): Full path (incl. filename, where the plot will be saved). 
                                     Don't save plot if None. Defaults to None.
    """
    plt.figure(figsize=(12,16))
    plt.imshow(np.concatenate([img, img], axis=1),
            cmap=plt.cm.bone, interpolation='none', resample=False) # vmax=1024

    overlay = np.concatenate([gt_mask, preds], axis=1)
    plt.imshow(overlay,  cmap=my_cmap, interpolation='none',alpha=0.5, resample=False, vmax=my_cmap.N)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([matplotlib.patches.Patch(facecolor=my_cmap.colors[1],label='Right ventrice (1)'),
                    matplotlib.patches.Patch(facecolor=my_cmap.colors[2],label='Myocardium (2)'),
                    matplotlib.patches.Patch(facecolor=my_cmap.colors[3],label='Left ventrice (3)')])
    plt.gca().legend(handles=handles, ncol=2,)
    plt.axis('off')   
    plt.title(plot_title)
    
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
        
    plt.savefig(out_path, dpi=1200, bbox_inches='tight',pad_inches=0.0)
    plt.close()



def plot_iou_histograms(per_class_ious, labels, output_path=".", bins=50):
    plt.figure()
    plt.hist(np.mean(per_class_ious, axis=1), bins=bins)
    plt.title("Histogram of mean IoUs")
    plt.savefig(os.path.join(output_path, "iou_histogram_mean.pdf"), bbox_inches='tight')
    plt.close()

    plt.figure()
    fig, axs = plt.subplots(2, 2, figsize=(12,9))

    for i in range(per_class_ious.shape[1]):
        ax = axs[(i//2)%2, i%2]
        hist_data = per_class_ious[:,i]
        # hist_data = hist_data[np.logical_not(np.isnan(hist_data))]
        ax.hist(hist_data, bins=bins, range=(0,1))
        ax.set_title(f"Histogram of {labels[i]} IoUs")

    plt.savefig(os.path.join(output_path, "iou_histogram_per_class.pdf"), bbox_inches='tight')
    plt.close()

def plot_confmat(confmat, labels, output_path="."):
    # normalize over the targets
    counts = np.sum(confmat, axis=1, keepdims=True)
    norm_conf = confmat / counts

    plt.figure()
    sns.heatmap(norm_conf, annot=True, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True label')
    plt.title("Pixel-level Confusion Matrix")
    plt.savefig(os.path.join(output_path,'confmat.pdf'))
    plt.close()