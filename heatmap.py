import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

## Code courtesy of matplotlib documentation
## modified for use with thesis

def main():
  vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                "potato", "wheat", "barley"]
  farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
             "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

  harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                      [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                      [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                      [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                      [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                      [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                      [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
  title = "Harvest of local farmers (in tons/year)"
                      
  # fig, ax = plt.subplots()

  #im, cbar = heatmap(harvest, vegetables, farmers,cmap="magma", cbarlabel="harvest [t/year]",
  # title="Example",saveFile="filename_b")
  # #texts = annotate_heatmap(im, valfmt="{x:.1f} t")

  # fig.tight_layout()
  #plt.show()
  
  categories = ['GA Sim', 'GA DD', 'Greedy Sim', 'Greedy Disc Deg', 'Random K', 'Max Degree', 'Max Betweenness Wtd']
  x = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]
  y = [i*.005 for i in range(int(np.log(100))*4,0,-1)]
  k = 2
  title = "ER graphs under IC, Init Inf=" + str(k)
  x_title = "Infection Probability"
  y_title = "Uniform Edge Probability Pij"

  data = np.array([
    [1, 3, 2, 6, 1, 0, 6, 2, 3, 0],
    [2, 5, 5, 5, 0, 6, 3, 0, 6, 0],
    [1, 3, 2, 3, 0, 4, 2, 6, 2, 0],
    [5, 5, 3, 2, 1, 4, 6, 2, 1, 0],
    [1, 1, 2, 3, 2, 0, 5, 0, 0, 0],
    [3, 5, 6, 6, 6, 5, 0, 0, 6, 0],
    [6, 6, 3, 5, 6, 4, 0, 0, 0, 0],
    [6, 2, 2, 3, 6, 0, 0, 0, 0, 0],
    [5, 5, 5, 3, 5, 0, 1, 0, 0, 0],
    [2, 3, 6, 1, 3, 2, 1, 0, 0, 0],
    [1, 3, 5, 1, 1, 6, 3, 6, 4, 0],
    [2, 5, 5, 2, 1, 1, 2, 0, 1, 0],
    [3, 2, 1, 3, 1, 2, 2, 2, 5, 0],
    [2, 3, 5, 1, 5, 1, 5, 3, 1, 0],
    [5, 5, 3, 2, 2, 1, 1, 0, 2, 0],
    [2, 2, 3, 2, 3, 0, 0, 0, 0, 0]])
    
  figure_title = title + "\n  Algorithm with Max Spread"
  filename = "heatmaps/" + title + " Dominators"
  im = category_map(data,np.round(y,2),np.round(x,2),categories,
    title=figure_title,x_title=x_title,y_title=y_title,saveFile=filename)

  # im = category_map(data, y, x, categories,
  #                   cmap="Set3", title="Example Title",saveFile =filename)
  #texts = annotate_heatmap(im, valfmt="{x:.1f} t")

  #fig.tight_layout()
  plt.show()
  
def build_legend_patches(categories,cmap):
  cmap = plt.get_cmap(cmap)
  colors = cmap(np.linspace(0, 1, len(categories)))
  patchList = []
  for c in range(len(categories)):
          patch_entry = mpatches.Patch(color=colors[c], label=categories[c])
          patchList.append(patch_entry)

  plt.legend(handles=patchList,loc='center left', bbox_to_anchor=(1, 0.5))  
  
## EXAMPLE CALL ##
##   im = category_map(data, y, x, categories,
##     cmap="Set3", title="Example Title",saveFile="filename") ##  
  
def category_map(data, row_labels, col_labels, categories, cmap="tab10", ax=None, cmax=None,
            cbar_kw={}, cbarlabel="", title = "Default",  x_title=" ",y_title=" ", saveFile = None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M), values in [0,len(categories)-1].
    categories
        a list of category labels the indices in data correspond to
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    #plt.clf()
    if not ax:
        ax = plt.gca()
        # Shrink current axis by 20%

    # Plot the heatmap
    cmap = plt.get_cmap(cmap)
    if cmax == None:
      cmax = len(categories) - 1
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=cmax, clip=True)
    im = ax.imshow(data, cmap, norm = norm)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Horizontal Axis Position
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Legend 
    build_legend_patches(categories,cmap)
    
    ax.set_xlabel(x_title,fontsize=12)
    ax.set_ylabel(y_title,fontsize=12)  # 10
    ax.set_title(title,fontsize=12) #10, pad=-10)
    
    if saveFile != None:
      plt.savefig(saveFile + '.png', bbox_inches='tight') 

    return im

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", title = "Default", x_title=" ",y_title=" ",saveFile = None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    #plt.clf()
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw) # pad=0.1
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on bottom
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # plt.title(title)
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    ax.set_xlabel(x_title,fontsize=12)
    ax.set_ylabel(y_title,fontsize=12)  # 10
    ax.set_title(title,fontsize=12) #10, pad=-10)
  
    
    if saveFile != None:
      plt.savefig(saveFile + '.png', bbox_inches='tight') 

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
    