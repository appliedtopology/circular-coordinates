import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import numpy as np
import math
from .scroll import ScrollableWindow





def plot_PCA(data_pca,vert,xlabel='Principal Component 1',ylabel='Principal Component 2',fig_size=(10,10),ax=None, pt_style=None):
    """
    Function to plot data_pca with circular cordinates represented as colors on the rgb color wheel
    ----------
    Parameters:
        data_pca :(ndarray) 
            pca of the input data in 2 dimensions
        vert : ndarray
                array of circulare coordinates mapped to (0,1)
        xlabel : string
                label of the x-axis
        ylabel : string
                label of the y-axis
        fig_size : tuple
                size of plotted figure(default=(10,10))
        ax: AxesSubplot
            Axes that should be used for plotting (Default: None)
        pt_style: dict
            argments passed to `ax.scatter` for style of points.
        

    """
    pt_kwargs = {}
    if pt_style is not None:
        pt_kwargs.update(pt_style)

    plt.figure(figsize=fig_size)   
    if ax is None:
        ax = plt.axes()
    ax.scatter(data_pca[:,0], data_pca[:,1], c=vert,cmap=plt.cm.hsv, **pt_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def plot_check_all(rips,vert,fig_size=(10,10),ax=None, pt_style=None):
    """
    Function to plot circular cordinates of all persistance barcodes
    ----------
    Parameters:
        rips :(dict) 
            ripser output dictionary
        vert : ndarray
                array of circulare coordinates mapped to (0,1)
        fig_size : tuple
                size of plotted figure(default=(10,10))
        ax: AxesSubplot
            Axes that should be used for plotting (Default: None)
        pt_style: dict
            argments passed to `ax.scatter` for style of points.
        

    """
    pt_kwargs = {}
    if pt_style is not None:
        pt_kwargs.update(pt_style)

    plt.figure(figsize=fig_size)
    if ax is None:
        ax = plt.axes()
    for ind,eps in enumerate(rips["dgms"][1]):
        eps=eps[1]
        verti=vert[ind]
        y=np.ones(len(verti))*eps
        ax.scatter(verti, y, c=verti,cmap=plt.cm.hsv, **pt_kwargs)
    ax.set_xlabel('Circular Cordinates')
    ax.set_ylabel('epsilon')
    plt.show()   
        
def plot_check_max(vert,vert_list,fig_size=(10,10),ax=None, pt_style=None):
    """
    Function to plot circular cordinates over the largest persistance barcode
    ----------
    Parameters:
        vert : ndarray
                array of circulare coordinates mapped to (0,1)
        vert_list : list
               list of points between birth and death of the largest persistance barcode
        fig_size : tuple
                size of plotted figure(default=(10,10))
        ax: AxesSubplot
            Axes that should be used for plotting (Default: None)
        pt_style: dict
            argments passed to `ax.scatter` for style of points.
        

    """
    pt_kwargs = {}
    if pt_style is not None:
        pt_kwargs.update(pt_style)
    plt.figure(figsize=fig_size)
    if ax is None:
        ax = plt.axes()
    for ind,eps in enumerate(vert_list):
        verti=vert[ind]
        y=np.ones(len(verti))*eps
        ax.scatter(verti, y, c=verti,cmap=plt.cm.hsv, **pt_kwargs)
    ax.set_xlabel('Circular Cordinates')
    ax.set_ylabel('epsilon')
    plt.show()


def plot_eps(p1,vert,dist=None,rips=None,type=None,vert_list=None,fig_size=(10,10),ax=None, pt_style=None,scrollable=False):

        """
        Function to plot external data with the circular coordinates as a scatter plot
        ----------
        Parameters:
            p1 :(ndarray) 
                external data
            vert : ndarray
                    array of circulare coordinates mapped to (0,1)
            type : str
                    specify plotting against circular coordinates of all persistence barcodes or largest persistence barcode: 'All' or 'Max' 
            vert_list: ndarray
                list of points between the birth and death of the largest persistance barcode 
            fig_size : tuple
                    size of plotted figure(default=(10,10)) (only used if All and Max are False)
            ax: AxesSubplot
                Axes that should be used for plotting (only used if All and Max are False)
                pt_style: dict
                argments passed to `ax.scatter` for style of points.
            scrollable : bool
                    used to specify whether the plot should be inside a scrollable window since matplotlib does not support this feature locally
            

        """

        pt_kwargs = {}
        if pt_style is not None:
            pt_kwargs.update(pt_style)

        if type=='All':
            fig=plt.figure(figsize=(10,len(vert)))
            xx=math.ceil(len(vert)/3)
            for ind,eps in enumerate(rips["dgms"][1]):
                plt.subplot(xx, 3,ind+1 )
                plt.scatter(p1,vert[ind], c=vert[ind],cmap=plt.cm.hsv ,**pt_kwargs)
                plt.title('eps='+str(eps[1])[:4] +', dist='+ str(dist[ind])[:4])
            plt.tight_layout()
            if scrollable:
                a = ScrollableWindow(fig)
            else:
                plt.show()
            
        elif type=='Max':
            fig=plt.figure(figsize=(10,len(vert)))
            xx=math.ceil(len(vert)/3)
            for ind,eps in enumerate(vert_list):
                plt.subplot(xx, 3,ind+1 )
                plt.scatter(p1,vert[ind], c=vert[ind],cmap=plt.cm.hsv, **pt_kwargs)
                plt.title('eps='+str(eps)[:4] +', dist='+ str(dist[ind])[:4])
            plt.tight_layout()
            if scrollable:
                a = ScrollableWindow(fig)
                

            else:
                plt.show()
          
        else:
            if ax is None:
                ax = plt.axes()
            ax.scatter(p1, vert, **pt_kwargs)
            ax.set_xlabel('Data')
            ax.set_ylabel('\u03B8', size=18, rotation=0)
            # else:
                
            #     ax.set_xlabel(label)
           
            plt.show()

def plot_eps_3d(p1,vert,rips=None,type='All',vert_list=None,fig_size=(10,10)):

        """
        Function to plot external data with the circular coordinates as a 3d scatter plot
        ----------
        Parameters:
            p1 :(ndarray) 
                external data
            vert : ndarray
                    array of circulare coordinates mapped to (0,1)
            type : str
                    specify plotting against circular coordinates of all persistence barcodes or largest persistence barcode: 'All' or 'Max' 
            vert_list: ndarray
                list of points between the birth and death of the largest persistance barcode 
            fig_size : tuple
                    size of plotted figure(default=(10,10)) 
            
            

        """

        if type=='All':
            fig=plt.figure(figsize=(10,10))
            
            xx=math.ceil(len(vert)/3)
            ax = fig.gca(projection='3d')
            for ind,eps in enumerate(rips["dgms"][1]):
                ax.scatter(p1, eps[1],vert[ind], c=vert[ind],cmap=plt.cm.hsv)
            
        elif type=='Max':
            fig=plt.figure(figsize=(10,10))
        
            xx=math.ceil(len(vert)/3)
            ax = fig.gca(projection='3d')
            for ind,eps in enumerate(vert_list):
                ax.scatter(p1,  eps, vert[ind], c=vert[ind],cmap=plt.cm.hsv)

        
        plt.xlabel('Data')
        plt.ylabel('\u03B8', size=18, rotation=0)
        # else:
        #     plt.xlabel(label)
       
        plt.show()


# def cc(arg):
#     return mcolors.to_rgba(arg, alpha=0.6)
            
# def cols(leng):
#     """
#     Function to cycle red,green,blue and yellow colors for matplotlib
#     ----------
#     Parameters:
#         leng : int 
#             number of times to cycle
#     Returns:
#         lis: list
#             list of cycled colors
        
        

#     """
#     colr=['r','g','b','y']
#     x=0
#     lis=[]
#     while x<leng:
#         inds=x%4
#         lis.append(cc(colr[inds]))
#         x+=1
#     return lis
            

def plot_bars(dgm, order='birth', ax=None, bar_style=None):
    """
    Plot the barcode. 
    adapted from "https://github.com/mrzv/dionysus"

    Parameters:
        dgm: ndarray 
            persistance barcode diagram 

        order (str): How to sort the bars, either 'death' or 'birth'
                        (Default: 'birth')

        ax (AxesSubplot): Axes that should be used for plotting (Default: None)
        **bar_style: Arguments passed to `ax.plot` for style of the bars.
                        (Defaults: color='b')
    """


    bar_kwargs = {'color': 'b'}
    if bar_style is not None:
        bar_kwargs.update(bar_style)

    if order == 'death':
        generator = enumerate(sorted(dgm, key = lambda p: p[1]))
    else:
        generator = enumerate(sorted(dgm, key = lambda p: p[0]))

    if ax is None:
        ax = plt.axes()

    for i,p in generator:
        ax.plot([p[0], p[1]], [i,i], **bar_kwargs)

    plt.show()


def plot_diagram_density(dgm, lognorm=True, diagonal=True,
                         labels=False, ax=None, hist_style=None):
    """
    Plot the histogram of point density.
    adapted from "https://github.com/mrzv/dionysus"

   Parametes:
        dgm: ndarray 
            persistance barcode diagram 

        lognorm (bool): Use logarithmic norm (Default: True)
        diagonal (bool):  (Default: True)
        labels (bool): Set axis labels. (Default: False)
        ax (AxesSubplot): Axes that should be used for plotting (Default: None)
        **hist_style: Arguments passed to `ax.hist2d` for style of the histogram.
            (Defaults: bins=200)
    """


    hist_kwargs = {'bins': 200}
    if hist_style is not None:
        hist_kwargs.update(hist_style)

    if lognorm:
        norm = LogNorm()
    else:
        norm = Normalize()

    inf = float('inf')
    min_birth = min(p[0] for p in dgm if p[0] != inf)
    #max_birth = max(p.birth for p in dgm if p.birth != inf)
    #min_death = min(p.death for p in dgm if p.death != inf)
    max_death = max(p[1] for p in dgm if p[1] != inf)

    if ax is None:
        _, ax = plt.subplots()

    hist2d, histx, histy, im = ax.hist2d([p[0] for p in dgm if p[0] != inf and p[1] != inf], [p[1] for p in dgm if p[0] != inf and p[1] != inf], norm=norm, **hist_kwargs)
    ax.set_aspect('equal', 'datalim')
    if labels:
        ax.set_xlabel('birth')
        ax.set_ylabel('death')

    if diagonal:
        ax.plot([min_birth, max_death], [min_birth, max_death])

    ## clip the view
    #plt.axes().set_xlim([min_birth, max_birth])
    #plt.axes().set_ylim([min_death, max_death])

    if labels:
        plt.colorbar(im, ax=ax, label='overlap quantity')
    else:
        plt.colorbar(im, ax=ax)


    plt.show()



def plot_diagram(dgm, labels=False, ax=None,
                 line_style=None, pt_style=None):
    """
    Plot the persistence diagram.
    adapted from "https://github.com/mrzv/dionysus"
      Parameters:
        dgm: ndarray 
            persistance barcode diagram 
        labels (bool): Set axis labels. (Default: False)
        ax (AxesSubplot): Axes that should be used for plotting (Default: None)
        pt_style (dict): argments passed to `ax.scatter` for style of points.
        line_style (dict): argments passed to `ax.plot` for style of diagonal line.
    """

    import matplotlib.pyplot as plt

    line_kwargs = {}
    pt_kwargs = {}
    if pt_style is not None:
        pt_kwargs.update(pt_style)
    if line_style is not None:
        line_kwargs.update(line_style)


    inf = float('inf')
    min_birth = min((p[0] for p in dgm if p[0] != inf), default=0)
    max_birth = max((p[0] for p in dgm if p[1] != inf), default=1)
    min_death = min((p[1] for p in dgm if p[1] != inf), default=min_birth)
    max_death = max((p[1] for p in dgm if p[1] != inf), default=max_birth)

    if ax is None:
        ax = plt.axes()
    ax.set_aspect('equal', 'datalim')

    min_diag = min(min_birth, min_death)
    max_diag = max(max_birth, max_death)
    ax.scatter([p[0] for p in dgm], [p[1] for p in dgm], **pt_kwargs)
    ax.plot([min_diag, max_diag], [min_diag, max_diag], **line_kwargs)

    if labels:
        ax.set_xlabel('birth')
        ax.set_ylabel('death')

    ## clip the view
    #plt.axes().set_xlim([min_birth, max_birth])
    #plt.axes().set_ylim([min_death, max_death])


    plt.show()