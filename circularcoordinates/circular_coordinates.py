from scipy import sparse
from sklearn import decomposition, preprocessing
from ripser import ripser
import numpy as np
from scipy import spatial
from  .plotting import *


def ripsr(data,prime):
    """
    convenience function to compute Vietoris–Rips persistence barcodes using the ripser library
    ----------

    Parameters:

    data : ndarray or pandas dataframe
            input data
        prime: int
            prime basis to compute homology

    Returns
        rips : dict
            result of ripser
    
    """
    if isinstance(data, np.ndarray) :
            data1=data
    else:
            data1=data.to_numpy()
    rips = ripser(data1, coeff=prime, do_cocycles=True)
    return rips

def errs(str_,err_,errs_):
        if err_ not in errs_:
            raise ValueError(str_+ ' can only be '+ str(errs_)[1:-1])


class circular_coordinate():

    """
    This is the  main class for circular-coordinates library used to create and plot circular coordinates from persistent cohomology 
    ----------
    Parameters:
        data :(ndarray or pandas dataframe) 
            input data
        prime : int
                the prime basis over which to compute homology

    """

    def __init__(self,data,prime):
        self.data=data
        self.prime=prime
        self.rips=ripsr(data,prime)
        self.vertex_values=None
        

    def get_epsilon(self):
        """
        Finds epsilon from the homology persistence diagram
        ----------

        Returns
        arg_eps : int
                the index of epsilon
        """
        arg_eps = max(enumerate(self.rips["dgms"][1]), key = lambda pt: pt[1][1] - pt[1][0])[0]
        self.eps=self.rips["dgms"][1][arg_eps][1]
        return arg_eps

    def boundary_cocycles(self,rout, epsilon,spec=None):

        """
        Covenience function that combines 'delta' and 'cocycles'
        ----------
        Parameters:
            rout : dict
                The result from ripser(computing the vitoris rips complex) on the input data.
        epsilon: float
            epsilon used to truncate the vitoris rips
        spec: int
            the index of the specific cocycle to extract. If spec is 'None' all cocycles will be retuned

        Returns
            Delta : ndarray
                 boundary (𝛿⁰)
            cocycles : ndarray
                specific cocycle or all cocycles

        """

        distances = rout["dperm2all"]
        edges = np.array((distances<=epsilon).nonzero()).T

        # Construct 𝛿⁰
        Delta=self.delta(distances,edges)
        
        # Extract the cocycles
        dshape=Delta.shape[0]
        cocycles=self.cocycles(rout,distances,edges,dshape,spec)
        return Delta,cocycles
    
    def delta(self,distances,edges):

        """
        Constructs the boundary (𝛿⁰)
        ----------
        Parameters:

        distances: ndarray
            Distances from points in the ripser greedy permutation to points in the original data point set
        edges: ndarray
            distances truncated to epsilon

        Returns
        Delta : ndarray
                boundary (𝛿⁰)
        """
        I = np.c_[np.arange(edges.shape[0]),np.arange(edges.shape[0])]
        I = I.flatten()
        J = edges.flatten()
        V = np.c_[-1 * np.ones(edges.shape[0]), np.ones(edges.shape[0])]
        V = V.flatten()
        Delta = sparse.coo_matrix((V,(I,J)), shape=(edges.shape[0], distances.shape[0]))
        return Delta
    
    def cocycles(self,rout,distances,edges,dshape,spec=None):
        """
        extracts cocycles
        ----------
        Parameters:

        rout :dict
            The result from ripser on the input data.
        distances: ndarray
            Distances from points in the greedy permutation to points in the original point set
        edges: ndarray
            distances truncated to epsilon
        dshape: int
            number of rows in boundary (𝛿⁰)
        spec: int
            the index of the specific cocycle to extract. If spec is 'None' all cocycles will be retuned


        Returns
            cocycles : ndarray
            specific cocycle or all cocycles
        """



        cocycles = []
        if spec is not None:
           
            cocycle=rout["cocycles"][1][spec]
            val = cocycle[:,2]
            val[val > (self.prime-1)/2] -= self.prime
            Y = sparse.coo_matrix((val,(cocycle[:,0],cocycle[:,1])), shape=(distances.shape[0],distances.shape[0]))
            Y = Y - Y.T
            Z = np.zeros((dshape,))
            Z = Y[edges[:,0],edges[:,1]]
            return [Z]
        else:
            for cocycle in rout["cocycles"][1]:
                val = cocycle[:,2]
                val[val > (self.prime-1)/2] -= self.prime
                Y = sparse.coo_matrix((val,(cocycle[:,0],cocycle[:,1])), shape=(distances.shape[0],distances.shape[0]))
                Y = Y - Y.T
                Z = np.zeros((dshape,))
                Z = Y[edges[:,0],edges[:,1]]
                cocycles.append(Z)
                return cocycles




    
    def minimize(self,Delta,cocycle):

        """
        minimizes ∥ζ-𝛿1α∥2 and computes vertex values map to [0,1]
        ----------
        Parameters:

        Delta : ndarray
                boundary (𝛿⁰)
        cocycle: ndarray
            corresponding cocycle (ζ)

        Returns
        vertex_values : ndarray
                circular coordinates mapped to [0,1] interval
        """

        mini = sparse.linalg.lsqr(Delta, np.array(cocycle).squeeze())
        vertex_values=np.mod(np.array(mini[0]), 1.0)
        return vertex_values
    

    def PCA_(self,data,n=2):

        """
        convenience function to perform PCA on input data
        ----------
        Parameters:

        n : int
                number of dimensions pca should reduce the data to.

        """

        if isinstance(data, np.ndarray):
            data1=data
        else:
            data1=data.to_numpy()
        pca = decomposition.PCA(n_components=n)
        self.data_pca = pca.fit_transform(data1)
        # return self.data_pca
  
        
    def circular_coordinate(self,plot=False,arg_eps=None,check=None,intr=10):
        
        """
            computes and plots the circular_coordinates
        ----------
        Parameters:

            plot : boolean
                Flag to indicate whether to plot the coordinates
            arg_eps: int
                index of epsilon
            check: string
                can be 'All' or 'Max' or None:
                    All: compute circular coordinates for all persistance barcodes
                    Max: compute cicular coordinates over the largest persistance barcode
                    None:  compute cicular coordinates for the largest persistance barcode or epsilon
            intr: int
                Only required if check is 'Max', specifies as to how many points to compute coordinates over the max persistance barcode
            
        Returns
                circular_coordinates : ndarray
                circular coordinates
        """
        checks=['All','Max',None]
        errs('check',check,checks)
        
        if check=='All':

            if self.vertex_values is None:
                self.circular_coordinate()

            all_,_=self.all_verts()
            if plot:
                plot_check_all(self.rips,all_)
            return all_

        elif check=='Max':

            if self.vertex_values is None:
                self.circular_coordinate()

            max_,max_list,_=self.max_verts(intr=intr)
            if plot:
                plot_check_max(max_,max_list)
            return max
        else:
            if arg_eps is None:
                arg_eps=self.get_epsilon()
            # eps=self.rips["dgms"][1][arg_eps][1]
            delta,cocycles=self.boundary_cocycles(self.rips,self.eps,arg_eps)
            self.vertex_values=self.minimize(delta,cocycles)
            if plot:
                self.PCA_(self.data)
                plot_PCA(self.data_pca,self.vertex_values)
            return self.vertex_values
            # self.plot_diagram_density(self.rips["dgms"][1])
            # self.plot_bars(self.rips["dgms"][1])
        
  
    
    def all_verts(self,dist='l1'):

        """
        function to find circular coordinates for all persistance barcodes
        ----------
        Parameters:

        dist : string
            distance calculation metric betweeen the circular coordinates: 'l1','l2' or 'cosine'


        Returns
        all_ : list
            circular coordinates for all persistance barcodes
        dist_ : list
            distances between the the circular coordinates
        """
        dists=['l1','l2' ,'cosine']
        errs('dist',dist,dists)
        all_=[]
        dist_=[]
        for indi,eps in enumerate(self.rips["dgms"][1]):
                eps=eps[1]
                delta,cocycles=self.boundary_cocycles(self.rips,eps,indi)
                vertex_values=self.minimize(delta,cocycles)
                all_.append(vertex_values)
                
                if dist=='l1':
                    dist_.append( np.linalg.norm(vertex_values- self.vertex_values, ord=1))
                elif dist=='l2':
                    dist_.append( np.linalg.norm(vertex_values- self.vertex_values))
                elif dist=="cosine":
                    dist_.append(spatial.distance.cosine(vertex_values, self.vertex_values))
            

               
        return all_,dist_

    
    def max_verts(self, intr=10,dist='l1'):

        """
        function to find circular coordinates over the largest persistance barcode
        ----------

        Parameters:

        intr : int
            specifies as to how many points to compute the circular coordinates of over the largest persistance barcode
        dist : string
            distance calculation metric betweeen the circular coordinates: 'l1','l2' or 'cosine'

        Returns
        max_ : list
            circular coordinates over the largest persistance barcode

        arr: ndarray
            list of points used between the birth and death of the largest barcode

        dist_ : list
            distances between the the circular coordinates
        

        """

        dists=['l1','l2' ,'cosine']
        errs('dist',dist,dists)
        max_=[]
        dist_=[]
        arg_eps=self.get_epsilon()
        eps_=self.rips["dgms"][1][arg_eps]
        arr= np.linspace(eps_[0],eps_[1],intr)
        for ep in arr:
            delta,cocycles=self.boundary_cocycles(self.rips,self.eps,arg_eps)
            vertex_values=self.minimize(delta,cocycles)
            max_.append(vertex_values)
            if dist=='l1':
                dist_.append( np.linalg.norm(vertex_values- self.vertex_values, ord=1))
            elif dist=='l2':
                dist_.append( np.linalg.norm(vertex_values- self.vertex_values))
            elif dist=="cosine":
                dist_.append(spatial.distance.cosine(vertex_values, self.vertex_values))
            
       
        return max_,arr,dist_

    def plot_eps(self,p1,type='2d',**kwargs):

        """
        Function to plot external data with the circular coordinates
        ----------

        Parameters:

        p1 : ndarray
            external data
        type : string
            type of plot:
            2d: 2d scatter plot
            2d_max:2d scatter plot over the largest persistance barcode
            2d_all:2d scatter plot with all persistance barcodes
            3d_all:3d scatter plot with all persistance barcodes
            3d_max:3d scatter plot over the largest persistance barcode
            # poly_all(not recommended):3d polygon plot with all persistance barcodes
            # poly_max:3d polygon plot over the largest persistance barcode
        all other parameters please refer to plot_eps and plot_eps_3d in py

       
        """
        
        types=['2d','2d_max', '2d_all','3d_max', '3d_all']
        errs('type',type,types)
        
        if 'intr' not in kwargs:
                intr=10
        else:
                intr=kwargs['intr']  
                kwargs.pop('intr', None)
        if 'dist' not in kwargs:
                dist='l1'
        else:
                dist=kwargs['dist']  
                kwargs.pop('dist', None)
                   

        if type=='2d':
            twodkwargs={"fig_size":(10,10),'ax':None,"pt_style":None}
            if kwargs is not None:
                twodkwargs.update(kwargs)
            plot_eps(p1,self.vertex_values,**twodkwargs)

        elif type=='2d_max':
            vert,vert_list,dist_=self.max_verts(intr=intr,dist=dist)
            twodkwargs={'ax':None,"pt_style":None,"scrollable":False}
            if kwargs is not None:
                twodkwargs.update(kwargs)
            plot_eps(p1,vert,type='Max',vert_list=vert_list,dist=dist_,**twodkwargs)
        elif type=='2d_all':
            vert,dist_=self.all_verts(dist=dist)
            twodkwargs={'ax':None,"pt_style":None,"scrollable":False}
            if kwargs is not None:
                twodkwargs.update(kwargs)
            plot_eps(p1,vert,rips=self.rips,type='All',dist=dist_,**twodkwargs)
        elif type=='3d_max':
            threedkwargs={"fig_size":(10,10)}
            if kwargs is not None:
                
                threedkwargs.update(kwargs)
            vert,vert_list,_=self.max_verts(intr=intr)
            plot_eps_3d(p1,vert,type='Max',vert_list=vert_list,**threedkwargs)
        elif type=='3d_all':
            threedkwargs={"fig_size":(10,10)}
            if kwargs is not None:
                
                threedkwargs.update(kwargs)
            vert,_=self.all_verts()
            plot_eps_3d(p1,vert,rips=self.rips,type='All',**threedkwargs)
  

    def plot_barcode(self,type='bar',**kwargs):

        """
        Function to plot persistance barcodes
        ----------

        Parameters:
        type:string
            type of plot:
            scatter:  Plot the persistence diagram as a scatter plot with line
            hist:Plot the histogram of point density
            bar:Plot the barcode as bars.

        all other parameters please refer to plot_diagram, plot_diagram_density and plot_bars in py


        """
        
        
       
        types=['scatter','hist', 'bar']
        errs('type',type,types)
        if type=='scatter':
            scatkwargs={"labels":True,"line_style":None,"pt_style":None,'ax':None,"pt_style":None}
            if kwargs is not None:
                scatkwargs.update(kwargs)

            plot_diagram(self.rips['dgms'][1],**scatkwargs)

        elif type=='hist':
            histkwargs={"labels":True,"hist_style":None,"diagonal":True,'ax':None,"lognorm":True}
            if kwargs is not None:
                histkwargs.update(kwargs)


            plot_diagram_density(self.rips['dgms'][1],**histkwargs)
            
        elif type=='bar':
            barkwargs={"order":'birth',"bar_style":None,'ax':None}
            if kwargs is not None:
                barkwargs.update(kwargs)

            plot_bars(self.rips['dgms'][1],**barkwargs)




    



  

