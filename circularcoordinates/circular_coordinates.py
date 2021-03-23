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


class circular_coordinates():
    """
        This is the  main class for circular-coordinates library used to create and plot circular coordinates from persistent cohomology 
        ----------
        Parameters:
            prime : int
                    the prime basis over which to compute homology

    """

    def __init__(self,prime):
        self.prime=prime


    


    def get_epsilon(self,dgms):
        """
        Finds index of epsilon from the homology persistence diagram
        ----------

        Returns
        arg_eps : int
                the index of epsilon
        """
        arg_eps = max(enumerate(dgms), key = lambda pt: pt[1][1] - pt[1][0])[0]
        return arg_eps

    def boundary_cocycles(self,rips, epsilon,prime,spec=None):

        """
        Covenience function that combines 'delta' and 'cocycles'
        ----------
        Parameters:
            rips : dict
                The result from ripser(computing the vitoris rips complex) on the input data.
            prime : int
                 the prime basis over which to compute homology.
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

        distances = rips["dperm2all"]
        edges = np.array((distances<=epsilon).nonzero()).T

        # Construct 𝛿⁰
        Delta=self.delta(distances,edges)
        
        # Extract the cocycles
        dshape=Delta.shape[0]
        cocycles=self.cocycles(rips,distances,edges,dshape,prime,spec)
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
    
    def cocycles(self,rips,distances,edges,dshape,prime,spec=None):
        """
        extracts cocycles
        ----------
        Parameters:

        rips :dict
            The result from ripser on the input data.
        distances: ndarray
            Distances from points in the greedy permutation to points in the original point set
        edges: ndarray
            distances truncated to epsilon
        dshape: int
            number of rows in boundary (𝛿⁰)
        prime : int
                 the prime basis over which to compute homology.
        spec: int
            the index of the specific cocycle to extract. If spec is 'None' all cocycles will be retuned


        Returns
            cocycles : ndarray
            specific cocycle or all cocycles
        """



        cocycles = []
        if spec is not None:
           
            cocycle=rips["cocycles"][1][spec]
            val = cocycle[:,2]
            val[val > (prime-1)/2] -= prime
            Y = sparse.coo_matrix((val,(cocycle[:,0],cocycle[:,1])), shape=(distances.shape[0],distances.shape[0]))
            Y = Y - Y.T
            Z = np.zeros((dshape,))
            Z = Y[edges[:,0],edges[:,1]]
            return [Z]
        else:
            for cocycle in rips["cocycles"][1]:
                val = cocycle[:,2]
                val[val > (prime-1)/2] -= prime
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
        data_pca = pca.fit_transform(data1)
        return data_pca
  
        
    def circular_coordinates(self,rips,prime,vertex_values=None,arg_eps=None,check=None,intr=10):
        
        """
            computes and plots the circular_coordinates
        ----------
        Parameters:

           rips : dict
                The result from ripser(computing the vitoris rips complex) on the input data.
            prime : int
                 the prime basis over which to compute homology.
            vertex_values : ndarray
                circular coordinates of the longest barcode
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
                vertex_values : ndarray
                    circular coordinates
                OR
                max_/all_ : ndarray
                    list of circular coordinates of all persistance barcodes or binned from the longest barcode based on the 'check' input
                all_list/max_list: ndarray
                    list of epsilon for which the circular coordinates are computed 
        """
        checks=['All','Max',None]
        errs('check',check,checks)
        
        if check=='All':

            if vertex_values is None:
                vertex_values=self.circular_coordinates(rips,prime)

            all_,_=self.all_verts(rips,prime,vertex_values)
            return all_,[x[1] for x in rips['dgms'][1]]

        elif check=='Max':

            if vertex_values is None:
                vertex_values=self.circular_coordinates(rips,prime)

            max_,max_list,_=self.max_verts(rips,prime,vertex_values,intr=intr)

            return max_,max_list
        else:
            if arg_eps is None:
                arg_eps=self.get_epsilon(rips['dgms'][1])
            eps=rips["dgms"][1][arg_eps][1]
            delta,cocycles=self.boundary_cocycles(rips,eps,prime,arg_eps)
            vertex_values=self.minimize(delta,cocycles)
            return vertex_values
 
  
    
    def all_verts(self,rips,prime,init_verts,dist=None):

        """
        function to find circular coordinates for all persistance barcodes
        ----------
        Parameters:
        rips : dict
                The result from ripser(computing the vitoris rips complex) on the input data.
        prime : int
                 the prime basis over which to compute homology.
        init_verts: ndarray
               circular coordinates of the longest persistance barcode 
        dist : string
            distance calculation metric betweeen the circular coordinates: 'l1','l2', 'cosine' or 'None'


        Returns
        all_ : list
            circular coordinates for all persistance barcodes
        dist_ : list
            distances between the the circular coordinates
        """
        dists=['l1','l2' ,'cosine',None]
        errs('dist',dist,dists)
        all_=[]
        dist_=[]
        for indi,eps in enumerate(rips["dgms"][1]):
                eps=eps[1]
                delta,cocycles=self.boundary_cocycles(rips,eps,prime,indi)
                vertex_values=self.minimize(delta,cocycles)
                all_.append(vertex_values)
                
                if dist is not None:
                    dist_.append(self.get_dist(init_verts,vertex_values,dist))
            

               
        return all_,dist_

    
    def max_verts(self,rips,prime,init_verts,intr=10,dist=None):

        """
        function to find circular coordinates over the largest persistance barcode
        ----------

        Parameters:
        rips : dict
                The result from ripser(computing the vitoris rips complex) on the input data.
        prime : int
                 the prime basis over which to compute homology.
        init_verts: ndarray
               circular coordinates of the longest persistance barcode 
        intr : int
            specifies as to how many points to compute the circular coordinates of over the largest persistance barcode
        dist : string
            distance calculation metric betweeen the circular coordinates: 'l1','l2', 'cosine' or 'None'

        Returns
        max_ : list
            circular coordinates over the largest persistance barcode

        arr: ndarray
            list of points used between the birth and death of the largest barcode

        dist_ : list
            distances between the the circular coordinates
        

        """

        dists=['l1','l2' ,'cosine',None]
        errs('dist',dist,dists)
        max_=[]
        dist_=[]
        arg_eps=self.get_epsilon(rips['dgms'][1])
        eps_=rips["dgms"][1][arg_eps]
        arr= np.linspace(eps_[0],eps_[1],intr)
        for ep in arr:
            delta,cocycles=self.boundary_cocycles(rips,ep,prime,arg_eps)
            vertex_values=self.minimize(delta,cocycles)
            max_.append(vertex_values)
            
            if dist is not None:
                dist_.append(self.get_dist(init_verts,vertex_values,dist))
            
       
        return max_,arr,dist_

    def get_dist(self,init_verts,vert,dist='l1'):

        """
        convenience function to find distance between two sets of circular coordinates
        ----------

        Parameters:
        init_verts : ndarray
                the first array of circular coordinates
        vert : ndarray
                the second array of circular coordinates
        dist : string
            distance calculation metric betweeen the circular coordinates: 'l1','l2' or 'cosine'

        Returns
        d : float
            distance

        

        """
        if dist=='l1':
            d= np.linalg.norm(vert- init_verts, ord=1)
        elif dist=='l2':
            d= np.linalg.norm(vert- init_verts)
        elif dist=="cosine":
            d=spatial.distance.cosine(vert, init_verts)
        return d

    def get_dist_all(self,init_verts,vertex_values,dist='l1'):

        """
        convenience function to find distance between circular coordinates of the largest barcode and a list of max/all barcode circular coordinates
        ----------

        Parameters:
        init_verts : ndarray
                the first array of circular coordinates
        vertex_values : ndarray
                array of arrays of circular coordinates
        dist : string
            distance calculation metric betweeen the circular coordinates: 'l1','l2' or 'cosine'

        Returns
        dist_ : list
            list of distances

        

        """


        dist_=[]
        for vert in vertex_values:
            dist_.append(self.get_dist(init_verts,vert,dist))
        return dist_



    def plot_eps(self,p1,vertex_values,vert_list=None,dist_=None,type='2d',**kwargs):

        """
        Function to plot external data with the circular coordinates
        ----------

        Parameters:

        p1 : ndarray
            external data
        vertex_values : ndarray(2d or 3d)
                array of circular coordinates or array of arrays of circular coordinates depending on the type of plot
        vert_list : ndarray
                list of epsilons. Required if vertex_values is array of arrays
        dist : list
            list of distances to show along side plots
        
        type : string
            type of plot:
            2d: 2d scatter plot
            2d_multi:2d scatter plot over the largest persistance barcode or all persistance barcodes
            3d_multi:3d scatter plot over the largest persistance barcode or all persistance barcodes
    

        all other parameters please refer to plot_eps and plot_eps_3d in plotting.py

       
        """
        
        types=['2d','2d_multi','3d_multi']
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
            plot_eps(p1,vertex_values,**twodkwargs)

        elif type=='2d_multi':
            # vert,vert_list,dist_=self.max_verts(intr=intr,dist=dist)
            twodkwargs={'ax':None,"pt_style":None,"scrollable":False}
            if kwargs is not None:
                twodkwargs.update(kwargs)
            plot_eps(p1,vertex_values,type='multi',vert_list=vert_list,dist=dist_,**twodkwargs)
     
        elif type=='3d_multi':
            threedkwargs={"fig_size":(10,10)}
            if kwargs is not None:
                
                threedkwargs.update(kwargs)
            plot_eps_3d(p1,vertex_values,vert_list=vert_list,**threedkwargs)
     
  

    def plot_barcode(self,dgms=None,type='bar',**kwargs):

        """
        Function to plot persistance barcodes
        ----------

        Parameters:

        dgms:ndarray or list
            list of birth and death of persistance barcodes

        type:string
            type of plot:
            scatter:  Plot the persistence diagram as a scatter plot with line
            hist:Plot the histogram of point density
            bar:Plot the barcode as bars.

        all other parameters please refer to plot_diagram, plot_diagram_density and plot_bars in plotting.py


        """
        
        
       
        types=['scatter','hist', 'bar']
        errs('type',type,types)

        if dgms is None:
            dgms=self.rips['dgms'][1]

        if type=='scatter':
            scatkwargs={"labels":True,"line_style":None,"pt_style":None,'ax':None,"pt_style":None}
            if kwargs is not None:
                scatkwargs.update(kwargs)

            plot_diagram(dgms,**scatkwargs)

        elif type=='hist':
            histkwargs={"labels":True,"hist_style":None,"diagonal":True,'ax':None,"lognorm":True}
            if kwargs is not None:
                histkwargs.update(kwargs)


            plot_diagram_density(dgms,**histkwargs)
            
        elif type=='bar':
            barkwargs={"order":'birth',"bar_style":None,'ax':None}
            if kwargs is not None:
                barkwargs.update(kwargs)

            plot_bars(dgms,**barkwargs)

    	    
    def fit_transform(self,data):
        """
        Function to find circular coordinates from persistent cohomology of input data
        ----------

        Parameters:

         data :(ndarray or pandas dataframe) 
            input data
       


        """
        
        self.rips=ripsr(data,self.prime)
        vertex_values=self.circular_coordinates(self.rips,self.prime)
        return vertex_values

    def plot_pca(self, data,vertex_values,**kwargs):
            """
            Function to plot pca of data with circular cordinates represented as colors on the rgb color wheel
            ----------
            Parameters:

            data :(ndarray or pandas dataframe) 
                input data
            vertex_values: ndarray
                circular coordinates
            all other parameters please refer to plot_2dim in plotting.py
            """
            plot_kwargs={'xlabel':'Principal Component 1','ylabel':'Principal Component 2','fig_size':(10,10),'ax':None, 'pt_style':None}
            plot_kwargs.update(kwargs)
            data_pca=self.PCA_(data)
            plot_2dim(data_pca,vertex_values,**plot_kwargs)

        

    def plot_multi(sef,vertex_values,vertex_list,**kwargs):

        """
        Function to plot circular cordinates of multiple barcodes
        ----------
        Parameters:
         vertex_values: ndarray
            list of circular coordinates

         vertex_list: ndarray
            list of epsions 
        all other parameters please refer to plot_multi in plotting.py
        """

        p_kwargs={'fig_size':(10,10),'ax':None, 'pt_style':None}
        p_kwargs.update(kwargs)
        plot_multi(vertex_values,vertex_list,**p_kwargs)



    
