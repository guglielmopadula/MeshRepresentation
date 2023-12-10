import numpy as np
import scipy.sparse
import itertools

def compute_edges_undirected(n_faces):
    h=n_faces.shape[1]
    comb=np.array(list(itertools.combinations(range(h),2)))
    comb_bak=comb.copy()
    comb[:,1]=comb_bak[:,0]
    comb[:,0]=comb_bak[:,1]
    comb=np.vstack((comb,comb_bak))
    faces_comb=n_faces[:,comb].reshape(-1,2)
    faces_comb=np.sort(faces_comb,axis=1)
    faces_comb=np.unique(faces_comb,axis=0)
    return faces_comb




def laplacian(edges):
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] = deg(i)       , if i == j
    L[i, j] = -1  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: array of shape (V, 3) containing the vertices of the graph
        edges: array of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = np.max(edges) + 1

    e0, e1 = edges[:,0].reshape(-1,1),edges[:,1].reshape(-1,1)

    idx01 = np.concatenate([e0, e1], axis=1)  # (E, 2)
    idx10 = np.concatenate([e1, e0], axis=1)  # (E, 2)
    idx = np.concatenate([idx01, idx10], axis=0).T  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    idx0=idx[0,:]
    idx1=idx[1,:]
    ones = np.ones(idx.shape[1], dtype=np.float32)

    A = scipy.sparse.coo_array((ones, (idx0,idx1)), shape=(V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = A.sum(axis=1)
    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = -1 if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = np.where(deg0 > 0.0, -1.0, deg0)
    deg1 = deg[e1]
    deg1 = np.where(deg1 > 0.0, -1.0, deg1)
    val = np.concatenate([deg0, deg1]).reshape(-1)
    L = scipy.sparse.coo_array((val, (idx0,idx1)), shape=(V, V))
    # Then we add the diagonal values L[i, i] = deg(i).
    idx = np.arange(V)
    L += scipy.sparse.coo_array((deg, (idx,idx)), shape=(V, V))
    return L


class MeshLaplacianEmbedder():
    def __init__(self,triangles,k=512):
        self.edges=compute_edges_undirected(triangles)
        self.L=laplacian(self.edges)
        self.k=k
        _,self.V=scipy.sparse.linalg.lobpcg(self.L,np.random.rand(self.L.shape[0],self.k),largest=False)

    def compute_reduction(self,quantity):
        v=self.V
        num_points=v.shape[0]
        num_red=v.shape[1]
        quantity_=quantity.reshape(quantity.shape[0],num_points,-1).copy()
        out_shape=quantity_.shape[2]
        quantity_=np.transpose(quantity_,(0,2,1))
        quantity_=quantity_@v
        return quantity_.reshape(-1,out_shape*num_red)

    def revert_reduction(self,quantity):
        v=self.V
        num_points=v.shape[0]
        num_red=v.shape[1]
        quantity_=quantity.reshape(quantity.shape[0],-1,num_red).copy()
        out_shape=quantity_.shape[1]
        quantity_=(quantity_@v.T)
        quantity_=np.transpose(quantity_,(0,2,1))
        quantity_=quantity_.reshape(quantity_.shape[0],num_points,out_shape)
        return quantity_

    


if __name__=="__main__":
    import meshio
    from meshlaplacianembedder import MeshLaplacianEmbedder
    triangles=meshio.read("Stanford_Bunny.stl").cells_dict["triangle"]
    mle=MeshLaplacianEmbedder(triangles,k=1024)
    points=meshio.read("Stanford_Bunny.stl").points
    points=points.reshape(1,-1,3)
    print("Coarse rec is",np.linalg.norm(points-mle.revert_reduction(mle.compute_reduction(points)))/np.linalg.norm(points))
