import meshio
from tqdm import trange
import numba
mesh=meshio.read("Stanford_Bunny.stl")
import numpy as np
faces=mesh.cells_dict["triangle"]
points=mesh.points.reshape(-1)
points_old=points.copy().reshape(-1)
points=points.reshape(-1)
points=points+np.random.rand(*points.shape)*0.01*np.linalg.norm(points)/len(points)
points=points.reshape(-1,3)
points_old=points_old.reshape(-1,3)
bound=0.01*np.linalg.norm(points)/len(points)
edges=set([])
from scipy.sparse import csr_array
for face in trange(len(faces)):
    for i in faces[face]:
        for j in faces[face]:
            if i<j:
                edges.add(tuple([i,j]))

edges=[list(x) for x in edges]

edges=np.array(edges,dtype=np.int32)
edge_values=np.zeros((edges.shape[0]))

for i in range(len(edges)):
    edge_values[i]=np.linalg.norm(points[edges[i,0]]-points[edges[i,1]])

points=points.reshape(-1)
points_old=points_old.reshape(-1,3)

def loss(x,edges,edge_values):
    loss=0
    x=x.reshape((-1,3))
    for i in range(len(edges)):
        loss=loss+(np.linalg.norm(x[edges[i,0]]-x[edges[i,1]])-edge_values[i])**2
    return loss





'''
def gradient(x,edges,edge_values):
    data=np.zeros(6*len(edges))
    indices_x=np.zeros(6*len(edges),dtype=int)
    indices_y=np.zeros(6*len(edges),dtype=int)
    for k in range(len(edges)):
        edge=edges[k]
        i=edge[0]
        j=edge[1]
        indices_x[k:k+6]=k
        indices_y[k:k+3]=3*i+np.arange(3)
        indices_y[k+3:k+6]=3*j+np.arange(3)
        data[k]=4*(x[3*i]-x[3*j])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        data[k+1]=4*(x[3*i+1]-x[3*j+1])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        data[k+2]=4*(x[3*i+2]-x[3*j+2])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        data[k+3]=-4*(x[3*i]-x[3*j])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        data[k+4]=-4*(x[3*i+1]-x[3*j+1])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        data[k+5]=-4*(x[3*i+2]-x[3*j+2])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
    A=csr_array((data, (indices_x, indices_y)), [len(edges), 3*(np.max(edges)+1)])
    return A
'''

def gradient(x,edges,edge_values):
    gradient=np.zeros(x.shape[0])
    for k in range(len(edges)):
        edge=edges[k]
        i=edge[0]
        j=edge[1]
        gradient[3*i]=gradient[3*i]+4*(x[3*i]-x[3*j])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        gradient[3*i+1]=gradient[3*i+1]+4*(x[3*i+1]-x[3*j+1])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        gradient[3*i+2]=gradient[3*i+2]+4*(x[3*i+2]-x[3*j+2])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        gradient[3*j]=gradient[3*j]-4*(x[3*i]-x[3*j])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        gradient[3*j+1]=gradient[3*j+1]-4*(x[3*i+1]-x[3*j+1])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
        gradient[3*j+2]=gradient[3*j+2]-4*(x[3*i+2]-x[3*j+2])*((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2+(x[3*i+1]-x[3*j+1])**2-edge_values[k]**2)
    return gradient

def gradient_vectorized(x, edges, edge_values):
    gradient = np.zeros(x.shape[0])
    
    for k, edge in enumerate(edges):
        i, j = edge
        diff = x[3*i:3*i+3] - x[3*j:3*j+3]
        dist_squared = np.sum(diff ** 2)
        edge_diff = dist_squared - edge_values[k] ** 2
        gradient[3*i:3*i+3] += 4 * edge_diff * diff
        gradient[3*j:3*j+3] -= 4 * edge_diff * diff
    
    return gradient

x0=points_old.reshape(-1)
from scipy.optimize import minimize,Bounds
#x=x0
#print(loss(x,edges,edge_values))
#for i in range(1000):
#    x=x-0.00001*gradient_vectorized(x,edges,edge_values)
#    print(loss(x,edges,edge_values))

x=minimize(loss,x0,args=(edges,edge_values),jac=gradient_vectorized,method="L-BFGS-B",bounds=Bounds(lb=x0-bound,ub=x0+bound))
print(loss(x.x,edges,edge_values))
print(np.linalg.norm(points.reshape(-1)-x.x)/np.linalg.norm(points))
