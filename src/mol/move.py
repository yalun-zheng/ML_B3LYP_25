import numpy as np
from .xyz2graph import MolGraph
import copy
import matplotlib.pyplot as plt

def draw(graph, show_stretch=False, priority=0):
    ele_sig = list('HCONFSPF') + ['Cl','Li', 'Si', 'Be', 'B', 'Na', 'Mg', 'Al']
    if show_stretch:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    for ele, (x, y, z), color in graph:
        if show_stretch:
            ax.scatter([x], [y], [z], marker='.', s=400, c=color)
    sig_record = {}
    for i, j in graph.edges():
        new_sig = ele_sig.index(graph[i][0])-0.01*len(graph.adj_list[i]) + ele_sig.index(graph[j][0])-0.01*len(graph.adj_list[j]) # order-0.01*num_neighbours
        # if sig < new_sig:
        sig_record[new_sig] = i,j
        if show_stretch:
            ax.plot([graph[i][1][0], graph[j][1][0]], [graph[i][1][1], graph[j][1][1]], [graph[i][1][2], graph[j][1][2]], c='k')
    # print(sig_record)
    if sig_record:
        sig_bond = sig_record[sorted(sig_record, reverse=True)[min([priority, len(sig_record)-1])]]
    else:
        # for BeH..., bond is too long, but shoud be improved in Graph later
        sig_bond = 0, 1

    if show_stretch:
        i, j = sig_bond
        ax.plot([graph[i][1][0], graph[j][1][0]], [graph[i][1][1], graph[j][1][1]], [graph[i][1][2], graph[j][1][2]], c='r')
        ax.set_xlabel('X / Bohr')
        ax.set_ylabel('Y / Bohr')
        ax.set_zlabel('Z / Bohr')
        plt.show()
        plt.close()
    return sig_bond

def rotationMatrix(alpha, v, unit='degree'):
    """
    return a rotation matrix based on an angle alpha and a normalized axis, vector v
    """

    assert unit in ['degree','radian'], "unit can only be degree/radian, degree as default."
    if unit=='degree':
        alpha = alpha/180*np.pi
    v = np.array(v) / np.linalg.norm(v)
    vx, vy, vz = v[0], v[1], v[2]
    return np.array([[(1-np.cos(alpha))*vx**2+np.cos(alpha),
    vx*vy*(1-np.cos(alpha))-vz*np.sin(alpha),vx*vz*(1-np.cos(alpha))+vy*np.sin(alpha)],
    [vx*vy*(1-np.cos(alpha))+vz*np.sin(alpha),(1-np.cos(alpha))*vy**2+np.cos(alpha),
    vz*vy*(1-np.cos(alpha))-vx*np.sin(alpha)],[vx*vz*(1-np.cos(alpha))-vy*np.sin(alpha),
    vy*vz*(1-np.cos(alpha))+vx*np.sin(alpha),(1-np.cos(alpha))*vz**2+np.cos(alpha)]])

def stretch(graph, i, j, deviation=0., toZ=True):
    # deviation in Angstrom
    graph = copy.deepcopy(graph)
    vec = (graph.coords[j] - graph.coords[i]) / np.linalg.norm(graph.coords[j] - graph.coords[i])
    if toZ:
        graph.coords -= ((graph.coords[i]+graph.coords[j])/2)
        cos_alpha = np.dot(vec,[0,0,1])
        alpha = np.arccos(min([1,max([cos_alpha,-1])]))
        norm = np.cross(vec,[0,0,1])
        if np.linalg.norm(norm)>1e-7:
            graph.coords = rotationMatrix(alpha,norm,'radian').dot(graph.coords.T).T
    vec = np.array([0,0,0.5])
    graph.coords[i] = graph.coords[i] - deviation*vec
    graph.coords[j] = graph.coords[j] + deviation*vec
    for neighbor in graph.adj_list[i]:
        if (neighbor != j) and len(graph.adj_list[neighbor])==1:
            graph.coords[neighbor] = graph.coords[neighbor] - deviation*vec
    for neighbor in graph.adj_list[j]:
        if (neighbor != i) and len(graph.adj_list[neighbor])==1:
            graph.coords[neighbor] = graph.coords[neighbor] + deviation*vec
    return graph

if __name__ == '__main__':
    import os
    xyz_folder = '../../data/xyz/'
    
    for fp in os.listdir(xyz_folder):
        print(fp)
        struct = MolGraph(os.path.join(xyz_folder, fp))
        i, j = draw(struct, show_stretch=True)
        print(i, j)
        
