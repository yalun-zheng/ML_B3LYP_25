import re
import numpy as np

atomic_radii = dict(
    Ac=1.88,
    Ag=1.59,
    Al=1.35,
    Am=1.51,
    As=1.21,
    Au=1.50,
    B=0.83,
    Ba=1.34,
    Be=0.35,
    Bi=1.54,
    Br=1.21,
    C=0.68,
    Ca=0.99,
    Cd=1.69,
    Ce=1.83,
    Cl=0.99,
    Co=1.33,
    Cr=1.35,
    Cs=1.67,
    Cu=1.52,
    D=0.23,
    Dy=1.75,
    Er=1.73,
    Eu=1.99,
    F=0.64,
    Fe=1.34,
    Ga=1.22,
    Gd=1.79,
    Ge=1.17,
    H=0.32,
    He=0.31,
    Ne=0.38,
    Ar=0.98,
    Hf=1.57,
    Hg=1.70,
    Ho=1.74,
    I=1.40,
    In=1.63,
    Ir=1.32,
    K=1.33,
    La=1.87,
    Li=0.68,
    Lu=1.72,
    Mg=1.10,
    Mn=1.35,
    Mo=1.47,
    N=0.68,
    Na=0.97,
    Nb=1.48,
    Nd=1.81,
    Ni=1.50,
    Np=1.55,
    O=0.68,
    Os=1.37,
    P=1.05,
    Pa=1.61,
    Pb=1.54,
    Pd=1.50,
    Pm=1.80,
    Po=1.68,
    Pr=1.82,
    Pt=1.50,
    Pu=1.53,
    Ra=1.90,
    Rb=1.47,
    Re=1.35,
    Rh=1.45,
    Ru=1.40,
    S=1.02,
    Sb=1.46,
    Sc=1.44,
    Se=1.22,
    Si=1.20,
    Sm=1.80,
    Sn=1.46,
    Sr=1.12,
    Ta=1.43,
    Tb=1.76,
    Tc=1.35,
    Te=1.47,
    Th=1.79,
    Ti=1.47,
    Tl=1.55,
    Tm=1.72,
    U=1.58,
    V=1.33,
    W=1.37,
    Y=1.78,
    Yb=1.94,
    Zn=1.45,
    Zr=1.56,
)

colors = dict(
    Ar="cyan",
    B="salmon",
    Ba="darkgreen",
    Be="darkgreen",
    Br="darkred",
    C="black",
    Ca="darkgreen",
    Cl="green",
    Cs="violet",
    F="green",
    Fe="darkorange",
    Fr="violet",
    H="white",
    He="cyan",
    I="darkviolet",
    K="violet",
    Kr="cyan",
    Li="violet",
    Mg="darkgreen",
    N="blue",
    Na="violet",
    Ne="cyan",
    O="red",
    P="orange",
    Ra="darkgreen",
    Rb="violet",
    S="yellow",
    Sr="darkgreen",
    Ti="gray",
    Xe="cyan",
)

class MolGraph:
    def __init__(self, xyz_file, to_center=True):
        self.spin = 0
        self.charge = 0
        self.elements, self.coords = self.xyz_to_dict(xyz_file)
        self.elements = [e[0].upper()+e[1:].lower() for e in self.elements]
        if to_center:
            self.coords -= self.coords.mean(axis=0)
        self.adj_list = {}
        self.atomic_radii = np.array([atomic_radii[element] for element in self.elements])
        self.colors = [colors.get(element,'violet') for element in self.elements]
        self.bond_lengths = {}
        self._generate_adjacency_list()

    def xyz_to_dict(self, xyz_fp):
        with open(xyz_fp, 'r') as xyz_f:
            lines = xyz_f.readlines()
            xyz_dict = {'atom_list':[], 'xyz':[]}
        for line in lines[2:]:
            atom_name, x, y, z = line.split()
            x, y, z = float(x), float(y), float(z)
            xyz_dict['atom_list'] += [atom_name]
            xyz_dict['xyz'] += [[x, y, z]]
        charge, spin = lines[1].split()
        self.charge = int(charge)
        self.spin = int(spin)
        return xyz_dict['atom_list'], np.array(xyz_dict['xyz'])
    
    def _generate_adjacency_list(self):
        distances = self.coords[:,None,:]-self.coords
        distances = np.sqrt(np.einsum("mni,mni->mn", distances, distances))
        distance_bond = (self.atomic_radii[:, None] + self.atomic_radii) * 1.3
        adj_matrix = np.logical_and(0.1 < distances, distance_bond > distances).astype(
            int
        )
        for i, j in zip(*np.nonzero(np.triu(adj_matrix))):
            self.adj_list.setdefault(i, set()).add(j)
            self.adj_list.setdefault(j, set()).add(i)
            self.bond_lengths[frozenset([i, j])] = round(distance_bond[i, j], 5)

        self.adj_matrix = adj_matrix

    def edges(self):
        edges = set()
        for node, neighbours in self.adj_list.items():
            for neighbour in neighbours:
                edge = frozenset([node, neighbour])
                if edge in edges:
                    continue
                edges.add(edge)
                yield node, neighbour

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, position):
        return self.elements[position], self.coords[position], self.colors[position]
