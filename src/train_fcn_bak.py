import pickle as pk
import numpy as np
from build import Tree
from pyscf import gto, dft
from einsum import dm2rho01, dm2eT2, dm2eJ, dm2eK, dm2rho01_sep, dm2eT
from opt_einsum import contract
import os
import time

def gen_cube(center, mol, dm, mesh, ni, *args):
    if len(dm)==2:
        dm = dm[0] + dm[1]
    phi012 = ni.eval_ao(mol, np.array([center]), deriv=2)
    grad = dm2rho01(dm, phi012[:4])[1:, 0]
    phi2 = np.zeros((3,3)+phi012[0].shape)
    phi2[0,:] = phi012[4:7]
    phi2[1,0] = phi012[5]
    phi2[1,1] = phi012[7]
    phi2[1,2] = phi012[8]
    phi2[2,0] = phi012[6]
    phi2[2,1] = phi012[8]
    phi2[2,2] = phi012[9]
    I = 2*(contract('ij,pri,qrj->rpq',dm, phi012[1:4], phi012[1:4])+
            contract('ij,ri,pqrj->rpq',dm, phi012[0], phi2))[0]
    sort_index = np.argsort(-np.abs(grad))
    e, v = np.linalg.eigh(I)
    dot = np.dot(v[:,sort_index[0]], grad)
    if dot < 0:
        v[:,sort_index[0]] = -v[:,sort_index[0]]
    dot = np.dot(v[:,sort_index[1]], grad)
    if dot < 0:
        v[:,sort_index[1]] = -v[:,sort_index[1]]
    dot = np.dot(v[:,sort_index[2]], grad)
    if dot < 0:
        v[:,sort_index[2]] = -v[:,sort_index[2]]
    cube = center + v.dot(mesh.T).T
    phi01 = ni.eval_ao(mol, cube, deriv=1)
    assert len(dm) != 2 or np.allclose(dm[1], 0)
    rho01 = dm2rho01(dm, phi01)
    rho = rho01[0]
    gnorm = contract('dr,dr->r', rho01[1:], rho01[1:])**0.5
    tau = 0.5*contract('ij, dri, drj->r', dm, phi01[1:], phi01[1:])
    return np.hstack((rho.reshape(-1, 1), gnorm.reshape(-1, 1), tau.reshape(-1, 1)))

def gen_cube_spin(center, mol, dm, mesh, ni, *args):
    if len(dm) != 2:
        dm = 0.5*dm, 0.5*dm
    assert len(dm) == 2
    phi012 = ni.eval_ao(mol, np.array([center]), deriv=2)
    grad = dm2rho01(dm[0]+dm[1], phi012[:4])[1:, 0]
    phi2 = np.zeros((3,3)+phi012[0].shape)
    phi2[0,:] = phi012[4:7]
    phi2[1,0] = phi012[5]
    phi2[1,1] = phi012[7]
    phi2[1,2] = phi012[8]
    phi2[2,0] = phi012[6]
    phi2[2,1] = phi012[8]
    phi2[2,2] = phi012[9]
    I = 2*(contract('ij,pri,qrj->rpq',dm[0]+dm[1], phi012[1:4], phi012[1:4])+
            contract('ij,ri,pqrj->rpq',dm[0]+dm[1], phi012[0], phi2))[0]
    sort_index = np.argsort(-np.abs(grad))
    e, v = np.linalg.eigh(I)
    dot = np.dot(v[:,sort_index[0]], grad)
    if dot < 0:
        v[:,sort_index[0]] = -v[:,sort_index[0]]
    dot = np.dot(v[:,sort_index[1]], grad)
    if dot < 0:
        v[:,sort_index[1]] = -v[:,sort_index[1]]
    dot = np.dot(v[:,sort_index[2]], grad)
    if dot < 0:
        v[:,sort_index[2]] = -v[:,sort_index[2]]
    cube = center + v.dot(mesh.T).T
    phi01 = ni.eval_ao(mol, cube, deriv=1)
    
    rho01 = dm2rho01(dm[0], phi01), dm2rho01(dm[1], phi01)
    rho = np.vstack((rho01[0][0]+rho01[1][0], (rho01[0][0]-rho01[1][0])/(rho01[0][0]+rho01[1][0]+1e-7)))
    gnorm = contract('dr,dr->r', rho01[0][1:], rho01[0][1:])**.5, contract('dr,dr->r', rho01[1][1:], rho01[1][1:])**.5, contract('dr,dr->r', rho01[0][1:]+rho01[1][1:], rho01[0][1:]+rho01[1][1:])**.5
    gnorm = np.vstack(gnorm)/((rho01[0][0]+rho01[1][0]+1e-7)**(4/3))/(2*(3*np.pi)**(1/3))
    gnorm = gnorm[2:]
    tau = 0.5*contract('ij, dri, drj->r', dm[0], phi01[1:], phi01[1:]), 0.5*contract('ij, dri, drj->r', dm[1], phi01[1:], phi01[1:])
    # tau = np.vstack(tau+(tau[0]+tau[1],))
    t_w = 1/8*contract('dr,dr->r', rho01[0][1:]+rho01[1][1:], rho01[0][1:]+rho01[1][1:])/(rho01[0][0]+rho01[1][0]+1e-7)
    t_unif = 3/10*(3*np.pi**2)**(2/3)*(rho01[0][0]+rho01[1][0]+1e-7)**(5/3)
    tau = np.vstack( (t_w/(tau[0]+tau[1]+1e-7), (tau[0]+tau[1]-t_w)/(t_unif+1e-7), (tau[0]+tau[1])/(t_unif+1e-7)) )
    
    #rho_up, rho_down, rho, g_up, g_down, g, tau_up, tau_down, tau
    return np.concatenate((rho, gnorm, tau), axis=0).T


def get_mesh(a, n_samples=6):
    pos = np.linspace(-a/2+a/(n_samples-1)*.5, a/2-a/(n_samples-1)*.5, n_samples-1)
    mesh = np.meshgrid(pos,pos,pos, indexing='ij')
    mesh = np.stack(mesh, axis=-1).reshape(-1,3)
    in_sphere_id = contract('ri,ri->r', mesh, mesh)<=(.5*a)**2
    mesh = mesh[in_sphere_id]
    return mesh

def get_feature(data, mol, ni, a, n_samples=6, dm='b3'):
    mesh = get_mesh(a, n_samples)
    res = []
    for center in data['gc']:
        cube = gen_cube_spin(center, mol, data['dm_'+dm], mesh, ni)
        res.append(cube)
    return np.array(res)

def cal_mu(gc, atoms_charges, atoms_coords):
    res = 0
    for z, c in zip(atoms_charges, atoms_coords):
        dc = np.linalg.norm(gc-c, axis=-1)
        res -= z/dc
    return res

def stretch_mol(fn, priority, basis, grid_level, test=False):
    print(f'no pkl file {fn}, generating it...')
    from mol.move import MolGraph, draw, stretch
    from gen_eden import gen_pkl, read_xyz_to_dict, gen_test_pkl
    sta_time = time.time()
    xyz_path = os.path.join(os.path.dirname(os.path.dirname(fn)), 'xyz/'+os.path.basename(fn).split('_')[0]+'.xyz')
    assert os.path.exists(xyz_path), f'error! pls put {os.path.basename(xyz_path)} into {os.path.dirname(xyz_path)}'
    dist = float(os.path.basename(fn).split('_')[1])
    struct = MolGraph(xyz_path)
    if abs(dist) < 1e-6:
        dist = 0
    if len(struct)>1:
        i, j = draw(struct, priority=priority)
        new_struct = stretch(struct, i, j, dist)
    else:
        new_struct = struct
    tmp = fn.replace('.pkl', '.xyz').replace('_train_valid', '').replace('data/pkl', 'data/tmp')
    tmp = fn.replace('.pkl', '.xyz').replace('_test', '').replace('data/pkl', 'data/tmp')
    with open(tmp, 'w') as f:
        f.write(str(len(new_struct.elements))+'\n')
        f.write(f'{struct.charge} {struct.spin}\n')
        for ele, coord in zip(new_struct.elements, new_struct.coords):
            f.write(' '.join([ele, str(coord[0]), str(coord[1]), str(coord[2])])+'\n')
    xyz_dict, charge, spin = read_xyz_to_dict(tmp)
    if test:
        data = gen_test_pkl(xyz_dict, charge, spin, basis, grid_level)
    else:
        data = gen_pkl(xyz_dict, charge, spin, basis, grid_level)
    with open(fn, 'wb') as f:
        pk.dump(data, f)
    print(f'Ellapsed time: {time.time()-sta_time:.2f}s')
    print(data['converged'])
    
def gen_train_data(path_list, eps=1e-7, a=0.9, n_samples=6, priority=0, basis='aug-cc-pvtz', grid_level=3, base_method='b3lyp', target_method='ccsd', e_T2=True, train=True):
    if 'b3' in base_method.lower():
        base_method = 'b3'
    target_method = target_method.lower()
    for fn in path_list:
        if not os.path.exists(fn):
            stretch_mol(fn, priority, basis, grid_level)
    x, e, rho, rho_base, valid_list, e_base, e_T, e_J, e_ext, e_xc, coeff = [], [], [], [], [], [], [], [], [], [], []
    for fn in path_list:
        print(f"generating {'training' if train else 'validating'} data for", fn)
        with open(fn, 'rb') as f:
            data = pk.load(f)
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in
                zip(data['atoms_charges'], data['atoms_coords'])], basis=data['basis'],
                        spin=data['spin'], charge=data['charge'], unit='bohr', verbose=0)
        ni = dft.numint.NumInt()
        phi012 = ni.eval_ao(mol, data['gc'], deriv=2)
        data['phi01'] = phi012[:4]
        phi2 = phi012[[4,7,9],:]
        if e_T2:
            e_T_base = .5*dm2eT2(data[f'dm_{base_method}'], data['phi01'][0], phi2)
            e_T_target = data[f'e_T_{target_method}2']
        else:
            e_T_base = .5*dm2eT(data[f'dm_{base_method}'], data['phi01'][1:])
            e_T_target = data[f'e_T_{target_method}']
        nu = mol.intor('int1e_grids_sph', grids=data['gc'])
        mu = cal_mu(data['gc'], data['atoms_charges'], data['atoms_coords'])
        e_ext_base = mu*data[f'rho_{base_method}']
        e_ext_target = mu*data[f'rho_{target_method}']
        e_J_base = 0.5*dm2eJ(data[f'dm_{base_method}'], data['phi01'][0], nu)
        e_J_target = 0.5*dm2eJ(data[f'dm_{target_method}'], data['phi01'][0], nu)
        if base_method == 'b3':
            e_xc_base = ni.eval_xc('b3lypg', data['rho01_b3'], data['spin'], relativity=0, deriv=0, verbose=None)[0]*data['rho_b3'] - 0.2*.25*dm2eK(data['dm_b3'], data['phi01'][0], nu)
        else:
            # TODO
            e_xc_base = None
        e_corr = e_T_target-e_T_base+e_ext_target-e_ext_base+e_J_target-e_J_base+data[f'e_xc_{target_method}']-e_xc_base
        if train:
            x.append(get_feature(data, mol, ni, a, n_samples, dm=base_method))
            rho.append(data[f'rho_{target_method}'])
            rho_base.append(data[f'rho_{base_method}'])
            e_base.append((e_T_base+e_ext_base+e_J_base+e_xc_base)/(data[f'rho_{base_method}']+eps))
            e_T.append(e_T_base)
            e_J.append(e_J_base)
            e_ext.append(e_ext_base)
            e_xc.append(e_xc_base)
            e.append(e_corr/(data[f'rho_{base_method}']+eps))
            coeff.append(np.array([e_T_target/e_T_base, e_ext_target/e_ext_base, e_J_target/e_J_base, data[f'e_xc_{target_method}']/e_xc_base]).T)
        else:
            x = get_feature(data, mol, ni, a, n_samples, dm=base_method)
            e = e_corr/(data[f'rho_{base_method}']+eps)
            valid_list.append({'x':x, 'e':e, 'rho_base':data[f'rho_{base_method}'], 'gc':data['gc'], 'gw':data['gw'], 'rho_target':data[f'rho_{target_method}'],
                          'fn':fn, 'atoms_charges':data['atoms_charges'], 'atoms_coords':data['atoms_coords'], 'dipole_base':data[f'dipole_{base_method}'],
                          'dipole_target':data[f'dipole_{target_method}'], 'E_base':data[f'E_{base_method}'], 'E_target':data[f'E_{target_method}'], 'I_base':data[f'I_{base_method}'],
                          'spin':data['spin'], 'charge':data['charge'],
                          'E_N':data['E_N'],
                          'e_T_target':e_T_target, 'e_T_base':e_T_base,
                        #   'e_T_target2':data[f'e_T_{target_method}2'], 'e_T_base2':.5*dm2eT2(data[f'dm_{base_method}'], data['phi01'][0], phi2),
                          'e_ext_target':e_ext_target, 'e_ext_base':e_ext_base,
                          'e_J_target':e_J_target, 'e_J_base':e_J_base,
                          'e_xc_target':data[f'e_xc_{target_method}'], 'e_xc_base':e_xc_base,
                          'e_base':(e_T_base+e_ext_base+e_J_base+e_xc_base)/(data[f'rho_{base_method}']+eps),
                          'coeff':np.array([e_T_target/e_T_base, e_ext_target/e_ext_base, e_J_target/e_J_base, data[f'e_xc_{target_method}']/e_xc_base]).T})
    if train:
        return np.vstack(x), np.concatenate(e), np.concatenate(rho), np.concatenate(rho_base), np.concatenate(e_base), np.concatenate(e_T), np.concatenate(e_ext), np.concatenate(e_J), np.concatenate(e_xc), np.concatenate(coeff), path_list
    else:
        return valid_list


if __name__ == "__main__":
    from cfg import get_cfg
    import sys
    from hashlib import md5
    import xgboost as xgb
    import model
    yml = get_cfg(sys.argv[1])
    trainset = []
    for molname, bond_lengths in yml.train.geo.items():
        for bond_length in bond_lengths:
            if type(bond_length)==list:
                for bd in np.linspace(*bond_length):
                    trainset.append(f'{molname}_{bd:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')            
            else:
                trainset.append(f'{molname}_{bond_length:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')
    trainset = set([os.path.join(yml.homepath, f'data/pkl/{f}') for f in trainset])
    md5value = md5(str(sorted(trainset)).encode()).hexdigest()

    model_path = os.path.join(yml.homepath, f'model/{md5value:.10}_{yml.base_method}_to_{yml.target_method}_{yml.n_samples}_eT2_{yml.e_T2}_nn_test.pt')
    if os.path.exists(model_path) and not yml.retrain:
        print(model_path, 'already exists.')
    else:
        traindata = gen_train_data(trainset, a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.train.bond_priority, basis=yml.basis, grid_level=yml.grid_level, base_method=yml.base_method, target_method=yml.target_method, e_T2=yml.e_T2)
        x, e, rho, rho_base, _, e_T, e_ext, e_J, e_xc, coeff = traindata[:10]
        e = e*(rho_base+1e-7)
        x = x.reshape(len(x), -1)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        print(x.shape, e.shape)
        print('e mean', e.mean(), e.shape)
        dataFlag = True
        with open(os.path.join(yml.homepath, f'model/meta'), 'a') as f:
            f.write(f"{md5value:.10} ->{str(trainset)}\n")
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test, coeff_train, coeff_test = train_test_split(x, e, coeff, test_size=0.1)
        print(x_train.shape)
        print(x_test.shape)
        model = model.Model(feature_num=len(x_train[0]))
        model.fit(x_train, y_train, x_test, y_test, coeff_train, coeff_test, model_path)
    

