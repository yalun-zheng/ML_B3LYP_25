import pickle as pk
import numpy as np
# from build import Tree
from pyscf import gto, dft, scf, cc
from einsum import dm2rho01, dm2eT2, dm2eJ, dm2eK, dm2rho01_sep, dm2eT, cal_dipole, dm2J, dm2K, dm2E1e
from opt_einsum import contract
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6, 7'
import time
from mol.move import MolGraph, draw, stretch
from gen_eden import gen_pkl, read_xyz_to_dict, gen_test_pkl, xyz_dict_to_str
# from pyscf.grad.uccsd_t import Gradients as guccsdt
# from pyscf.grad.ccsd_t import Gradients as gccsdt
# from pyscf.grad.uccsd import Gradients as guccsd
# from pyscf.grad.ccsd import Gradients as gccsd
from pyscf.grad.uks import Gradients as guks
from pyscf.grad.rks import Gradients as gks
from pyscf.grad.rhf import Gradients as ghf
from pyscf.grad.uhf import Gradients as guhf
import dftd3.pyscf as d3
# import torch
# torch.set_default_dtype(torch.float64)


def gen_cube(center, mol, dm, mesh, ni, *args):
    if len(dm)==2:
        dm = dm[0] + dm[1]
    if len(mesh)==1:
        cube = center + mesh
    else:
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
    if len(mesh)==1:
        cube = center + mesh
    else:
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
    '''
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
    return np.concatenate((rho, gnorm, tau), axis=0).T'''

    phi01 = ni.eval_ao(mol, cube, deriv=1)
    rho01 = dm2rho01(dm[0], phi01), dm2rho01(dm[1], phi01)
    rho = np.vstack((rho01[0][0], rho01[1][0]))
    gnorm = contract('dr,dr->r', rho01[0][1:], rho01[0][1:]), contract('dr,dr->r', rho01[0][1:], rho01[1][1:]), contract('dr,dr->r', rho01[1][1:], rho01[1][1:])
    gnorm = np.vstack(gnorm)/((rho01[0][0]+rho01[1][0]+1e-7)**(4/3))/(2*(3*np.pi)**(1/3))
    tau = 0.5*contract('ij, dri, drj->r', dm[0], phi01[1:], phi01[1:]), 0.5*contract('ij, dri, drj->r', dm[1], phi01[1:], phi01[1:])
    t_unif = 3/10*(3*np.pi**2)**(2/3)*(rho01[0][0]+rho01[1][0]+1e-7)**(5/3)
    tau = np.vstack(tau)/(t_unif+1e-7)
    # rho_up, rho_down, g_up, g_down, g, tau_up, tau_down
    return np.concatenate((rho, gnorm, tau), axis=0).T

def get_mesh(a, n_samples=6):
    pos = np.linspace(-a/2+a/(n_samples-1)*.5, a/2-a/(n_samples-1)*.5, n_samples-1)
    mesh = np.meshgrid(pos,pos,pos, indexing='ij')
    mesh = np.stack(mesh, axis=-1).reshape(-1,3)
    in_sphere_id = contract('ri,ri->r', mesh, mesh)<=(.5*a)**2
    mesh = mesh[in_sphere_id]
    return mesh

def get_feature(data, mol, ni, a, n_samples=6, dm='b3'):
    if n_samples==2:
        return get_feature1(data, mol, ni, a, dm)
    mesh = get_mesh(a, n_samples)
    res = []
    for center in data['gc']:
        cube = gen_cube_spin(center, mol, data['dm_'+dm], mesh, ni)
        res.append(cube)
    return np.array(res).reshape(len(res), -1)

def get_feature1(data, mol, ni, a, dm='b3'):
    dm = data['dm_'+dm]
    if len(dm) != 2:
        dm = 0.5*dm, 0.5*dm
    assert len(dm) == 2
    phi01 = ni.eval_ao(mol, data['gc'], deriv=1)
    rho01 = dm2rho01(dm[0], phi01), dm2rho01(dm[1], phi01)
    # rho = np.vstack((rho01[0][0], rho01[1][0]))
    # gnorm = contract('dr,dr->r', rho01[0][1:], rho01[0][1:]), contract('dr,dr->r', rho01[0][1:], rho01[1][1:]), contract('dr,dr->r', rho01[1][1:], rho01[1][1:])
    # gnorm = np.vstack(gnorm)/((rho01[0][0]+rho01[1][0]+1e-7)**(4/3))/(2*(3*np.pi)**(1/3))
    # tau = 0.5*contract('ij, dri, drj->r', dm[0], phi01[1:], phi01[1:]), 0.5*contract('ij, dri, drj->r', dm[1], phi01[1:], phi01[1:])
    # t_unif = 3/10*(3*np.pi**2)**(2/3)*(rho01[0][0]+rho01[1][0]+1e-7)**(5/3)
    # tau = np.vstack(tau)/(t_unif+1e-7)
    # rho_up, rho_down, g_up, g_down, g, tau_up, tau_down
    # return np.concatenate((rho, gnorm), axis=0).T
    # rho is ((den_u,grad_xu,grad_yu,grad_zu)
    #                 (den_d,grad_xd,grad_yd,grad_zd))
    return rho01

def cal_mu(gc, atoms_charges, atoms_coords):
    res = 0
    for z, c in zip(atoms_charges, atoms_coords):
        dc = np.linalg.norm(gc-c, axis=-1)
        res -= z/dc
    return res

def stretch_mol(fn, priority, basis, grid_level, skip_dm=False):
    print(f'no pkl file {fn}, generating it...')
    sta_time = time.time()
    xyz_path = os.path.join(os.path.dirname(fn).replace('/pkl/', '/xyz/'), os.path.basename(fn).split('_')[0]+'.xyz')
    assert os.path.exists(xyz_path), f'error! pls put {os.path.basename(xyz_path)} into {os.path.dirname(xyz_path)}'
    dist = float(os.path.basename(fn).split('_')[1])
    struct = MolGraph(xyz_path)
    if len(struct)>1 and abs(dist) > 1e-6:
        i, j = draw(struct, priority=priority)
        new_struct = stretch(struct, i, j, dist)
    else:
        new_struct = struct
    repo = os.path.basename(os.path.dirname(fn))
    tmp = fn.replace('.pkl', '.xyz').replace('_train_valid', '').replace('data/pkl', 'data/tmp').replace(f'/{repo}/', '/')
    tmp = fn.replace('.pkl', '.xyz').replace('_test', '').replace('data/pkl', 'data/tmp').replace(f'/{repo}/', '/')
    with open(tmp, 'w') as f:
        f.write(str(len(new_struct.elements))+'\n')
        f.write(f'{struct.charge} {struct.spin}\n')
        for ele, coord in zip(new_struct.elements, new_struct.coords):
            f.write(' '.join([ele, str(coord[0]), str(coord[1]), str(coord[2])])+'\n')
    xyz_dict, charge, spin = read_xyz_to_dict(tmp)
    if skip_dm:
        data = gen_test_pkl(xyz_dict, charge, spin, basis, grid_level)
    else:
        data = gen_pkl(xyz_dict, charge, spin, basis, grid_level)
    with open(fn, 'wb') as f:
        pk.dump(data, f)
    print(f'Ellapsed time: {time.time()-sta_time:.2f}s')
    print(data['converged'])


def gen_train_data_E(path_list, a=0.9, n_samples=6, priority=0, basis='aug-cc-pvtz', grid_level=3, base_method='b3lypg', target_method='ccsd', e_T2=True, train=True, complex_geo=[], model=None):
    if 'b3' in base_method.lower():
        base_method = 'b3'
    target_method = target_method.lower()
    for fn in path_list:
        if not os.path.exists(fn):
            molname = os.path.basename(fn).split('_')[0]
            stretch_mol(fn, priority, basis, grid_level, skip_dm=True if molname in complex_geo else False)
    valid_list = []
    for fn in path_list:
        print(f"generating {'training' if train else 'validating'} data for", fn)
        with open(fn, 'rb') as f:
            data = pk.load(f)
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in
                zip(data['atoms_charges'], data['atoms_coords'])], basis=data['basis'],
                        spin=data['spin'], charge=data['charge'], unit='bohr', verbose=0)
        
        ni = dft.numint.NumInt()
        if model is not None:
            base_method = 'scf'
            # res = model.scf(data, data['basis'], grid_level=grid_level, dm_target=data.get(f'dm_{target_method}', data['dm_ccsd']))
            res = model.scf(data, data['basis'], grid_level=grid_level, xc='b3lypg', xctype='GGA', hyb=.2)
            data.update(res)
            x = res['rho01_scf']
        else:
            x = get_feature(data, mol, ni, a, n_samples, dm=target_method if f'dm_{target_method}' in data else 'ccsd')
        
        # itg_2e = mol.intor('int2e_sph')
        
        
        # phi012 = ni.eval_ao(mol, data['gc'], deriv=2)
        # phi2 = np.zeros((3,3)+phi012.shape[-2:])
        # phi2[0] = phi012[4:7]
        # phi2[:,0] = phi012[4:7]
        # phi2[1,1] = phi012[7]
        # phi2[1,2] = phi012[8]
        # phi2[2,1] = phi012[8]
        # phi2[2,2] = phi012[9]
        # x = get_feature(data, mol, ni, a, n_samples, dm=target_method if f'dm_{target_method}' in data else 'ccsd')
        # x = get_feature(data, mol, ni, a, n_samples, dm=base_method)
        
        if 'T' in data:
            itg_2e = mol.intor('int2e_sph')  
            F0 = data['T'] + data['V_ext'] + dm2J(data.get(f'dm_{target_method}', data['dm_ccsd']), itg_2e)
            K = (dm2K(data[f'dm_{base_method}'][0], itg_2e), dm2K(data[f'dm_{base_method}'][1], itg_2e)) if data['spin'] else dm2K(data[f'dm_{base_method}'], itg_2e)
            valid_list.append({'x':x, 'rho_base':data[f'rho_{base_method}'], 'gc':data['gc'], 'gw':data['gw'], 'rho_target':data.get(f'rho_{target_method}', data['rho_ccsd']),
                            'fn':fn, 'atoms_charges':data['atoms_charges'], 'atoms_coords':data['atoms_coords'], 'dipole_base':data[f'dipole_{base_method}'], 'dipole_target':data.get(f'dipole_{target_method}', data['dipole_ccsd']),
                            'E_base':data[f'E_{base_method}'], 'E_target':data[f'E_{target_method}'],
                            'spin':data['spin'], 'charge':data['charge'],
                            'E_N':data['E_N'], 'dm_target':data.get(f'dm_{target_method}', data['dm_ccsd']), 'dm_base': data[f'dm_{base_method}'],
                            'F_base': data[f'F_{base_method}'], 'F_target': data.get(f'F_{target_method}', data['F_ccsd']),
                            'n_elec': data['n_elec'], 'F0': F0, 'S': data['S'], 'phi01': data['phi01'], 'rho01': data.get(f'rho01_{target_method}', data['rho01_ccsd']),
                            'J': F0 - (data['T'] + data['V_ext']), 'E_N': data['E_N'], 'H1':data['T'] + data['V_ext'], 'itg_2e':itg_2e, 'K': K,
                            'T': data['T'], 'E_ext_J': 0.5*dm2E1e(data.get(f'dm_{target_method}', data['dm_ccsd']), F0 - (data['T'] + data['V_ext']))+dm2E1e(data.get(f'dm_{target_method}', data['dm_ccsd']), data['V_ext']),
                            'basis': data['basis'], 'V_ext': data['V_ext'],
                            # 'offset': mol.offset_nr_by_atom(), 'phi2': phi2, 'phi01': data['phi01'], 'dS': g.get_ovlp(),
                            # 'mo_coeff': ks.mo_coeff, 'mo_occ': ks.mo_occ, 'mo_energy': ks.mo_energy,
                            })
        else:
            valid_list.append({'x':x, 'rho_base':data[f'rho_{base_method}'], 'gc':data['gc'], 'gw':data['gw'], 'rho_target':data.get(f'rho_{target_method}', data['rho_ccsd']),
                            'fn':fn, 'atoms_charges':data['atoms_charges'], 'atoms_coords':data['atoms_coords'], 'dipole_base':data[f'dipole_{base_method}'], 'dipole_target':data.get(f'dipole_{target_method}', data['dipole_ccsd']),
                            'E_base':data[f'E_{base_method}'], 'E_target':data[f'E_{target_method}'],
                            'spin':data['spin'], 'charge':data['charge'],
                            'E_N':data['E_N'], 'dm_target':data.get(f'dm_{target_method}', data['dm_ccsd']), 'dm_base': data[f'dm_{base_method}'],
                            'F_base': data[f'F_{base_method}'], 'F_target': data.get(f'F_{target_method}', data['F_ccsd']),
                            'phi01': data['phi01'], 'rho01': data[f'rho01_{base_method}'],
                            })
        # for valid in valid_list:
        #     for k in valid:
        #         if k not in ['fn', 'spin', 'charge', 'offset', 'mo_occ', 'atoms_charges']:
        #             valid[k] = np.array(valid[k]).astype(np.float32)
    return valid_list


def gen_test_data_E(path_list, a=0.9, n_samples=6, priority=0, basis='aug-cc-pvtz', grid_level=3, base_method='b3lyp'):
    if 'b3' in base_method.lower():
        base_method = 'b3'
    for fn in path_list:
        if not os.path.exists(fn):
            molname = os.path.basename(fn).split('_')[0]
            print(f'no pkl file {fn}, generating it...')
            xyz_path = os.path.join(os.path.dirname(fn).replace('/pkl/', '/xyz/'), os.path.basename(fn).split('_')[0]+'.xyz')
            assert os.path.exists(xyz_path), f'error! pls put {os.path.basename(xyz_path)} into {os.path.dirname(xyz_path)}'
            dist = float(os.path.basename(fn).split('_')[1])
            struct = MolGraph(xyz_path)
            if len(struct)>1 and abs(dist) > 1e-6:
                i, j = draw(struct, priority=priority)
                new_struct = stretch(struct, i, j, dist)
            else:
                new_struct = struct
            repo = os.path.basename(os.path.dirname(fn))
            tmp = fn.replace('.pkl', '.xyz').replace('_train_valid', '').replace('data/pkl', 'data/tmp').replace(f'/{repo}/', '/')
            tmp = fn.replace('.pkl', '.xyz').replace('_test', '').replace('data/pkl', 'data/tmp').replace(f'/{repo}/', '/')
            with open(tmp, 'w') as f:
                f.write(str(len(new_struct.elements))+'\n')
                f.write(f'{struct.charge} {struct.spin}\n')
                for ele, coord in zip(new_struct.elements, new_struct.coords):
                    f.write(' '.join([ele, str(coord[0]), str(coord[1]), str(coord[2])])+'\n')
            xyz_dict, charge, spin = read_xyz_to_dict(tmp)
            data = {}
            mol = gto.M(atom=xyz_dict_to_str(xyz_dict), charge=charge, spin=spin, basis=basis)
            data['atoms_charges'] = mol.atom_charges()
            data['atoms_coords'] = mol.atom_coords()
            data['basis'] = basis
            data['spin'] = spin
            data['charge'] = charge
            
            # B3 calculations
            df = dft.KS(mol)
            df.xc = 'b3lypg'
            df.grids.level = grid_level
            df.grids.build()
            data['gc'], data['gw'] = df.grids.coords, df.grids.weights
            ni = dft.numint.NumInt()
            data['phi01'] = ni.eval_ao(mol, data['gc'], deriv=1)
            mf = scf.HF(mol)
            mf.kernel()
            data['n_elec'] = mol.nelectron
            data['E_N'] = mol.energy_nuc()
            df.kernel()
            data['E_b3'] = df.e_tot
            data['dm_b3'] = df.make_rdm1()
            data['E_hf'] = mf.e_tot
            data['dm_hf'] = mf.make_rdm1()
            if data['spin']:
                data['rho01_b3'] = dm2rho01(data['dm_b3'][0], data['phi01']), dm2rho01(data['dm_b3'][1], data['phi01'])
                data['rho01_hf'] = dm2rho01(data['dm_hf'][0], data['phi01']), dm2rho01(data['dm_hf'][1], data['phi01'])
                data['rho_b3'] = data['rho01_b3'][0][0] + data['rho01_b3'][1][0]
                data['rho_hf'] = data['rho01_hf'][0][0] + data['rho01_hf'][1][0]
                data['F_b3'] = -guks(df).kernel()
                data['F_hf'] = -guhf(mf).kernel()
            else:
                data['F_b3'] = -gks(df).kernel()
                data['F_hf'] = -ghf(mf).kernel()
                data['rho01_b3'] = dm2rho01(data['dm_b3'], data['phi01'])
                data['rho01_hf'] = dm2rho01(data['dm_hf'], data['phi01'])
                data['rho_b3'] = data['rho01_b3'][0]
                data['rho_hf'] = data['rho01_hf'][0]
            data['dipole_b3'] = cal_dipole(data['rho_b3'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
            data['dipole_hf'] = cal_dipole(data['rho_hf'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
            # data['dipole_pbe'] = cal_dipole(data['rho_pbe'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
            data['d3bj'] = d3.DFTD3Dispersion(mol, xc="b3lyp", version="d3bj").kernel()[0]
            with open(fn, 'wb') as f:
                pk.dump(data, f)
    test_list = []
    for fn in path_list:
        print(f"generating testing data for", fn)
        with open(fn, 'rb') as f:
            data = pk.load(f)
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in
                zip(data['atoms_charges'], data['atoms_coords'])], basis=data['basis'],
                        spin=data['spin'], charge=data['charge'], unit='bohr', verbose=0)
        ni = dft.numint.NumInt()
        x = get_feature(data, mol, ni, a, n_samples, dm=base_method)
        test_list.append({'x':x, 'rho_base':data[f'rho_{base_method}'], 'gc':data['gc'], 'gw':data['gw'],
                        'fn':fn, 'atoms_charges':data['atoms_charges'], 'atoms_coords':data['atoms_coords'], 'dipole_base':data[f'dipole_{base_method}'],
                        'E_base':data[f'E_{base_method}'],
                        'spin':data['spin'], 'charge':data['charge'],
                        'E_N':data['E_N'], 'dm_base': data[f'dm_{base_method}'],
                        'F_base': data[f'F_{base_method}'], 
                        'phi01': data['phi01'], 'rho01': data[f'rho01_{base_method}'], })
    return test_list

if __name__ == "__main__":
    from cfg import get_cfg
    import sys
    from hashlib import md5
    import xgboost as xgb
    from model import ModelE_b3, ModelE_post
    yml = get_cfg(sys.argv[1])
    trainset = []
    for molname, bond_lengths in yml.train.geo.items():
        for bond_length in bond_lengths:
            if type(bond_length)==list:
                for bd in np.linspace(*bond_length):
                    if '*' in molname:
                        repo = os.path.dirname(molname)
                        for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                            if mn.endswith('.xyz'):
                                trainset.append(f'{repo}/{mn[:-4]}_{bd:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')
                    else:
                        trainset.append(f'{molname}_{bd:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')
            else:
                if '*' in molname:
                    repo = os.path.dirname(molname)
                    for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                        if mn.endswith('.xyz'):
                            trainset.append(f'{repo}/{mn[:-4]}_{bond_length:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')
                else:
                    trainset.append(f'{molname}_{bond_length:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')
    trainset = set([os.path.join(yml.homepath, f'data/pkl/{f}') for f in trainset])
    # trainset = set([os.path.join(yml.homepath, f'data/pkl/{f}') for f in trainset if os.path.basename(f).split('_')[0] not in yml.train.complex_geo])
    
        
    md5value = md5(str(sorted(trainset)).encode()).hexdigest()

    validset = []
    for molname, bond_lengths in yml.train.valid_geo.items():
        for bond_length in bond_lengths:
            if type(bond_length)==list:
                for bd in np.linspace(*bond_length):
                    if '*' in molname:
                        repo = os.path.dirname(molname)
                        for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                            if mn.endswith('.xyz'):
                                validset.append(f'{repo}/{mn[:-4]}_{bd:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')
                    else:
                        validset.append(f'{molname}_{bd:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')
            else:
                if '*' in molname:
                    repo = os.path.dirname(molname)
                    for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                        if mn.endswith('.xyz'):
                            validset.append(f'{repo}/{mn[:-4]}_{bond_length:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')
                else:
                    validset.append(f'{molname}_{bond_length:.4f}_{yml.train.bond_priority}_{yml.basis}_train_valid.pkl')
    validset = set([os.path.join(yml.homepath, f'data/pkl/{f}') for f in validset])
    traindata =None
    model_path = os.path.join(yml.homepath, f'model/{md5value:.10}_{yml.basis}_{yml.base_method}_to_{yml.target_method}_{yml.n_samples}_newparam_zigzagb3pre15gganob3bigger.pt')
    if os.path.exists(model_path) and not yml.retrain:
        print(model_path, 'already exists.')
    else:
        traindata = gen_train_data_E(trainset, a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.train.bond_priority, basis=yml.basis, grid_level=yml.grid_level, base_method='b3lypg', target_method=yml.target_method, e_T2=yml.e_T2, train=True, complex_geo=yml.train.complex_geo)
        validdata = gen_train_data_E(validset, a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.train.bond_priority, basis=yml.basis, grid_level=yml.grid_level, base_method='b3lypg', target_method=yml.target_method, e_T2=yml.e_T2, train=False, complex_geo=yml.train.complex_geo)
        with open(os.path.join(yml.homepath, f'model/meta'), 'a') as f:
            f.write(f"{md5value:.10} ->{str(trainset)}\n")
        print(len(traindata))
        print(len(validdata))
        
        model = ModelE_b3(feature_num=3, device=yml.device)
        # model.load('/home/alfred/tree_regression/model/95f80d4876_aug-cc-pvdz_b3lypg_to_ccsdt_1_newparam_zigzag_old.pt')
        import torch
        # with torch.no_grad():
        for p in model.parameters():
            # torch.nn.init.zeros_(p)
            torch.nn.init.normal_(p, 0, 0.05)
        model.fit(traindata, validdata, model_path, lr=yml.train.lr, max_iter=5000, batch_size=yml.train.batch_size, off=False)

        new_path = model_path
        for iteration in range(0):
            print(iteration)
            model.load(new_path)
            new_path = model_path.replace('zigzag', 'zigzag_'+str(iteration))
            traindata = gen_train_data_E(trainset, a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.train.bond_priority, basis=yml.basis, grid_level=yml.grid_level, target_method=yml.target_method, e_T2=yml.e_T2, train=True, complex_geo=yml.train.complex_geo, model=model)
            validdata = gen_train_data_E(validset, a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.train.bond_priority, basis=yml.basis, grid_level=yml.grid_level, target_method=yml.target_method, e_T2=yml.e_T2, train=False, complex_geo=yml.train.complex_geo, model=model)
            # TODO 
            # model = ModelE_b3(device=yml.device)
            model.fit(traindata, validdata, new_path, lr=yml.train.lr, max_iter=4000, batch_size=yml.train.batch_size)
        
        model.load(new_path)
        traindata = gen_train_data_E(trainset, a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.train.bond_priority, basis=yml.basis, grid_level=yml.grid_level, target_method=yml.target_method, e_T2=yml.e_T2, train=True, complex_geo=yml.train.complex_geo, model=model)
        validdata = gen_train_data_E(validset, a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.train.bond_priority, basis=yml.basis, grid_level=yml.grid_level, target_method=yml.target_method, e_T2=yml.e_T2, train=False, complex_geo=yml.train.complex_geo, model=model)
        model = ModelE_post(feature_num=5, device=yml.device)
        new_path = model_path.replace('zigzag', 'zigzag_post2')
        model.fit(traindata, validdata, new_path, lr=yml.train.lr, max_iter=30000, batch_size=yml.train.batch_size)
