import pickle as pk
import numpy as np
from build import Tree
from pyscf import gto, dft, cc, scf
import matplotlib.pyplot as plt
from einsum import dm2rho, dm2rho01, dm2eT2, dm2eJ, dm2eK, cal_dipole, cal_I
from train import cal_mu, get_feature, stretch_mol
import time

def gen_test_data(path_list, eps=1e-7, a=0.9, n_samples=6, priority=0, basis='aug-cc-pvtz', grid_level=3, base_method='b3lyp'):
    if 'b3' in base_method.lower():
        base_method = 'b3'
    data_list = []
    for fn in path_list:
        if not os.path.exists(fn):
            stretch_mol(fn, priority, basis, grid_level, test=True)
    for fn in path_list:
        print('generating testing data for', fn)
        with open(fn, 'rb') as f:
            data = pk.load(f)
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in
                zip(data['atoms_charges'], data['atoms_coords'])], basis=data['basis'],
                        spin=data['spin'], charge=data['charge'], unit='bohr', verbose=0)
        ni = dft.numint.NumInt()
        phi012 = ni.eval_ao(mol, data['gc'], deriv=2)
        data['phi01'] = phi012[:4]
        x = get_feature(data, mol, ni, a, n_samples, dm=dm)
        data_list.append({'x':x, 'rho_base':data[f'rho_{base_method}'], 'gc':data['gc'], 'gw':data['gw'], 'fn':fn, 
                          'atoms_charges':data['atoms_charges'], 'atoms_coords':data['atoms_coords'],
                          'dipole_base':data[f'dipole_{base_method}'], 'E_base':data[f'E_{base_method}'], 'spin':data['spin'], 'charge':data['charge']})
    return data_list

if __name__ == "__main__":
    from cfg import get_cfg
    import sys
    import os
    yml = get_cfg(sys.argv[1])
    assert os.path.exists(yml.test.model), f'no {yml.test.model}, pls assign an existing model.'
    with open(yml.test.model, 'rb') as f:
        tree_e, tree_rho, scale = pk.load(f)
    testset = []
    for molname, bond_lengths in yml.test.geo.items():
        for bond_length in bond_lengths:
            if type(bond_length)==list:
                for bd in np.linspace(*bond_length):
                    testset.append(f'{molname}_{bd:.4f}_{yml.test.bond_priority}_{yml.basis}_test.pkl')            
            else:
                testset.append(f'{molname}_{bond_length:.4f}_{yml.test.bond_priority}_{yml.basis}_test.pkl')
    testset = set([os.path.join(yml.homepath, f'data/pkl/{f}') for f in testset])
    
    for fn in testset:
        test = gen_test_data([fn], a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.test.bond_priority, basis=yml.basis, grid_level=yml.grid_level, base_method=yml.base_method)[0]
        print(f'================{test["fn"]}================')
        inference_e = np.array([tree_e.inference(x.reshape(-1)) for x in (test['x']/scale)])
        
        inference_rho = np.array([tree_rho.inference(x.reshape(-1)) for x in (test['x']/scale)])**3
        neg_id = inference_rho<0
        neg_charge = np.sum(inference_rho[neg_id]*test['gw'][neg_id])
        pos_charge = np.sum(inference_rho[~neg_id]*test['gw'][~neg_id])
        ratio = abs(pos_charge/neg_charge)
        if ratio < 1:
            inference_rho[neg_id] *= ratio
        else:
            inference_rho[~neg_id] /= ratio
        res = {f'E_base': test['E_base'], 'rho_base': test['rho_base'], 'dipole_base': test['dipole_base']}
        if 'E' in yml.test.output:
            E = test['E_base']+np.sum(inference_e*test['rho_base']*test['gw'])
            res['E'] = E
        if 'F' in yml.test.output:
            mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in 
                zip(test['atoms_charges'], test['atoms_coords'])], basis=yml.basis, 
                        spin=test['spin'], charge=test['charge'], unit='bohr')
            mol.verbose = 0
            if yml.base_method == 'b3lypg':
                ks = dft.KS(mol)
                ks.xc = 'b3lypg'
                ks.grids.level = yml.grid_level
                ks.kernel()
                F_base = -np.array(ks.nuc_grad_method().kernel())
                res['F'] = F_base
        if 'dipole' in yml.test.output:
            dipole = cal_dipole(test['rho_base']+inference_rho, test['gc'], test['gw'], test['atoms_charges'], test['atoms_coords'])
            res['dipole'] = dipole
        if 'rho' in yml.test.output:
            res['rho'] = test['rho_base']+inference_rho
        print(res)
        with open(os.path.join(yml.homepath, f'data/test/{os.path.basename(fn)[:-8]}'+
            f'{os.path.basename(yml.test.model).split("_")[0]}_{os.path.basename(yml.test.model).split("_")[4][:-4]}.out'), 'wb') as f:
            pk.dump(res, f)
        