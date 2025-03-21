import pickle as pk
import numpy as np
from build import Tree
from pyscf import gto, dft, cc, scf
import matplotlib.pyplot as plt
from einsum import dm2rho, dm2rho01, dm2eT2, dm2eJ, dm2eK, cal_dipole, cal_I, dm2rho01_sep
from train_fcn_b3 import cal_mu, get_feature, gen_test_data_E
import time
import os
import torch

if __name__ == "__main__":
    from model import ModelE_b3, ModelE_post
    from cfg import get_cfg
    import sys
    import pandas as pd
    yml = get_cfg(sys.argv[1])
    if not yml.test.csv_name:
        yml.test.csv_name = os.path.basename(yml.test.model)[:-3]+'_testb3.csv'
    df = pd.DataFrame(columns=['Molecule', 'Bond length', f'E {yml.base_method}', 'E scf', 'E model', f'Dipole {yml.base_method}', 'Dipole model'])
    assert os.path.exists(yml.test.model), f'no {yml.test.model}, pls assign an existing model.'
    print('loading', yml.test.model)
    testset = []
    for molname, bond_lengths in yml.test.geo.items():
        for bond_length in bond_lengths:
            if type(bond_length)==list:
                for bd in np.linspace(*bond_length):
                    if '*' in molname:
                        repo = os.path.dirname(molname)
                        for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                            if mn.endswith('.xyz'):
                                testset.append(f'{repo}/{mn[:-4]}_{bd:.4f}_{yml.test.bond_priority}_{yml.basis}_test.pkl')
                    else:
                        testset.append(f'{molname}_{bd:.4f}_{yml.test.bond_priority}_{yml.basis}_test.pkl')
            else:
                if '*' in molname:
                    repo = os.path.dirname(molname)
                    for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                        if mn.endswith('.xyz'):
                            testset.append(f'{repo}/{mn[:-4]}_{bond_length:.4f}_{yml.test.bond_priority}_{yml.basis}_test.pkl')
                else:
                    testset.append(f'{molname}_{bond_length:.4f}_{yml.test.bond_priority}_{yml.basis}_test.pkl')
    testset = set([os.path.join(yml.homepath, f'data/pkl/{f}') for f in testset])
    for count, fn in enumerate(sorted(testset)):
        # if os.path.basename(fn).split('_')[0] in yml.train.complex_geo:
        #     print(fn, 'is too complex, not to be tested ...')
        #     continue
        test = gen_test_data_E([fn], a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.test.bond_priority, basis=yml.basis, grid_level=yml.grid_level, base_method='b3lypg')[0]
        if count==0:
            model_e = ModelE_b3(device=yml.device)
            model_e.load(yml.test.model)
            
            model_post = ModelE_post(feature_num=5, device=yml.device)
            model_post.load(yml.test.model.replace('b3pre2', '_post2b3pre2'))
        
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in 
            zip(test['atoms_charges'], test['atoms_coords'])], basis=yml.basis, 
                    spin=test['spin'], charge=test['charge'], unit='bohr')
        ni = dft.numint.NumInt()
        test['phi01'] = ni.eval_ao(mol, test['gc'], deriv=1)
        res = model_e.scf(test, yml.basis, grid_level=3, hyb=0.2, xc='b3lypg')
        e_post = model_post.forward(res['rho01_scf']).detach().cpu().numpy()
        E_post = np.sum(e_post*res['rho_scf']*test['gw'])
        # error_F_base = np.sum((test['F_base'] - test['F_target'])**2)**.5
        # error_F_model = np.sum((res['F_scf'] - test['F_target'])**2)**.5
        df.loc[count] = [*os.path.basename(fn).split('_')[:2], test['E_base'], res['E_scf'], res['E_scf']+E_post, 
                         test['dipole_base'], cal_dipole(res['rho_scf'], test['gc'], test['gw'], test['atoms_charges'], test['atoms_coords'])]
        df.to_csv(yml.test.csv_name, index=False)