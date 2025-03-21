import pickle as pk
import numpy as np
from build import Tree
from pyscf import gto, dft, cc, scf
import matplotlib.pyplot as plt
from einsum import dm2rho, dm2rho01, dm2eT2, dm2eJ, dm2eK, cal_dipole, cal_I, dm2rho01_sep
from train_fcn import gen_test_data_E
import time
import os
import torch

if __name__ == "__main__":
    from model import ModelE, ModelD, ModelF
    from cfg import get_cfg
    import sys
    import pandas as pd
    yml = get_cfg(sys.argv[1])
    if not yml.test.csv_name:
        yml.test.csv_name = os.path.basename(yml.test.model)[:-3]+'_test.csv'
    df = pd.DataFrame(columns=['Molecule', 'Bond length', f'E {yml.base_method}', 'E model'])
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
        test = gen_test_data_E([fn], a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.test.bond_priority, basis=yml.basis, grid_level=yml.grid_level, base_method=yml.base_method)[0]
        if count==0:
            model_e = ModelE(feature_num=len(test['x'][0]), device=yml.device)
            model_e.load(yml.test.model)
        test['x'] = torch.tensor(test['x']).to(model_e.device)
        with torch.no_grad():
            inference_e = model_e.forward(test['x']).detach().cpu().numpy()
        df.loc[count] = [*os.path.basename(fn).split('_')[:2], test['E_base'], test['E_base']+np.sum(inference_e*test['rho_base']*test['gw'])]
        df.to_csv(yml.test.csv_name, index=False)
    