import pickle as pk
import numpy as np
from pyscf import gto, dft, cc, scf
from einsum import dm2rho, dm2rho01, dm2eT2, dm2eJ, dm2eK, cal_dipole, cal_I, dm2rho01_sep
from train_fcn import cal_mu, get_feature, gen_train_data_E
import time
import os
import torch

if __name__ == "__main__":
    from model import ModelE
    from cfg import get_cfg
    import sys
    import pandas as pd
    yml = get_cfg(sys.argv[1])
    if not yml.valid.csv_name:
        yml.valid.csv_name = os.path.basename(yml.valid.model)[:-3]+'_valid.csv'
    df = pd.DataFrame(columns=['Molecule', 'Bond length', f'E {yml.base_method}', 'E model', f'E {yml.target_method}', f'Error E {yml.base_method} in kcal/mol', f'Error E model in kcal/mol',
                               'time KS', 'time feature', 'time model', 'npoints', 'nbasis'])
    assert os.path.exists(yml.valid.model), f'no {yml.valid.model}, pls assign an existing model.'
    print('loading', yml.valid.model)
    validset = []
    for molname, bond_lengths in yml.valid.geo.items():
        for bond_length in bond_lengths:
            if type(bond_length)==list:
                for bd in np.linspace(*bond_length):
                    if '*' in molname:
                        repo = os.path.dirname(molname)
                        for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                            if mn.endswith('.xyz'):
                                validset.append(f'{repo}/{mn[:-4]}_{bd:.4f}_{yml.valid.bond_priority}_{yml.basis}_train_valid.pkl')
                    else:
                        validset.append(f'{molname}_{bd:.4f}_{yml.valid.bond_priority}_{yml.basis}_train_valid.pkl')
            else:
                if '*' in molname:
                    repo = os.path.dirname(molname)
                    for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                        if mn.endswith('.xyz'):
                            validset.append(f'{repo}/{mn[:-4]}_{bond_length:.4f}_{yml.valid.bond_priority}_{yml.basis}_train_valid.pkl')
                else:
                    validset.append(f'{molname}_{bond_length:.4f}_{yml.valid.bond_priority}_{yml.basis}_train_valid.pkl')
    validset = set([os.path.join(yml.homepath, f'data/pkl/{f}') for f in validset])
    rec = []
    i = 1
    for count, fn in enumerate(sorted(validset)):
        valid = gen_train_data_E([fn], a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.valid.bond_priority, basis=yml.basis, grid_level=yml.grid_level, base_method=yml.base_method, target_method=yml.target_method, e_T2=True, train=False, complex_geo=yml.train.complex_geo)[0]
        if count==0:
            model_e = ModelE(feature_num=len(valid['x'][0]), device=yml.device)
            model_e.load(yml.valid.model)
        valid['x'] = torch.tensor(valid['x']).to(model_e.device)
        with torch.no_grad():
            if i == 1:
                inference_e = model_e.forward(valid['x']).detach().cpu().numpy()
                i += 1
            deltaT = time.time()
            inference_e = model_e.forward(valid['x']).detach().cpu().numpy()
            deltaT = time.time() - deltaT
        
        df.loc[count] = [*os.path.basename(fn).split('_')[:2], valid['E_base'], valid['E_base']+np.sum(inference_e*valid['rho_base']*valid['gw']), valid['E_target'], (valid['E_base']-valid['E_target'])*627, (valid['E_base']+np.sum(inference_e*valid['rho_base']*valid['gw'])-valid['E_target'])*627,
                         valid['timeks'], valid['timefeature'], deltaT, valid['npoints'], valid['nbasis']]
        df.to_csv(yml.valid.csv_name, index=False)
    ma = []
    for column in df:
        if df[column].dtype == np.float64:
            ma.append(np.abs(df[column].to_numpy()).mean())
        else:
            ma.append(np.NaN)
    ma[0] = 'MA'
    df.loc[count+1] = ma
    df.to_csv(yml.valid.csv_name, index=False)
