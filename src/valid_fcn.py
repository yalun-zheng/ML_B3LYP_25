import pickle as pk
import numpy as np
from build import Tree
from pyscf import gto, dft, cc, scf
import matplotlib.pyplot as plt
from einsum import dm2rho, dm2rho01, dm2eT2, dm2eJ, dm2eK, cal_dipole, cal_I, dm2rho01_sep
from train_fcn import cal_mu, get_feature, gen_train_data_E
import time
import os
import torch

if __name__ == "__main__":
    from model import ModelE, ModelD, ModelF
    from cfg import get_cfg
    import sys
    import pandas as pd
    yml = get_cfg(sys.argv[1])
    if not yml.valid.csv_name:
        yml.valid.csv_name = os.path.basename(yml.valid.model)[:-3]+'_valid.csv'
    df = pd.DataFrame(columns=['Molecule', 'Bond length', f'E {yml.base_method}', 'E model', f'E {yml.target_method}', f'Error E {yml.base_method} in kcal/mol', f'Error E model in kcal/mol',
                                                          f'Force {yml.base_method}', 'Force model', f'Force {yml.target_method}',
                                                          f'Dipole {yml.base_method}', 'Dipole model', f'Dipole {yml.target_method}', f'Error Force {yml.base_method}', 'Error Force model', 'Error N model', 'time KS', 'time feature', 'time model', 'npoints', 'nbasis'])
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
        # if os.path.basename(fn).split('_')[0] in yml.train.complex_geo:
        #     print(fn, 'is too complex, not to be validated ...')
        #     continue
        #if ('c20cage' in fn) or ('omcb' in fn):
         #   print('skip it')
          #  continue
        valid = gen_train_data_E([fn], a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.valid.bond_priority, basis=yml.basis, grid_level=yml.grid_level, base_method=yml.base_method, target_method=yml.target_method, e_T2=True, train=False, complex_geo=yml.train.complex_geo)[0]
        if count==0:
            model_e = ModelE(feature_num=len(valid['x'][0]), device=yml.device)
            model_e.load(yml.valid.model)
            model_rho = ModelD(feature_num=len(valid['x'][0]), device=yml.device)
            model_rho.load(yml.valid.model.replace('_e.pt', '_d.pt'))
            model_f = ModelF(feature_num=len(valid['x'][0]), device=yml.device)
            model_f.load(yml.valid.model.replace('_e.pt', '_f.pt'))
            
            # valid['x'] = torch.tensor(valid['x']).to(model_e.device)
            # inference_e = model_e.forward(valid['x']).detach().cpu().numpy()
        valid['x'] = torch.tensor(valid['x']).to(model_e.device)
        with torch.no_grad():
            if i == 1:
                inference_e = model_e.forward(valid['x']).detach().cpu().numpy()
                i += 1
            deltaT = time.time()
            inference_e = model_e.forward(valid['x']).detach().cpu().numpy()
            deltaT = time.time() - deltaT
            batch_rho_p = model_rho.forward(valid['x']).detach().cpu().numpy()
            batch_img_rho = model_f.forward(valid['x']).detach().cpu().numpy()
        # rec.append(valid['e_xc_base']*valid['gw'])
        rho = np.where(batch_rho_p*valid['rho_base']>0, batch_rho_p*valid['rho_base'], 0.)
        diff_rho = rho - valid['rho_base']
        pos_idx = diff_rho>0
        # print('pos', np.sum((diff_rho*valid['gw'])[pos_idx]), 'neg', np.sum((diff_rho*valid['gw'])[~pos_idx]))
        ratio = -np.sum((diff_rho*valid['gw'])[pos_idx])/np.sum((diff_rho*valid['gw'])[~pos_idx])
        # print(ratio)
        diff_rho[~pos_idx] *= ratio
        rho = valid['rho_base'] + diff_rho
        
        # rho = batch_rho_p*valid['rho_base']
        
        d3 = np.sum((valid['atoms_coords'][:,None] - valid['gc'])**2, axis=-1)**1.5
        r = (valid['atoms_coords'][:,None] - valid['gc']).transpose(2,0,1)
        # r = (pkl['atoms_coords'][:,None] - pkl['gc']).transpose(1,2).transpose(0,1)
        force = valid['F_base'] + (-r/d3*np.einsum('i,j,j->ij', valid['atoms_charges'], batch_img_rho*valid['rho_base'], valid['gw'])).sum(axis=2).T
        # force -= force.mean(axis=0)
        error_F_base = np.sum((valid['F_base'] - valid['F_target'])**2)**.5
        error_F_model = np.sum((force - valid['F_target'])**2)**.5
        
        df.loc[count] = [*os.path.basename(fn).split('_')[:2], valid['E_base'], valid['E_base']+np.sum(inference_e*valid['rho_base']*valid['gw']), valid['E_target'], (valid['E_base']-valid['E_target'])*627, (valid['E_base']+np.sum(inference_e*valid['rho_base']*valid['gw'])-valid['E_target'])*627,
                         valid['F_base'], force, valid['F_target'],
                         valid['dipole_base'], cal_dipole(rho, valid['gc'], valid['gw'], valid['atoms_charges'], valid['atoms_coords']), valid['dipole_target'],
                         error_F_base, error_F_model,
                         np.sum(rho*valid['gw'])-np.sum(valid['rho_target']*valid['gw']),
                         valid['timeks'], valid['timefeature'], deltaT, valid['npoints'], valid['nbasis']]
        df.to_csv(yml.valid.csv_name, index=False)
    df[f'Error dipole {yml.base_method}'] = (df[f'Dipole {yml.base_method}']-df[f'Dipole {yml.target_method}']).apply(lambda a: np.sum(a**2)**.5)
    df['Error dipole model'] = (df['Dipole model']-df[f'Dipole {yml.target_method}']).apply(lambda a: np.sum(a**2)**.5)
    ma = []
    for column in df:
        if df[column].dtype == np.float64:
            ma.append(np.abs(df[column].to_numpy()).mean())
        else:
            ma.append(np.NaN)
    ma[0] = 'MA'
    df.loc[count+1] = ma
    df.to_csv(yml.valid.csv_name, index=False)
    # with open('exc_b3.pkl', 'wb') as f:
    #     pk.dump(rec, f)
