import pickle as pk
import numpy as np
from build import Tree
from pyscf import gto, dft, cc, scf
import matplotlib.pyplot as plt
from einsum import dm2rho, dm2rho01, dm2eT2, dm2eJ, dm2eK, cal_dipole, cal_I, dm2rho01_sep
from train_fcn_b3_mgga import cal_mu, get_feature, gen_train_data_E
import time
import os
import torch
from opt_einsum import contract

if __name__ == "__main__":
    from model import ModelE_MGGA, ModelE_post_MGGA
    from cfg import get_cfg
    import sys
    import pandas as pd
    yml = get_cfg(sys.argv[1])
    if not yml.valid.csv_name:
        yml.valid.csv_name = os.path.basename(yml.valid.model)[:-3]+'_validb3.csv'
    df = pd.DataFrame(columns=['Molecule', 'Bond length', f'E {yml.base_method}', 'E scf', 'E model', f'E {yml.target_method}', f'Error E {yml.base_method} in kcal/mol', 'Error E scf in kcal/mol', 'Error E model in kcal/mol',
                                                          f'Dipole {yml.base_method}', 'Dipole model', f'Dipole {yml.target_method}', f'Error Force {yml.base_method}', 'Error Force model', f'Error |drho*w| {yml.base_method}', 'Error |drho*w| model', f'Error |drho| {yml.base_method}', 'Error |drho| model'])
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
    for count, fn in enumerate(sorted(validset)):
        # if os.path.basename(fn).split('_')[0] in yml.train.complex_geo:
        #     print(fn, 'is too complex, not to be validated ...')
        #     continue
        valid = gen_train_data_E([fn], a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.valid.bond_priority, basis=yml.basis, grid_level=yml.grid_level, target_method=yml.target_method, e_T2=True, train=False, complex_geo=yml.train.complex_geo)[0]
        if count==0:
            model_e = ModelE_MGGA(feature_num=4, device=yml.device)
            model_e.load(yml.valid.model)
            
            model_post = ModelE_post_MGGA(feature_num=4, device=yml.device)
            model_post.load(yml.valid.model.replace('b3pre', '_post2b3pre'))
        # valid['x'] = torch.tensor(valid['x']).to(model_e.device)
        # with torch.no_grad():
        #     inference_e = model_e.forward(torch.tensor(valid['x']).to(model_e.device))
        #     inference_e = inference_e.detach().cpu().numpy()
        # print('E_model', valid['E_base']+np.sum(inference_e*valid['rho_base']*valid['gw']))
        # print('model add', np.sum(inference_e*valid['rho_base']*valid['gw']), np.sum(inference_e*valid['rho_base']*valid['gw'])*627)
        # valid['x'] = valid['x'].detach().cpu().numpy()
        # rho = np.vstack((*valid['x'].T[0:4], np.zeros(len(valid['x'])), valid['x'].T[4])), np.vstack((*valid['x'].T[5:9], np.zeros(len(valid['x'])), valid['x'].T[9]))
        # print(model_e.eval_xc(rho=rho))
        
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in 
            zip(valid['atoms_charges'], valid['atoms_coords'])], basis=yml.basis, 
                    spin=valid['spin'], charge=valid['charge'], unit='bohr')
        ni = dft.numint.NumInt()
        valid['phi01'] = ni.eval_ao(mol, valid['gc'], deriv=1)
        res = model_e.scf(valid, yml.basis, grid_level=3, hyb=0.2, xc='b3lypg')
        dm = res['dm_scf']
        if len(res['dm_scf']) != 2:
            dm = 0.5*dm, 0.5*dm
            res['rho01_scf'] = 0.5*res['rho01_scf'], 0.5*res['rho01_scf']
        tau = 0.5*contract('ij, dri, drj->r', dm[0], valid['phi01'][1:], valid['phi01'][1:]), 0.5*contract('ij, dri, drj->r', dm[1], valid['phi01'][1:], valid['phi01'][1:])
        x = np.array([*(res['rho01_scf'][0]), np.zeros_like(tau[0]), tau[0]]), np.array([*(res['rho01_scf'][1]), np.zeros_like(tau[1]), tau[1]])
        e_post = model_post.forward(x).detach().cpu().numpy()
        E_post = np.sum(e_post*res['rho_scf']*valid['gw'])
        error_F_base = np.sum((valid['F_base'] - valid['F_target'])**2)**.5
        error_F_model = np.sum((res['F_scf'] - valid['F_target'])**2)**.5
        df.loc[count] = [*os.path.basename(fn).split('_')[:2], valid['E_base'], res['E_scf'], res['E_scf']+E_post, valid['E_target'], (valid['E_base']-valid['E_target'])*627, (res['E_scf']-valid['E_target'])*627, (res['E_scf']+E_post-valid['E_target'])*627,
                         valid['dipole_base'], cal_dipole(res['rho_scf'], valid['gc'], valid['gw'], valid['atoms_charges'], valid['atoms_coords']), valid['dipole_target'],
                         error_F_base, error_F_model,
                         np.sum(np.abs((valid['rho_base']-valid['rho_target'])*valid['gw'])),
                         np.sum(np.abs((res['rho_scf']-valid['rho_target'])*valid['gw'])),
                         np.sum(np.abs(valid['rho_base']-valid['rho_target'])),
                         np.sum(np.abs(res['rho_scf']-valid['rho_target'])),
                         ]
        df.to_csv(yml.valid.csv_name, index=False)
        
    df[f'Error dipole {yml.base_method}'] = (df[f'Dipole {yml.base_method}']-df[f'Dipole {yml.target_method}']).apply(lambda a: np.sum(a**2)**.5)
    df['Error dipole model'] = (df['Dipole model']-df[f'Dipole {yml.target_method}']).apply(lambda a: np.sum(a**2)**.5)
    df.to_csv(yml.valid.csv_name, index=False)