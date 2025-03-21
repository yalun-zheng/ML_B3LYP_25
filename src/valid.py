import pickle as pk
import numpy as np
from build import Tree
from pyscf import gto, dft, cc, scf
import matplotlib.pyplot as plt
from einsum import dm2rho, dm2rho01, dm2eT2, dm2eJ, dm2eK, cal_dipole, cal_I, dm2rho01_sep
from train import cal_mu, get_feature, gen_train_data
import time
import os
import pandas as pd

if __name__ == "__main__":
    from cfg import get_cfg
    import sys
    yml = get_cfg(sys.argv[1])
    if not yml.valid.csv_name:
        yml.valid.csv_name = os.path.basename(yml.valid.model)[:-4]+'_tree_valid.csv'
    df = pd.DataFrame(columns=['Molecule', 'Bond length', 
                               'MAE e b3lypg', 'MAE e model', 'Error Exc b3lypg in kcal/mol', 'Error Exc model in kcal/mol',
                                                          f'Dipole {yml.base_method}', 'Dipole model', f'Dipole {yml.target_method}', 'Error N model'])
    assert os.path.exists(yml.valid.model), f'no {yml.valid.model}, pls assign an existing model.'
    print('loading', yml.valid.model)
    with open(yml.valid.model, 'rb') as f:
        tree_e, xgb_e, tree_rho, xgb_rho, scale = pk.load(f)
    validset = []
    
    for molname, bond_lengths in yml.valid.geo.items():
        for bond_length in bond_lengths:
            if type(bond_length)==list:
                for bd in np.linspace(*bond_length):
                    if '*' in molname:
                        repo = os.path.dirname(molname)
                        for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                            if mn.endswith('.xyz'):
                                validset.append(f'{repo}/{mn[:-4]}_{bd:.4f}_{yml.valid.bond_priority}_{yml.basis}_e_train_valid.pkl')
                    else:
                        validset.append(f'{molname}_{bd:.4f}_{yml.valid.bond_priority}_{yml.basis}_e_train_valid.pkl')
            else:
                if '*' in molname:
                    repo = os.path.dirname(molname)
                    for mn in os.listdir(os.path.join(yml.homepath, f'data/xyz/{repo}')):
                        if mn.endswith('.xyz'):
                            validset.append(f'{repo}/{mn[:-4]}_{bond_length:.4f}_{yml.valid.bond_priority}_{yml.basis}_e_train_valid.pkl')
                else:
                    validset.append(f'{molname}_{bond_length:.4f}_{yml.valid.bond_priority}_{yml.basis}_e_train_valid.pkl')
    validset = set([os.path.join(yml.homepath, f'data/pkl/{f}') for f in validset])
    
    
    for count, fn in enumerate(sorted(validset)):
        # if os.path.basename(fn).split('_')[0] in yml.train.complex_geo:
        #     print(fn, 'is too complex, not to be validated ...')
        #     continue
        valid = gen_train_data([fn], a=yml.edge_length, n_samples=yml.n_samples+1, priority=yml.valid.bond_priority, basis=yml.basis, grid_level=yml.grid_level, base_method=yml.base_method, target_method=yml.target_method, e_T2=True, train=False)[0]
        print(f"x shape, {valid['x'].shape}, e shape, {valid['e'].shape}")
        inference_e = np.array([tree_e.inference(x.reshape(-1)) for x in (valid['x']/scale)])
        # reg_e = xgb_e.predict((valid['x']/scale).reshape(len(valid['x']), -1))
        # print('MAE e_corr:', np.abs(valid['e']-inference_e).mean())
        # print('MAE xgb e_corr:', np.abs(valid['e']-reg_e).mean())
        # print('Error Tree in kcal/mol:', np.sum((valid['e']-inference_e)*valid['rho_base']*valid['gw'])*627)
        # print('Error xgb in kcal/mol:', np.sum((valid['e']-reg_e)*valid['rho_base']*valid['gw'])*627)
        # print(f'Error {yml.base_method} in kcal/mol:', np.sum(valid['e']*valid['rho_base']*valid['gw'])*627)
        mae_b3 = np.mean(np.abs((valid['e_base']-valid['e'])*valid['rho_base']))
        mae_model = np.mean(np.abs((inference_e-valid['e'])*valid['rho_base']))
        integral_b3 = np.sum((valid['e_base']-valid['e'])*valid['rho_base']*valid['gw'])*627
        integral_model = np.sum((inference_e-valid['e'])*valid['rho_base']*valid['gw'])*627
        
        inference_rho = np.array([tree_rho.inference(x.reshape(-1)) for x in (valid['x']/scale)])**3
        reg_rho = xgb_rho.predict((valid['x']/scale).reshape(len(valid['x']), -1))-valid['rho_base']
        neg_id = inference_rho<0
        neg_charge = np.sum(inference_rho[neg_id]*valid['gw'][neg_id])
        pos_charge = np.sum(inference_rho[~neg_id]*valid['gw'][~neg_id])
        ratio = abs(pos_charge/neg_charge)
        if ratio < 1:
            inference_rho[neg_id] *= ratio
        else:
            inference_rho[~neg_id] /= ratio
        rho = inference_rho+valid['rho_base']
        

        
        
        df.loc[count] = [*os.path.basename(fn).split('_')[:2],
                         mae_b3, mae_model, integral_b3, integral_model,
                         valid['dipole_base'], cal_dipole(rho, valid['gc'], valid['gw'], valid['atoms_charges'], valid['atoms_coords']), valid['dipole_target'],
                         np.sum(rho*valid['gw'])-np.sum(valid['rho_target']*valid['gw'])]
        df.to_csv(yml.valid.csv_name, index=False)
        # if 'F' in yml.valid.output:
        #     mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in 
        #         zip(valid['atoms_charges'], valid['atoms_coords'])], basis=yml.basis, 
        #                 spin=valid['spin'], charge=valid['charge'], unit='bohr')
        #     mol.verbose = 0
        #     if yml.base_method == 'b3lypg':
        #         ks = dft.KS(mol)
        #         ks.xc = 'b3lypg'
        #         ks.grids.level = yml.grid_level
        #         ks.kernel()
        #         F_base = -np.array(ks.nuc_grad_method().kernel())
        #     if yml.target_method == 'ccsd':
        #         mf = scf.HF(mol)
        #         mf.kernel()
        #         ccsd = cc.CCSD(mf)
        #         ccsd.kernel()
        #         F_target = -ccsd.nuc_grad_method().kernel()
        #     print(f"Force base\n{F_base}\nForce target\n{F_target}")
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



