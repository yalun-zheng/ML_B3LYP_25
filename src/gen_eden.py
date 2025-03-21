import os
import pickle as pk
import numpy as np
from pyscf import scf, gto, dft, cc
from pyscf.dft.numint import NumInt
from opt_einsum import contract
import time
import dftd3.pyscf as d3
from einsum import dm2eEE, dm2eJ, dm2eT2, dm2eK, eval_vh, dm2eT, dm2rho, dm2rho01, cal_I, cal_dipole, dm2E1e, dm2J, dm2K
from pyscf.cc import ccsd_t_rdm_slow, uccsd_t_rdm
from pyscf.grad.uccsd_t import Gradients as guccsdt
from pyscf.grad.ccsd_t import Gradients as gccsdt
from pyscf.grad.uccsd import Gradients as guccsd
from pyscf.grad.ccsd import Gradients as gccsd
from pyscf.grad.uks import Gradients as guks
from pyscf.grad.rks import Gradients as gks
from pyscf.grad.rhf import Gradients as ghf
from pyscf.grad.uhf import Gradients as guhf

atom_charge2spin = {1:1, 2:0, 3:1, 4:0, 5:1, 6:2, 7:3, 8:2, 9:1, 10:0, 11:1, 12:0, 13:1, 14:2, 15:3, 16:2, 17:1, 18:0, 19:1, 20:0}
atom_charge2energy = {}

def read_xyz_to_dict(xyz_fp):
    with open(xyz_fp, 'r') as xyz_f:
        lines = xyz_f.readlines()
        xyz_dict = {'atom_list':[], 'xyz':[]}
    for line in lines[2:]:
        atom_name, x, y, z = line.split()
        x, y, z = float(x), float(y), float(z)
        xyz_dict['atom_list'] += [atom_name]
        xyz_dict['xyz'] += [[x, y, z]]
    xyz_dict['xyz'] = np.array(xyz_dict['xyz'])
    charge, spin = lines[1].split()
    return xyz_dict, int(charge), int(spin)

def xyz_dict_to_str(xyz_dict):
    xyz_str = ""
    for i, atom_name in enumerate(xyz_dict['atom_list']):
        x, y, z = xyz_dict['xyz'][i]
        xyz_str += f"{atom_name} {x} {y} {z};"
    return xyz_str

def numerical_vxc(xs, ys):
    '''
    vrho*(rho_plusx-rho) + vgamma*(gamma_plusx-gamma) + vtau*(tau_plusx-tau)  = exc_plusx - exc
    vrho*(rho_plusy-rho) + vgamma*(gamma_plusy-gamma) + vtau*(tau_plusy-tau) = exc_plusy - exc
    vrho*(rho_plusz-rho) + vgamma*(gamma_plusz-gamma) + vtau*(tau_plusz-tau) = exc_plusz - exc
    returns a (3, N) matrix, 3 rows are vrho, vgamma, vtau ..., dependent on the features of xs
    '''
    res = []
    invalid_count = 0
    for i in range(len(ys)):
        x = xs[:1,i].T
        y = ys[i]
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression(fit_intercept=False)
        lr.fit(x, y)
        if lr.rank_ < x.shape[1]:
            invalid_count +=1
            res.append(lr.coef_*0.)
        else:
            res.append(lr.coef_)
    print('invalid_count', invalid_count, 'over', len(ys))
    return np.array(res).T

def gen_pkl_e(xyz_dict, charge, spin, basis, grid_level, delta=5e-3):
    t0 = time.time()
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
    ni = NumInt()
    phi012 = ni.eval_ao(mol, data['gc'], deriv=2)
    data['phi01'] = phi012[:4]
    phi2 = phi012[[4,7,9],:]
    mf = scf.HF(mol)
    mf.kernel()
    data['S'] = mol.intor('int1e_ovlp_sph')
    # data['itg_2e'] = mol.intor('int2e_sph')
    data['T'] = mol.intor('cint1e_kin_sph')
    data['V_ext'] = mol.intor('cint1e_nuc_sph')
    data['n_elec'] = mol.nelectron
    data['E_N'] = mol.energy_nuc()
    nu = mol.intor('int1e_grids_sph', grids=data['gc'])
    df.kernel()
    data['E_b3'] = df.e_tot
    data['dm_b3'] = df.make_rdm1()
    # df.xc = 'pbe'
    # df.kernel()
    # data['E_pbe'] = df.e_tot
    # data['dm_pbe'] = df.make_rdm1()
    data['E_hf'] = mf.e_tot
    data['dm_hf'] = mf.make_rdm1()
    C = mf.mo_coeff
    # CCSD calculations
    ccsd = cc.CCSD(mf)
    ccsd.kernel()
    data['dm_ccsd'] = ccsd.make_rdm1(ao_repr=True)
    rdm2_ao_ccsd = ccsd.make_rdm2(ao_repr=True)
    data['converged'] = ccsd.converged
    assert data['converged']
    data['E_ccsd'] = ccsd.e_tot
    data['E_ccsdt'] = data['E_ccsd'] + ccsd.ccsd_t()
    t1, t2 = ccsd.t1, ccsd.t2
    l1, l2 = ccsd.l1, ccsd.l2
    eris = ccsd.ao2mo()
    if spin:
        data['dm_ccsdt'] = uccsd_t_rdm.make_rdm1(ccsd, t1, t2, l1, l2, eris, True)
        rdm2_ao_ccsdt = uccsd_t_rdm.make_rdm2_ao(ccsd, t1, t2, l1, l2, eris)
        data['rho01_b3'] = dm2rho01(data['dm_b3'][0], data['phi01']), dm2rho01(data['dm_b3'][1], data['phi01'])
        data['rho01_hf'] = dm2rho01(data['dm_hf'][0], data['phi01']), dm2rho01(data['dm_hf'][1], data['phi01'])
        # data['rho01_pbe'] = dm2rho01(data['dm_pbe'][0], data['phi01']), dm2rho01(data['dm_pbe'][1], data['phi01'])
        data['rho01_ccsd'] = dm2rho01(data['dm_ccsd'][0], data['phi01']), dm2rho01(data['dm_ccsd'][1], data['phi01'])
        data['rho01_ccsdt'] = dm2rho01(data['dm_ccsdt'][0], data['phi01']), dm2rho01(data['dm_ccsdt'][1], data['phi01'])
        data['rho_ccsd'] = data['rho01_ccsd'][0][0] + data['rho01_ccsd'][1][0]
        data['rho_ccsdt'] = data['rho01_ccsdt'][0][0] + data['rho01_ccsdt'][1][0]
        data['rho_b3'] = data['rho01_b3'][0][0] + data['rho01_b3'][1][0]
        data['rho_hf'] = data['rho01_hf'][0][0] + data['rho01_hf'][1][0]
        # data['rho_pbe'] = data['rho01_pbe'][0][0] + data['rho01_pbe'][1][0]
        for method in ['ccsd', 'ccsdt', 'b3', 'hf']:
            # data[f'J_{method}'] = dm2J(data.get(f'dm_{method}'), data['itg_2e'])
            # data[f'K_{method}'] = dm2K(data.get(f'dm_{method}')[0], data['itg_2e']), dm2K(data.get(f'dm_{method}')[1], data['itg_2e'])
            data[f'E_T_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['T'])
            data[f'E_ext_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['V_ext'])
            # data[f'E_J_{method}'] = 0.5 * dm2E1e(data.get(f'dm_{method}'), data[f'J_{method}'])
        eEE_ccsd = dm2eEE(rdm2_ao_ccsd[0]+rdm2_ao_ccsd[2]+rdm2_ao_ccsd[1]+rdm2_ao_ccsd[1].transpose(2,3,0,1), data['phi01'][0], nu)
        eEE_ccsdt = dm2eEE(rdm2_ao_ccsdt[0]+rdm2_ao_ccsdt[2]+rdm2_ao_ccsdt[1]+rdm2_ao_ccsdt[1].transpose(2,3,0,1), data['phi01'][0], nu)
        # eEE_ccsd_sep = dm2eEE(rdm2_ao_ccsd[0], data['phi01'][0], nu), dm2eEE(rdm2_ao_ccsd[2], data['phi01'][0], nu), dm2eEE(rdm2_ao_ccsd[1]+rdm2_ao_ccsd[1].transpose(2,3,0,1), data['phi01'][0], nu)
        # eEE_ccsdt_sep = dm2eEE(rdm2_ao_ccsdt[0], data['phi01'][0], nu), dm2eEE(rdm2_ao_ccsdt[2], data['phi01'][0], nu), dm2eEE(rdm2_ao_ccsdt[1]+rdm2_ao_ccsdt[1].transpose(2,3,0,1), data['phi01'][0], nu)
        eJ_ccsd = dm2eJ(data['dm_ccsd'][0]+data['dm_ccsd'][1], data['phi01'][0], nu)
        data['e_J_ccsd'] = eJ_ccsd
        # eJ_ccsd_sep = dm2eJ(data['dm_ccsd'], data['phi01'][0], nu)
        eJ_ccsdt = dm2eJ(data['dm_ccsdt'][0]+data['dm_ccsdt'][1], data['phi01'][0], nu)
        data['e_J_ccsdt'] = eJ_ccsdt
        # eJ_ccsdt_sep = dm2eJ(data['dm_ccsdt'], data['phi01'][0], nu)
        
        data['e_T_ccsd2'] = .5*dm2eT2(data['dm_ccsd'][0]+data['dm_ccsd'][1], data['phi01'][0], phi2)
        data['e_T_ccsdt2'] = .5*dm2eT2(data['dm_ccsdt'][0]+data['dm_ccsdt'][1], data['phi01'][0], phi2)
        data['e_T_ccsd'] = .5*dm2eT(data['dm_ccsd'][0]+data['dm_ccsd'][1], data['phi01'][1:])
        data['e_T_ccsdt'] = .5*dm2eT(data['dm_ccsdt'][0]+data['dm_ccsdt'][1], data['phi01'][1:])
        # data['e_T_ccsd2_sep'] = dm2eT2(data['dm_ccsd'], data['phi01'][0], phi2)
        # data['e_T_ccsdt2_sep'] = dm2eT2(data['dm_ccsdt'], data['phi01'][0], phi2)
        # data['e_T_ccsd2_sep'] = data['e_T_ccsd2_sep'][0]*0.5, data['e_T_ccsd2_sep'][1]*0.5
        # data['e_T_ccsdt2_sep'] = data['e_T_ccsdt2_sep'][0]*0.5, data['e_T_ccsdt2_sep'][1]*0.5
        # data['e_T_ccsd_sep'] = dm2eT(data['dm_ccsd'], data['phi01'][1:])
        # data['e_T_ccsdt_sep'] = dm2eT(data['dm_ccsdt'], data['phi01'][1:])
        # data['e_T_ccsd_sep'] = data['e_T_ccsd_sep'][0]*0.5, data['e_T_ccsd_sep'][1]*0.5
        # data['e_T_ccsdt_sep'] = data['e_T_ccsdt_sep'][0]*0.5, data['e_T_ccsdt_sep'][1]*0.5
        data['e_xc_ccsd'] = 0.5 * ( eEE_ccsd - eJ_ccsd )
        data['e_xc_ccsdt'] = 0.5 * ( eEE_ccsdt - eJ_ccsdt )
        # data['e_xc_ccsd_sep'] = 0.5 * ( eEE_ccsd_sep[0] - eJ_ccsd_sep[0]), 0.5 * ( eEE_ccsd_sep[1] - eJ_ccsd_sep[1]), 0.5 * ( eEE_ccsd_sep[2] - eJ_ccsd_sep[2])
        # data['e_xc_ccsdt_sep'] = 0.5 * ( eEE_ccsdt_sep[0] - eJ_ccsdt_sep[0]), 0.5 * ( eEE_ccsdt_sep[1] - eJ_ccsdt_sep[1]), 0.5 * ( eEE_ccsdt_sep[2] - eJ_ccsdt_sep[2])
    else:
        # data['dm_ccsdt'] = ccsd_t_rdm_slow.make_rdm1(ccsd, t1, t2, l1, l2, eris, True)
        # rdm2_ao_ccsdt = ccsd_t_rdm_slow.make_rdm2_ao(ccsd, t1, t2, l1, l2, eris)
        data['rho01_b3'] = dm2rho01(data['dm_b3'], data['phi01'])
        data['rho01_hf'] = dm2rho01(data['dm_hf'], data['phi01'])
        # data['rho01_pbe'] = dm2rho01(data['dm_pbe'], data['phi01'])
        data['rho01_ccsd'] = dm2rho01(data['dm_ccsd'], data['phi01'])
        # data['rho01_ccsdt'] = dm2rho01(data['dm_ccsdt'], data['phi01'])
        data['rho_ccsd'] = data['rho01_ccsd'][0]
        # data['rho_ccsdt'] = data['rho01_ccsdt'][0]
        data['rho_b3'] = data['rho01_b3'][0]
        data['rho_hf'] = data['rho01_hf'][0]
        # data['rho_pbe'] = data['rho01_pbe'][0]
        for method in ['ccsd', 'b3', 'hf']:
            # data[f'J_{method}'] = dm2J(data.get(f'dm_{method}'), data['itg_2e'])
            # data[f'K_{method}'] = dm2K(data.get(f'dm_{method}'), data['itg_2e'])
            data[f'E_T_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['T'])
            data[f'E_ext_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['V_ext'])
            # data[f'E_J_{method}'] = 0.5 * dm2E1e(data.get(f'dm_{method}'), data[f'J_{method}'])
        eEE_ccsd = dm2eEE(rdm2_ao_ccsd, data['phi01'][0], nu)
        # eEE_ccsdt = dm2eEE(rdm2_ao_ccsdt, data['phi01'][0], nu)
        eJ_ccsd = dm2eJ(data['dm_ccsd'], data['phi01'][0], nu)
        # eJ_ccsdt = dm2eJ(data['dm_ccsdt'], data['phi01'][0], nu)
        data['e_T_ccsd2'] = .5*dm2eT2(data['dm_ccsd'], data['phi01'][0], phi2)
        # data['e_T_ccsdt2'] = .5*dm2eT2(data['dm_ccsdt'], data['phi01'][0], phi2)
        data['e_T_ccsd'] = .5*dm2eT(data['dm_ccsd'], data['phi01'][1:])
        # data['e_T_ccsdt'] = .5*dm2eT(data['dm_ccsdt'], data['phi01'][1:])
        # data['e_T_ccsd_sep'] = data['e_T_ccsd']*.5, data['e_T_ccsd']*.5
        # data['e_T_ccsdt_sep'] = data['e_T_ccsdt']*.5, data['e_T_ccsdt']*.5
        # data['e_T_ccsd2_sep'] = data['e_T_ccsd2']*.5, data['e_T_ccsd2']*.5
        # data['e_T_ccsdt2_sep'] = data['e_T_ccsdt2']*.5, data['e_T_ccsdt2']*.5
        data['e_xc_ccsd'] = 0.5 * ( eEE_ccsd - eJ_ccsd )
        # data['e_xc_ccsdt'] = 0.5 * ( eEE_ccsdt - eJ_ccsdt )
    # data['I_b3'] = cal_I(data['rho_ccsdt'], data['rho_b3'], data['gw'])
    # data['I_hf'] = cal_I(data['rho_ccsdt'], data['rho_hf'], data['gw'])
    # data['I_pbe'] = cal_I(data['rho_ccsdt'], data['rho_pbe'], data['gw'])
    data['dipole_ccsd'] = cal_dipole(data['rho_ccsd'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    # data['dipole_ccsdt'] = cal_dipole(data['rho_ccsdt'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    data['dipole_b3'] = cal_dipole(data['rho_b3'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    data['dipole_hf'] = cal_dipole(data['rho_hf'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    # data['dipole_pbe'] = cal_dipole(data['rho_pbe'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    data['d3bj'] = d3.DFTD3Dispersion(mol, xc="b3lyp", version="d3bj").kernel()[0]
    data['time'] = time.time() - t0
    return data

def gen_pkl(xyz_dict, charge, spin, basis, grid_level, dm_ccsdt=False):
    t0 = time.time()
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
    ni = NumInt()
    phi012 = ni.eval_ao(mol, data['gc'], deriv=2)
    data['phi01'] = phi012[:4]
    phi2 = phi012[[4,7,9],:]
    mf = scf.HF(mol)
    mf.kernel()
    data['S'] = mol.intor('int1e_ovlp_sph')
    # data['itg_2e'] = mol.intor('int2e_sph')
    data['T'] = mol.intor('cint1e_kin_sph')
    data['V_ext'] = mol.intor('cint1e_nuc_sph')
    data['n_elec'] = mol.nelectron
    data['E_N'] = mol.energy_nuc()
    nu = mol.intor('int1e_grids_sph', grids=data['gc'])
    df.kernel()
    data['E_b3'] = df.e_tot
    data['dm_b3'] = df.make_rdm1()
    # df.xc = 'pbe'
    # df.kernel()
    # data['E_pbe'] = df.e_tot
    # data['dm_pbe'] = df.make_rdm1()
    data['E_hf'] = mf.e_tot
    data['dm_hf'] = mf.make_rdm1()
    C = mf.mo_coeff
    # CCSD calculations
    ccsd = cc.CCSD(mf)
    ccsd.kernel()
    data['dm_ccsd'] = ccsd.make_rdm1(ao_repr=True)
    data['converged'] = ccsd.converged
    assert data['converged']
    data['E_ccsd'] = ccsd.e_tot
    data['E_ccsdt'] = data['E_ccsd'] + ccsd.ccsd_t()
    eris = ccsd.ao2mo()
    if spin:
        data['F_b3'] = -guks(df).kernel()
        data['F_hf'] = -guhf(mf).kernel()
        if data['n_elec'] == 1:
            data['F_ccsd'] = data['F_hf']
            data['F_ccsdt'] = data['F_hf']
        else:
            data['F_ccsd'] = -guccsd(ccsd).kernel(ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris=eris)
        if dm_ccsdt:
            data['F_ccsdt'] = -guccsdt(ccsd).kernel(ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris=eris)
            data['dm_ccsdt'] = uccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris, True)
            data['rho01_ccsdt'] = dm2rho01(data['dm_ccsdt'][0], data['phi01']), dm2rho01(data['dm_ccsdt'][1], data['phi01'])
            data['rho_ccsdt'] = data['rho01_ccsdt'][0][0] + data['rho01_ccsdt'][1][0]
        data['rho01_b3'] = dm2rho01(data['dm_b3'][0], data['phi01']), dm2rho01(data['dm_b3'][1], data['phi01'])
        data['rho01_hf'] = dm2rho01(data['dm_hf'][0], data['phi01']), dm2rho01(data['dm_hf'][1], data['phi01'])
        # data['rho01_pbe'] = dm2rho01(data['dm_pbe'][0], data['phi01']), dm2rho01(data['dm_pbe'][1], data['phi01'])
        data['rho01_ccsd'] = dm2rho01(data['dm_ccsd'][0], data['phi01']), dm2rho01(data['dm_ccsd'][1], data['phi01'])
        data['rho_ccsd'] = data['rho01_ccsd'][0][0] + data['rho01_ccsd'][1][0]
        data['rho_b3'] = data['rho01_b3'][0][0] + data['rho01_b3'][1][0]
        data['rho_hf'] = data['rho01_hf'][0][0] + data['rho01_hf'][1][0]
        # data['rho_pbe'] = data['rho01_pbe'][0][0] + data['rho01_pbe'][1][0]
        for method in (['ccsd', 'ccsdt', 'b3', 'hf'] if dm_ccsdt else ['ccsd', 'b3', 'hf']):
            # data[f'J_{method}'] = dm2J(data.get(f'dm_{method}'), data['itg_2e'])
            # data[f'K_{method}'] = dm2K(data.get(f'dm_{method}')[0], data['itg_2e']), dm2K(data.get(f'dm_{method}')[1], data['itg_2e'])
            data[f'E_T_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['T'])
            data[f'E_ext_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['V_ext'])
            # data[f'E_J_{method}'] = 0.5 * dm2E1e(data.get(f'dm_{method}'), data[f'J_{method}'])
    else:
        data['F_b3'] = -gks(df).kernel()
        data['F_hf'] = -ghf(mf).kernel()
        data['F_ccsd'] = -gccsd(ccsd).kernel(ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris=eris)
        if dm_ccsdt:
            data['F_ccsdt'] = -gccsdt(ccsd).kernel(ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris=eris)
            data['dm_ccsdt'] = ccsd_t_rdm_slow.make_rdm1(ccsd, ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris, True)
            data['rho01_ccsdt'] = dm2rho01(data['dm_ccsdt'], data['phi01'])
            data['rho_ccsdt'] = data['rho01_ccsdt'][0]
        data['rho01_b3'] = dm2rho01(data['dm_b3'], data['phi01'])
        data['rho01_hf'] = dm2rho01(data['dm_hf'], data['phi01'])
        # data['rho01_pbe'] = dm2rho01(data['dm_pbe'], data['phi01'])
        data['rho01_ccsd'] = dm2rho01(data['dm_ccsd'], data['phi01'])
        
        data['rho_ccsd'] = data['rho01_ccsd'][0]
        
        data['rho_b3'] = data['rho01_b3'][0]
        data['rho_hf'] = data['rho01_hf'][0]
        # data['rho_pbe'] = data['rho01_pbe'][0]
        for method in (['ccsd', 'ccsdt', 'b3', 'hf'] if dm_ccsdt else ['ccsd', 'b3', 'hf']):
            # data[f'J_{method}'] = dm2J(data.get(f'dm_{method}'), data['itg_2e'])
            # data[f'K_{method}'] = dm2K(data.get(f'dm_{method}'), data['itg_2e'])
            data[f'E_T_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['T'])
            data[f'E_ext_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['V_ext'])
            # data[f'E_J_{method}'] = 0.5 * dm2E1e(data.get(f'dm_{method}'), data[f'J_{method}'])
    # data['I_b3'] = cal_I(data['rho_ccsd'], data['rho_b3'], data['gw'])
    # data['I_hf'] = cal_I(data['rho_ccsd'], data['rho_hf'], data['gw'])
    # data['I_pbe'] = cal_I(data['rho_ccsd'], data['rho_pbe'], data['gw'])
    data['dipole_ccsd'] = cal_dipole(data['rho_ccsd'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    if dm_ccsdt:
        data['dipole_ccsdt'] = cal_dipole(data['rho_ccsdt'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    data['dipole_b3'] = cal_dipole(data['rho_b3'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    data['dipole_hf'] = cal_dipole(data['rho_hf'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    # data['dipole_pbe'] = cal_dipole(data['rho_pbe'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    data['d3bj'] = d3.DFTD3Dispersion(mol, xc="b3lyp", version="d3bj").kernel()[0]
    data['time'] = time.time() - t0
    return data

def gen_test_pkl(xyz_dict, charge, spin, basis, grid_level, F_ccsdt=False):
    t0 = time.time()
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
    ni = NumInt()
    phi012 = ni.eval_ao(mol, data['gc'], deriv=2)
    data['phi01'] = phi012[:4]
    mf = scf.HF(mol)
    mf.kernel()
    data['n_elec'] = mol.nelectron
    data['E_N'] = mol.energy_nuc()
    df.kernel()
    data['E_b3'] = df.e_tot
    data['dm_b3'] = df.make_rdm1()
    data['E_hf'] = mf.e_tot
    data['dm_hf'] = mf.make_rdm1()
    # CCSD calculations
    ccsd = cc.CCSD(mf)
    ccsd.kernel()
    data['converged'] = ccsd.converged
    assert data['converged']
    data['dm_ccsd'] = ccsd.make_rdm1(ao_repr=True)
    data['E_ccsd'] = ccsd.e_tot
    data['E_ccsdt'] = data['E_ccsd'] + ccsd.ccsd_t()
    
    eris = ccsd.ao2mo()
    if spin:
        data['F_b3'] = -guks(df).kernel()
        data['F_hf'] = -guhf(mf).kernel()
        if data['n_elec'] == 1:
            data['F_ccsd'] = data['F_hf']
            data['F_ccsdt'] = data['F_hf']
        else:
            data['F_ccsd'] = -guccsd(ccsd).kernel(ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris=eris)
            if F_ccsdt:
                data['F_ccsdt'] = -guccsdt(ccsd).kernel(ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris=eris)
        data['rho01_b3'] = dm2rho01(data['dm_b3'][0], data['phi01']), dm2rho01(data['dm_b3'][1], data['phi01'])
        data['rho01_hf'] = dm2rho01(data['dm_hf'][0], data['phi01']), dm2rho01(data['dm_hf'][1], data['phi01'])
        data['rho01_ccsd'] = dm2rho01(data['dm_ccsd'][0], data['phi01']), dm2rho01(data['dm_ccsd'][1], data['phi01'])
        data['rho_b3'] = data['rho01_b3'][0][0] + data['rho01_b3'][1][0]
        data['rho_hf'] = data['rho01_hf'][0][0] + data['rho01_hf'][1][0]
        data['rho_ccsd'] = data['rho01_ccsd'][0][0] + data['rho01_ccsd'][1][0]
    else:
        data['F_b3'] = -gks(df).kernel()
        data['F_hf'] = -ghf(mf).kernel()
        data['F_ccsd'] = -gccsd(ccsd).kernel(ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris=eris)
        if F_ccsdt:
            data['F_ccsdt'] = -gccsdt(ccsd).kernel(ccsd.t1, ccsd.t2, ccsd.l1, ccsd.l2, eris=eris)
        data['rho01_b3'] = dm2rho01(data['dm_b3'], data['phi01'])
        data['rho01_hf'] = dm2rho01(data['dm_hf'], data['phi01'])
        data['rho01_ccsd'] = dm2rho01(data['dm_ccsd'], data['phi01'])
        data['rho_b3'] = data['rho01_b3'][0]
        data['rho_hf'] = data['rho01_hf'][0]
        data['rho_ccsd'] = data['rho01_ccsd'][0]
    data['dipole_b3'] = cal_dipole(data['rho_b3'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    data['dipole_hf'] = cal_dipole(data['rho_hf'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords']) 
    data['dipole_ccsd'] = cal_dipole(data['rho_ccsd'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
    data['d3bj'] = d3.DFTD3Dispersion(mol, xc="b3lyp", version="d3bj").kernel()[0]
    data['time'] = time.time() - t0
    return data

# def gen_pkl_zmp(xyz_dict, charge, spin, basis, grid_level, delta=5e-3):
#     data = {}
#     mol = gto.M(atom=xyz_dict_to_str(xyz_dict), charge=charge, spin=spin, basis=basis)
#     data['atoms_charges'] = mol.atom_charges()
#     data['atoms_coords'] = mol.atom_coords()
#     data['basis'] = basis
#     data['spin'] = spin
#     data['charge'] = charge
#     # B3 calculations
#     df = dft.KS(mol)
#     df.xc = 'b3lypg'
#     df.grids.level = grid_level
#     df.kernel()
#     data['E_b3'] = df.e_tot
#     data['dm_b3'] = df.make_rdm1()
#     # df.xc = 'pbe'
#     # df.kernel()
#     # data['E_pbe'] = df.e_tot
#     # data['dm_pbe'] = df.make_rdm1()
#     data['gc'], data['gw'] = df.grids.coords, df.grids.weights
#     aux_gc = data['gc'][:,None,:] + [[0, 0, 0], [delta, 0, 0], [0, delta, 0], [0 ,0, delta], [0, delta, delta], [delta, 0, delta], [delta, delta, 0], [delta, delta, delta]]
#     aux_gc = aux_gc.reshape(-1, 3)
#     ni = NumInt()
#     aux_phi012 = ni.eval_ao(mol, aux_gc, deriv=2)
#     aux_phi01 = aux_phi012[:4]
#     aux_phi2 = aux_phi012[[4,7,9],:]
#     phi012 = ni.eval_ao(mol, data['gc'], deriv=2)
#     data['phi01'] = phi012[:4]
#     phi2 = phi012[[4,7,9],:]
#     # CCSD calculations
#     mf = scf.HF(mol)
#     mf.kernel()
#     C = mf.mo_coeff
#     ccsd = cc.CCSD(mf)
#     ccsd.kernel()
#     rdm1_hfmo = ccsd.make_rdm1()
#     rdm2_hfmo = ccsd.make_rdm2()
#     data['converged'] = ccsd.converged
#     data['E_ccsd'] = ccsd.e_tot
#     data['S'] = mol.intor('int1e_ovlp_sph')
#     data['itg_2e'] = mol.intor('int2e_sph')
#     data['T'] = mol.intor('cint1e_kin_sph')
#     data['V_ext'] = mol.intor('cint1e_nuc_sph')
#     data['n_elec'] = mol.nelectron
#     data['E_N'] = mol.energy_nuc()
#     aux_nu = mol.intor('int1e_grids_sph', grids=aux_gc)
#     nu = mol.intor('int1e_grids_sph', grids=data['gc'])
#     if spin:
#         if len(C)==2:
#             rdm1_ao1 = contract('pi,ij,qj->pq', C[0], rdm1_hfmo[0], C[0].conj())
#             rdm1_ao2 = contract('pi,ij,qj->pq', C[1], rdm1_hfmo[1], C[1].conj())
#             data['dm_ccsd'] = rdm1_ao1, rdm1_ao2
#             data['dm_zmp'], data['v_xc'] = gen_vxc(data)
#         else:
#             rdm1_ao1 = contract('pi,ij,qj->pq', C, rdm1_hfmo[0], C.conj())
#             rdm1_ao2 = contract('pi,ij,qj->pq', C, rdm1_hfmo[1], C.conj())
#             data['dm_ccsd'] = rdm1_ao1, rdm1_ao2
#             data['dm_zmp'], data['v_xc'] = gen_vxc(data)
#             data['dm_zmp2'] = rdm1_ao1, rdm1_ao2
#             # data['v_xc'] = -0.5 * dm2eK(data['dm_ccsd'][0], data['phi01'][0], nu)
#             data['v_xc2'] = -eval_vh(data['dm_ccsd'][0], nu)
#         data['rho01_b3'] = dm2rho01(data['dm_b3'][0], data['phi01']), dm2rho01(data['dm_b3'][1], data['phi01'])
#         # data['rho01_pbe'] = dm2rho01(data['dm_pbe'][0], data['phi01']), dm2rho01(data['dm_pbe'][1], data['phi01'])
#         data['rho01_ccsd'] = dm2rho01(data['dm_ccsd'][0], data['phi01']), dm2rho01(data['dm_ccsd'][1], data['phi01'])
#         data['rho01_zmp'] = dm2rho01(data['dm_zmp'][0], data['phi01']), dm2rho01(data['dm_zmp'][1], data['phi01'])
#         data['rho_ccsd'] = data['rho01_ccsd'][0][0] + data['rho01_ccsd'][1][0]
#         data['rho_b3'] = data['rho01_b3'][0][0] + data['rho01_b3'][1][0]
#         # data['rho_pbe'] = data['rho01_pbe'][0][0] + data['rho01_pbe'][1][0]
#         data['rho_zmp'] = data['rho01_zmp'][0][0] + data['rho01_zmp'][1][0]
#         for method in ['ccsd', 'b3', 'zmp']:
#             data[f'J_{method}'] = dm2J(data.get(f'dm_{method}'), data['itg_2e'])
#             data[f'K_{method}'] = dm2K(data.get(f'dm_{method}')[0], data['itg_2e']), dm2K(data.get(f'dm_{method}')[1], data['itg_2e'])
#             data[f'E_T_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['T'])
#             data[f'E_ext_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['V_ext'])
#             data[f'E_J_{method}'] = 0.5 * dm2E1e(data.get(f'dm_{method}'), data[f'J_{method}'])
#         if len(C)==2:
#             rdm2_ao_aa = contract('ijkl, pi, qj, rk, sl-> pqrs', rdm2_hfmo[0], C[0], C[0], C[0], C[0])
#             rdm2_ao_ab = contract('ijkl, pi, qj, rk, sl-> pqrs', rdm2_hfmo[1], C[0], C[0], C[1], C[1])
#             rdm2_ao_bb = contract('ijkl, pi, qj, rk, sl-> pqrs', rdm2_hfmo[2], C[1], C[1], C[1], C[1])
#             eEE = dm2eEE(rdm2_ao_aa+rdm2_ao_bb+2.*rdm2_ao_ab, data['phi01'][0], nu)
#         else:
#             rdm2_ao_aa = contract('ijkl, pi, qj, rk, sl-> pqrs', rdm2_hfmo[0], C, C, C, C)
#             eEE = dm2eEE(rdm2_ao_aa, data['phi01'][0], nu)
#         eJ = dm2eJ(rdm1_ao1+rdm1_ao2, data['phi01'][0], nu)
#         data['e_T_ccsd'] = .5*dm2eT2(rdm1_ao1+rdm1_ao2, data['phi01'][0], phi2)
#         data['e_T_zmp'] = .5*dm2eT2(data['dm_zmp'][0]+data['dm_zmp'][1], data['phi01'][0], phi2)
#         e_dT = data['e_T_ccsd'] - data['e_T_zmp']
#         data['e_xc_noT'] = 0.5 * ( eEE - eJ )
#         data['e_xc'] = data['e_xc_noT'] + e_dT
        
#         # data['E_xc_noT'] = 0.5*(np.sum(data['itg_2e']*rdm2_ao_aa) + np.sum(data['itg_2e']*rdm2_ao_bb)
#         #                         + 2*np.sum(data['itg_2e']*rdm2_ao_ab)) - data['E_J_ccsd']
#         # up = (data['n_elec'] + data['spin']) // 2
#         # down = data['n_elec'] - up
#         # F = data['T'] + data['V_ext'] + data['J_zmp'] + data['V_xc'][0], data['T'] + data['V_ext'] + data['J_zmp'] + data['V_xc'][1]
#         # e0, c0 = solve_KS_mat_2step(F[0], data['S'])
#         # e1, c1 = solve_KS_mat_2step(F[1], data['S'])
#         # data['mo_e_zmp'], data['mo_coeff_zmp'] = (np.diag(e0[:up]), np.diag(e1[:down])), (c0[:,:up], c1[:,:down])
#     else:
#         data['dm_ccsd'] = contract('pi,ij,qj->pq', C, rdm1_hfmo, C.conj())
#         data['dm_zmp'], data['v_xc'] = gen_vxc(data, 'b3lypg')
#         data['rho01_b3'] = dm2rho01(data['dm_b3'], data['phi01'])
#         # data['rho01_pbe'] = dm2rho01(data['dm_pbe'], data['phi01'])
#         aux_rho01 = dm2rho01(data['dm_ccsd'], aux_phi01)
#         aux_rho = aux_rho01[0].reshape(len(data['gc']), -1)
#         aux_gamma = contract('dr,dr->r', aux_rho01[1:], aux_rho01[1:]).reshape(len(data['gc']), -1)
#         aux_tau = .5*dm2eT(data['dm_zmp'], aux_phi01[1:]).reshape(len(data['gc']), -1)
#         data['rho01_ccsd'] = dm2rho01(data['dm_ccsd'], data['phi01'])
#         data['rho01_zmp'] = dm2rho01(data['dm_zmp'], data['phi01'])
#         data['rho_ccsd'] = data['rho01_ccsd'][0]
#         data['rho_b3'] = data['rho01_b3'][0]
#         # data['rho_pbe'] = data['rho01_pbe'][0]
#         data['rho_zmp'] = data['rho01_zmp'][0]
#         for method in ['ccsd', 'b3', 'zmp']:
#             data[f'J_{method}'] = dm2J(data.get(f'dm_{method}'), data['itg_2e'])
#             data[f'K_{method}'] = dm2K(data.get(f'dm_{method}'), data['itg_2e'])
#             data[f'E_T_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['T'])
#             data[f'E_ext_{method}'] = dm2E1e(data.get(f'dm_{method}'), data['V_ext'])
#             data[f'E_J_{method}'] = 0.5 * dm2E1e(data.get(f'dm_{method}'), data[f'J_{method}'])
#         rdm2_ao = contract('ijkl, pi, qj, rk, sl-> pqrs', rdm2_hfmo, C, C, C, C)
#         aux_eEE = dm2eEE(rdm2_ao, aux_phi01[0], aux_nu)
#         eEE = dm2eEE(rdm2_ao, data['phi01'][0], nu)
#         aux_eJ = dm2eJ(data['dm_ccsd'], aux_phi01[0], aux_nu)
#         eJ = dm2eJ(data['dm_ccsd'], data['phi01'][0], nu)
#         aux_e_T_ccsd = .5*dm2eT2(data['dm_ccsd'], aux_phi01[0], aux_phi2)
#         data['e_T_ccsd'] = .5*dm2eT2(data['dm_ccsd'], data['phi01'][0], phi2)
#         aux_e_T_zmp = .5*dm2eT2(data['dm_zmp'], aux_phi01[0], aux_phi2)
#         data['e_T_zmp'] = .5*dm2eT2(data['dm_zmp'], data['phi01'][0], phi2)
#         aux_e_dT = aux_e_T_ccsd - aux_e_T_zmp
#         e_dT = data['e_T_ccsd'] - data['e_T_zmp']
#         aux_e_xc_noT = 0.5 * ( aux_eEE - aux_eJ )
#         data['e_xc_noT'] = 0.5 * ( eEE - eJ )
#         aux_exc = (aux_e_xc_noT + aux_e_dT).reshape(len(data['gc']), -1)
#         # aux_exc, aux_vxc = ni.eval_xc('b3lypg', aux_rho01, spin, relativity=0, deriv=1, verbose=None)[:2]
#         # aux_exc = aux_exc.reshape(len(data['gc']), -1)*aux_rho
#         # aux_vxc = aux_vxc[0].reshape(len(data['gc']), -1)[:,0], aux_vxc[1].reshape(len(data['gc']), -1)[:,0]
#         data['e_xc'] = data['e_xc_noT'] + e_dT
#         xs = np.array([aux_rho[:,1:]-aux_rho[:,0:1], aux_gamma[:,1:]-aux_gamma[:,0:1], aux_tau[:,1:]-aux_tau[:,0:1]])
#         ys = aux_exc[:,1:]-aux_exc[:,0:1]
#         data['v_xc_num'] = numerical_vxc(xs, ys)
#         # data['v_xc_b3'] = aux_vxc
#         # data['E_xc_noT'] = 0.5*np.sum(data['itg_2e']*rdm2_ao) - data['E_J_ccsd']
#         # data['E_xc'] = data['E_ccsd'] - data['E_N'] - data['E_T_zmp'] - data['E_ext_zmp'] - data['E_J_zmp']
#         # up = data['n_elec'] // 2
#         # F = data['T'] + data['V_ext'] + data['J_zmp'] + data['V_xc']
#         # e, c = solve_KS_mat_2step(F, data['S'])
#         # data['mo_e_zmp'], data['mo_coeff_zmp'] = np.diag(e[:up]), c[:,:up]
        
#     data['E_xc'] = data['E_ccsd'] - data['E_N'] - data['E_T_zmp'] - data['E_ext_ccsd'] - data['E_J_ccsd']
#     data['I_b3'] = cal_I(data['rho_ccsd'], data['rho_b3'], data['gw'])
#     # data['I_pbe'] = cal_I(data['rho_ccsd'], data['rho_pbe'], data['gw'])
#     data['I_zmp'] = cal_I(data['rho_ccsd'], data['rho_zmp'], data['gw'])
#     data['dipole_ccsd'] = cal_dipole(data['rho_ccsd'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
#     data['dipole_b3'] = cal_dipole(data['rho_b3'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
#     # data['dipole_pbe'] = cal_dipole(data['rho_pbe'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
#     data['dipole_zmp'] = cal_dipole(data['rho_zmp'], data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
#     data['d3bj'] = d3.DFTD3Dispersion(mol, xc="b3lyp", version="d3bj").kernel()[0]
    
    
if __name__ == "__main__":
    xyz_dict, charge, spin = read_xyz_to_dict('/home/alfred/tree_regression/data/xyz/G2/c5h12.xyz')
    # data = gen_pkl(xyz_dict, charge, spin, 'aug-cc-pvdz', 3)
    data = gen_pkl_e(xyz_dict, charge, spin, 'aug-cc-pvdz', 3, delta=5e-3)
    with open('/home/alfred/tree_regression/data/pkl/extra/c5h12_0.0000_pvdz_e.pkl', 'wb') as f:
        pk.dump(data, f)
        
        
