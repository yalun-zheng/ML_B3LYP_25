from opt_einsum import contract
import math
import numpy as np

def dm2rho(dm, phi):
    if len(dm)==2:
        dm =dm[0]+dm[1]
    return contract('ri, rj, ij->r', phi, phi.conj(), dm)

def dm2rho01(dm, phi01):
    if len(dm)==2:
        dm =dm[0]+dm[1]
    rho01 = contract('dri, rj, ij->dr', phi01, phi01[0].conj(), dm)
    rho01[1:] *= 2.
    return rho01

def dm2rho01_sep(dm, phi01):
    if len(dm)==2:
        rho01 = contract('dri, rj, uij->udr', phi01, phi01[0].conj(), dm)
        rho01[0][1:] *= 2.
        rho01[1][1:] *= 2.
    else:      
        rho01 = contract('dri, rj, ij->dr', phi01, phi01[0].conj(), dm)
        rho01[1:] *= 2.
    return rho01

def cal_I(rho1, rho2, gw):
    if len(rho1)==2:
        rho1 = rho1[0]+rho1[1]
    if len(rho2)==2:
        rho2 = rho2[0]+rho2[1]
    return contract('r,r->', (rho1-rho2)**2, gw)/contract('r,r->', (rho1)**2+(rho2)**2, gw)

def cal_dipole(rho, gc, gw, atoms_charges, atoms_coords):
  if len(rho)==2:
    rho = rho[0]+rho[1]
  dipole = - contract('ri, r, r->i', gc, gw, rho)
  for an, ac in zip(atoms_charges, atoms_coords):
    dipole += contract('i->i', ac) * an
  return dipole

def dm2E1e(dm, o):
    if len(dm)==2:
        dm =dm[0]+dm[1]
    return contract('ij,ij->', dm, o)

def dm2J(dm, itg_2e):
    if len(dm)==2:
        dm =dm[0]+dm[1]
    return contract('ijkl, lk-> ij', itg_2e, dm)

def dm2K(dm, itg_2e):
    return contract('ijkl, jk-> il', itg_2e, dm)

# def cal_chi(mol, dm, phi_, gc_, w=0., chunk=4000):
#     try:
#         import fortran
#     except:
#         import os
#         curdir = os.path.abspath(os.path.curdir)
#         os.chdir(os.path.dirname(__file__))
#         os.system("f2py -c --f90flags='-fopenmp' -lgomp -llapack fortranscript.f90 -m fortran")
#         os.chdir(curdir)
#         import fortran
#     chi_ = []
#     nu_ = []
#     for chunk_id in range(math.ceil(len(gc_)/chunk)):
#         phi = phi_[chunk_id*chunk:chunk_id*chunk+chunk,:]
#         gc = gc_[chunk_id*chunk:chunk_id*chunk+chunk]
#         with mol.with_range_coulomb(omega=w):
#             nu = mol.intor('int1e_grids_sph', grids=gc)
#             nu_.append(nu)
#         # chi = contract('rlj,ri,ij->rl', nu, phi, dm)
#         # chi_J = contract('rij,rl,ij->rl', nu, phi, dm)
#         chi = np.asfortranarray(np.zeros_like(phi))
#         fortran.nu2chi(nu, phi, dm, chi)
#         chi_.append(chi)
#     return np.vstack(chi_), np.vstack(nu_)

def cal_nu(mol, gc_, chunk, w):
    with mol.with_range_coulomb(omega=w):
        for chunk_id in range(math.ceil(len(gc_)/chunk)):
            gc = gc_[chunk_id*chunk:chunk_id*chunk+chunk]
            nu = mol.intor('int1e_grids_sph', grids=gc)
            yield nu

def cal_chi(mol, dm, phi_, gc_, w=0., chunk=1000):
    try:
        import fortran
    except:
        import os
        curdir = os.path.abspath(os.path.curdir)
        os.chdir(os.path.dirname(__file__))
        os.system("f2py -c --f90flags='-fopenmp' -lgomp -llapack fortranscript.f90 -m fortran")
        os.chdir(curdir)
        import fortran
    chi_ = []
    for chunk_id, nu in enumerate(cal_nu(mol, gc_, chunk, w)):
        phi = phi_[chunk_id*chunk:chunk_id*chunk+chunk,:]
        chi = contract('rlj,ri,ij->rl', nu, phi, dm)
        # chi = np.asfortranarray(np.zeros_like(phi))
        # fortran.nu2chi(nu, phi, dm, chi)
        chi_.append(chi)
    return np.vstack(chi_), None

def cal_eden(chi, dm, phi):
    return -0.5 * contract('ij,rj,ri->r', dm, phi, chi)

def cal_eden_J(nu, dm, phi):
    return contract('ij,ri,rj,rkl,kl->r', dm, phi, phi, nu, dm)

def partial_den_dm(chi, phi, gw, v):
    return - contract('ri,rj,r,r->ij', chi, phi, gw, v)

def partial_den_dm_J(nu, dm, phi, gw, v):
    return contract('rkl,kl,ri,rj,r,r->ij', nu, dm, phi, phi, gw, v) + contract('rkl,ij,ri,rj,r,r->kl', nu, dm, phi, phi, gw, v)

def cal_feature(mol, phi01, dm, dm_zmp, spin, gc, omega=[0.,0.4], omega_J=False):
    e = []
    chi = []
    e_J = []
    if len(dm) == 2:
        rho010 = contract('dri, rj, ij->dr', phi01, phi01[0].conj(), dm[0])
        rho011 = contract('dri, rj, ij->dr', phi01, phi01[0].conj(), dm[1])
        rho010[1:] *= 2.
        rho011[1:] *= 2.
        rhou = rho010[0]
        rhod = rho011[0]
        dx1, dy1, dz1 = rho010[1:]
        dx2, dy2, dz2 = rho011[1:]
        nabla2u = dx1**2 + dy1**2 + dz1**2
        nabla2d = dx2**2 + dy2**2 + dz2**2
        nabla2 = (dx1+dx2)**2 + (dy1+dy2)**2 + (dz1+dz2)**2
        tauu = 0.5*contract('ij, dri, drj->r', dm_zmp[0], phi01[1:], phi01[1:])
        taud = 0.5*contract('ij, dri, drj->r', dm_zmp[1], phi01[1:], phi01[1:])
        for w in omega:
            chiu, nu = cal_chi(mol, dm_zmp[0], phi01[0], gc, w)
            e_HFxu = cal_eden(chiu, dm_zmp[0], phi01[0])
            if omega_J:
                e_ud = cal_eden_J(nu, dm_zmp[0]+dm_zmp[1], phi01[0])
                e_J.append(e_ud)
            chid, nu = cal_chi(mol, dm_zmp[1], phi01[0], gc, w)
            e_HFxd = cal_eden(chid, dm_zmp[1], phi01[0])
            e += [e_HFxu, e_HFxd]
            chi += [chiu, chid]
    else:
        rho01 = contract('dri, rj, ij->dr', phi01, phi01[0].conj(), dm)
        rho01[1:] *= 2.
        rho010 = rho011 = rho01 * 0.5
        dx, dy, dz = rho01[1:]
        rhou = rhod = rho01[0] * 0.5
        nabla2 = dx**2 +dy**2 +dz**2
        nabla2u = nabla2d = nabla2 * 0.25
        tau = 0.5*contract('ij, dri, drj->r', dm_zmp, phi01[1:], phi01[1:])
        tauu = taud = tau * 0.5
        for w in omega:
            chiu, nu = cal_chi(mol, dm_zmp*0.5, phi01[0], gc, w)
            chid = chiu
            e_HFxu = e_HFxd = cal_eden(chiu, dm_zmp*0.5, phi01[0])
            if omega_J:
                e_ud = cal_eden_J(nu, dm_zmp, phi01[0])
                e_J.append(e_ud)
            e += [e_HFxu, e_HFxd]
            chi += [chiu, chid]
    return np.vstack([rhou,rhod,nabla2u,nabla2d,nabla2,tauu,taud]+e+e_J).T, chi

def cal_feature_no_omega(phi01, dm, dm_tau=None):
    if dm_tau is None:
        dm_tau = dm
    if len(dm) == 2:
        rho010 = contract('dri, rj, ij->dr', phi01, phi01[0].conj(), dm[0])
        rho011 = contract('dri, rj, ij->dr', phi01, phi01[0].conj(), dm[1])
        rho010[1:] *= 2.
        rho011[1:] *= 2.
        # rhou = rho010[0]
        # rhod = rho011[0]
        # dx1, dy1, dz1 = rho010[1:]
        # dx2, dy2, dz2 = rho011[1:]
        # nabla2u = dx1**2 + dy1**2 + dz1**2
        # nabla2d = dx2**2 + dy2**2 + dz2**2
        # nabla2 = (dx1+dx2)**2 + (dy1+dy2)**2 + (dz1+dz2)**2
        tauu = 0.5*contract('ij, dri, drj->r', dm_tau[0], phi01[1:], phi01[1:])
        taud = 0.5*contract('ij, dri, drj->r', dm_tau[1], phi01[1:], phi01[1:])
    else:
        rho01 = contract('dri, rj, ij->dr', phi01, phi01[0].conj(), dm)
        rho01[1:] *= 2.
        rho010 = rho011 = 0.5 * rho01
        # dx, dy, dz = rho01[1:]
        # rhou = rhod = rho011[0]
        # nabla2 = dx**2 +dy**2 +dz**2
        # nabla2u = nabla2d = nabla2 * 0.25
        tau = 0.5*contract('ij, dri, drj->r', dm_tau, phi01[1:], phi01[1:])
        tauu = taud = tau * 0.5
    return np.vstack([rho010, tauu, rho011, taud]).T

def dm2eT(rdm1, phi1):
    if len(rdm1) == 2:
        rdm1 = rdm1[0] + rdm1[1]
        # return contract('dri,ij,drj->r', phi1, rdm1[0], phi1), contract('dri,ij,drj->r', phi1, rdm1[1], phi1)
    return contract('dri,ij,drj->r', phi1, rdm1, phi1)

def dm2eT2(rdm1, phi, phi2):
    if len(rdm1) == 2:
        # return -contract('ri,ij,drj->r', phi, rdm1[0], phi2), -contract('ri,ij,drj->r', phi, rdm1[1], phi2)
        return -contract('ri,ij,drj->r', phi, rdm1[0]+rdm1[1], phi2)
    else:
        return -contract('ri,ij,drj->r', phi, rdm1, phi2)

def dm2eJ(rdm1, phi, nu):
    if len(rdm1) == 2:
        # rhoup = contract('ri,ij,rj->r', phi, rdm1[0], phi)
        # rhodown = contract('ri,ij,rj->r', phi, rdm1[1], phi)
        # return contract('rij,r,ij->r', nu, rhoup, rdm1[0]), contract('rij,r,ij->r', nu, rhodown, rdm1[1]), contract('rij,r,ij->r', nu, rhoup, rdm1[1])+contract('rij,r,ij->r', nu, rhodown, rdm1[0])
        rho = contract('ri,ij,rj->r', phi, rdm1[0]+rdm1[1], phi)
        return contract('rij,r,ij->r', nu, rho, rdm1[0]+rdm1[1])
    else:
        rho = contract('ri,ij,rj->r', phi, rdm1, phi)
        return contract('rij,r,ij->r', nu, rho, rdm1)

def dm2eK(rdm1, phi, nu):
    if len(rdm1) == 2:
        Eir1 = contract('ij,ri->jr',rdm1[0], phi)
        Eir2 = contract('ij,ri->jr',rdm1[1], phi)
        # return contract('ir,jr,rij->r',Eir1, Eir1, nu)*2., contract('ir,jr,rij->r',Eir2, Eir2, nu)*2.
        return contract('ir,jr,rij->r',Eir1, Eir1, nu)*2. + contract('ir,jr,rij->r',Eir2, Eir2, nu)*2.
    else:
        Eir = contract('ij,ri->jr',rdm1, phi)
        return contract('ir,jr,rij->r',Eir, Eir, nu)

def dm2eEE(rdm2, phi, nu):
    Pr = contract('ijkl,ri,rj->rkl',rdm2, phi, phi)
    return contract('rij,rij->r', Pr, nu)

def dm2exc(rdm1, rdm2, phi01, nu, dm_zmp):
    eEE = dm2eEE(rdm2, phi01[0], nu)
    eJ = dm2eJ(rdm1, phi01[0], nu)
    e_dT = dm2eT(rdm1-dm_zmp, phi01[1:])
    return 0.5 * ( eEE - eJ + e_dT )

def eval_vh(dm, nu):
    if len(dm) == 2:
        return contract('rij,aij->ar', nu, dm)
    else:
        return contract('rij,ij->r', nu, dm)