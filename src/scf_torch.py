import torch
from opt_einsum import contract


def solve_KS_mat_2step(F, S):
    s_cutoff = 1.0e-10
    eps = 1.0e-10
    s, U = torch.linalg.eigh(S)
    s, U = torch.where(s>s_cutoff, s, 0), torch.where(s>s_cutoff, U, 0)
    X = U * torch.unsqueeze(1.0/(torch.sqrt(s)+eps), 0)
    Fp = torch.einsum('ji,jk,kl->il', X, F, X)
    e, Cp = torch.linalg.eigh(Fp)
    C = torch.einsum('ij,jk->ik', X, Cp)
    return e, C

def F_to_dm(F, S, n_up, n_down, device):
    lvls = torch.arange(len(S))
    occ_down, occ_up = torch.where(lvls<n_down,1.0,0.0), torch.where(lvls<n_up,1.0,0.0)
    if len(F) == 2:
        Fa, Fb = F
        ea, Ca = solve_KS_mat_2step(Fa, S)
        eb, Cb = solve_KS_mat_2step(Fb, S)
        return contract('ik,k,jk->ij', Ca, occ_up.to(device), Ca), contract('ik,k,jk->ij', Cb, occ_down.to(device), Cb)
    else:
        e, C = solve_KS_mat_2step(F, S)
        return contract('ik,k,jk->ij', C, (occ_down+occ_up).to(device), C)

# def F_to_C(F, S):
#     s_cutoff = 1.0e-10
#     eps = 1.0e-10
#     s, U = torch.linalg.eigh(S)
#     s, U = torch.where(s>s_cutoff, s, 0), torch.where(s>s_cutoff, U, 0)
#     X = U * torch.unsqueeze(1.0/(torch.sqrt(s)+eps), 0)
#     Fp = torch.einsum('ji,jk,kl->il', X, F, X)
#     e, Cp = torch.linalg.eigh(Fp)
#     C = torch.einsum('ij,jk->ik', X, Cp)
#     return e, C

class Diis:
    def __init__(self, S, spin, device, diis_len=14, notH=True):
        # print(f'Using diis_len={diis_len}')
        self.device = device
        S = torch.tensor(S).to(device)
        S_e, S_evec = torch.linalg.eigh(S)
        S_e_inv_sqrt = 1./torch.sqrt(S_e)
        self.S = S
        self.O = contract('ik,k,jk->ij', S_evec, S_e_inv_sqrt, S_evec)
        self.diis_len = diis_len
        self.spin = spin
        self.notH = notH
        if self.diis_len > 1:
            self.n_orbs = len(S[0])
            if self.spin and self.notH:
                self.ems = torch.zeros((self.diis_len, 2, self.n_orbs, self.n_orbs)).to(device)
                self.pms = torch.zeros((self.diis_len, 2, self.n_orbs, self.n_orbs)).to(device)
                # self.erra = []
                # self.errb = []
            else:
                self.ems = torch.zeros((self.diis_len, self.n_orbs, self.n_orbs)).to(device)
                self.pms = torch.zeros((self.diis_len, self.n_orbs, self.n_orbs)).to(device)
                # self.err = []
            self.itr = 0
            
    
    def __call__(self, H, dm):
        if self.diis_len > 1:
            if self.spin and self.notH:
                ema = contract('ij,jk,kl->il', H[0], dm[0], self.S)
                emb = contract('ij,jk,kl->il', H[1], dm[1], self.S)
                ema = ema - ema.T
                emb = emb - emb.T
                ema = contract('ji,jk,kl->ij',self.O, ema, self.O)
                emb = contract('ji,jk,kl->ij',self.O, emb, self.O)
                # self.erra += np.max(np.abs(ema))
                # self.errb += np.max(np.abs(ema))
                self.ems[self.itr%self.diis_len, 0] = ema
                self.ems[self.itr%self.diis_len, 1] = emb
                self.pms[self.itr%self.diis_len, 0] = H[0]
                self.pms[self.itr%self.diis_len, 1] = H[1]
                if self.itr == 0: 
                    self.itr += 1
                    return H
                #Solve BC = A to find C
                nb = min(self.itr+1, self.diis_len)
                B = -torch.ones((2, nb+1, nb+1)).to(self.device)
                B[:, nb, nb] = 0.
                B[:, :nb, :nb] = contract('asij,bsji->sab', self.ems[:nb, :, :, :], self.ems[:nb, :, :, :])
                A = torch.zeros(2, nb+1).to(self.device)
                A[:, nb] = -1.0
                Ca = torch.linalg.solve(B[0], A[0])
                Cb = torch.linalg.solve(B[1], A[1])
                # form new extrapolated diis fock matrix
                diis_Ha = torch.sum(Ca[:-1, None, None]*self.pms[:nb, 0], axis=0)
                diis_Hb = torch.sum(Cb[:-1, None, None]*self.pms[:nb, 1], axis=0)
                self.itr += 1
                # print(np.sum(C[:-1]), np.max(np.abs(diis_H-H)))
                return diis_Ha, diis_Hb
            else:
                em = contract('ij,jk,kl->il', H, dm, self.S)
                em = em - em.T
                em = contract('ji,jk,kl->ij',self.O, em, self.O)
                # self.err += np.max(np.abs(em))
                self.ems[self.itr%self.diis_len] = em
                self.pms[self.itr%self.diis_len] = H
                if self.itr == 0: 
                    self.itr += 1
                    return H
                #Solve BC = A to find C
                nb = min(self.itr+1, self.diis_len)
                B = -torch.ones((nb+1, nb+1)).to(self.device)
                B[nb, nb] = 0.
                B[:nb, :nb] = contract('aij,bji->ab', self.ems[:nb, :, :], self.ems[:nb, :, :])
                A = torch.zeros(nb+1).to(self.device)
                A[nb] = -1.0
                C = torch.linalg.solve(B, A)
                # form new extrapolated diis fock matrix
                diis_H = torch.sum(C[:-1,None,None]*self.pms[:nb], axis=0)
                self.itr += 1
                # print(np.sum(C[:-1]), np.max(np.abs(diis_H-H)))
                return diis_H
        else:
            return H

def scf_once(Vxc, n_elec, dm, S, J, spin, device, T, Vext, diis=None):
    n_up = (n_elec+spin)//2
    n_down = n_elec - n_up
    if spin:
        Fa = torch.tensor(T + Vext, device=device) + J + Vxc[0]
        Fb = torch.tensor(T + Vext, device=device) + J + Vxc[1]
        if diis:
            if torch.abs(dm[1]).sum()>1e-10:
                Fa, Fb = diis((Fa, Fb), dm)
                dm_a, dm_b = F_to_dm((Fa, Fb), torch.tensor(S, device=device), n_up, n_down, device)
                dm_new = torch.stack((dm_a, dm_b))
            else:
                # Fa = diis(Fa, dm[0])
                # dm_a = F_to_dm(Fa, torch.tensor(S, device=device), n_up, n_down, device)
                # dm_new = torch.stack((dm_a, torch.zeros_like(dm_a)))
                dm_a, dm_b = F_to_dm((Fa, Fb), torch.tensor(S, device=device), n_up, n_down, device)
                dm_new = torch.stack((dm_a, dm_b))
        else:
            dm_a, dm_b = F_to_dm((Fa, Fb), torch.tensor(S, device=device), n_up, n_down, device)
            dm_new = torch.stack((dm_a, dm_b))
    else:
        F = torch.tensor(T + Vext, device=device) + J + Vxc
        if diis:
            F = diis(F, dm)
        dm_new = F_to_dm(F, torch.tensor(S, device=device), n_up, n_down, device)
    return dm_new


def energy(E_xc, dm, itg_2e, spin, device, T, Vext):
    if spin:
        E_T_new, E_ext_new = torch.sum((dm[0]+dm[1]) * torch.tensor(T, device=device)), torch.sum((dm[0]+dm[1]) * torch.tensor(Vext, device=device))
        E_J_new = 0.5 * torch.sum(dm2J(dm, itg_2e) * (dm[0]+dm[1]))
    else:
        E_T_new, E_ext_new = torch.sum(dm * torch.tensor(T, device=device)), torch.sum(dm * torch.tensor(Vext, device=device))
        E_J_new = 0.5 * torch.sum(dm2J(dm, itg_2e) * dm)
    E_new = E_T_new + E_ext_new + E_J_new
    return E_new+E_xc, {'E_T':E_T_new.detach().cpu().numpy(), 'E_ext':E_ext_new.detach().cpu().numpy(), 'E_J':E_J_new.detach().cpu().numpy(), 'E_xc':E_xc}

def gen_features(dm, phi01, point='single', info=['rho', 'gnorm', 'tau']):
    if point=='single':
        feature = []
        rho01 = dm2rho01_sep(dm, phi01)
        if 'rho' in info:
            feature.append(rho01[0])
        if 'gnorm' in info:
            feature.append(contract('dr,dr->r', rho01[1:], rho01[1:])**0.5)
        if 'tau' in info:
            feature.append(0.5*contract('ij, dri, drj->r', dm, phi01[1:], phi01[1:]))
        if 'bni' in info:
            tau_s  = 0.5*contract('ij, dri, drj->r', dm, phi01[1:], phi01[1:])
            tau_w = 1/8*contract('dr,dr->r', rho01[1:], rho01[1:])/rho01[0]
            feature.append((tau_s-tau_w)/tau_w)
        return np.array(feature).T

def cal_Vxc(v_xc, phi01, gw, rho01, device):
    phi01 = torch.tensor(phi01).to(device)
    gw = torch.tensor(gw).to(device)
    phi, phi1 = phi01[0], phi01[1:]
    if len(v_xc) == 3:
        rho01 = torch.tensor(rho01, device=device)
        Vxc = torch.einsum('ri,r,r,rj->ij', phi, v_xc[0], gw, phi)
        Vxc += 4.0*torch.einsum('dri,r,dr,r,rj->ij', phi1, v_xc[1], rho01[1:], gw, phi)
        Vxc += 0.5*torch.einsum('dri,r,r,drj->ij', phi1, v_xc[2], gw, phi1)
        Vxc = (Vxc + Vxc.conj().T) / 2.0
        return Vxc
    else:
        rho1_a, rho1_b = torch.tensor(rho01[0][1:], device=device), torch.tensor(rho01[1][1:], device=device)
        
        Vxc_a = torch.einsum('ri,r,r,rj->ij', phi, v_xc[0][0], gw, phi)
        Vxc_a += 4. * torch.einsum('dri,r,dr,r,rj->ij', phi1, v_xc[1][0], rho1_a, gw, phi)
        Vxc_a += 2. * torch.einsum('dri,r,dr,r,rj->ij', phi1, v_xc[1][1], rho1_b, gw, phi)
        Vxc_a += 0.5 * torch.einsum('dri,r,r,drj->ij', phi1, v_xc[2][0], gw, phi1)
        
        
        Vxc_b = torch.einsum('ri,r,r,rj->ij', phi, v_xc[0][1], gw, phi)
        Vxc_b += 4. * torch.einsum('dri,r,dr,r,rj->ij', phi1, v_xc[1][2], rho1_b, gw, phi)
        Vxc_b += 2. * torch.einsum('dri,r,dr,r,rj->ij', phi1, v_xc[1][1], rho1_a, gw, phi)
        Vxc_b += 0.5 * torch.einsum('dri,r,r,drj->ij', phi1, v_xc[2][1], gw, phi1)
        
        Vxc_a = (Vxc_a + Vxc_a.conj().T) / 2.0
        Vxc_b = (Vxc_b + Vxc_b.conj().T) / 2.0
        return Vxc_a, Vxc_b


def scf(n_elec, dm, S, itg_2e, spin, device, E_N, T, Vext, gw, max_iter=200, criterion=1e-6, notH=True):
    with torch.no_grad():
        Exc = torch.sum(exc*rho*gw)
        Exc, Vxc = gen_xc(dm.detach().cpu().numpy(),aux_gc, aux_phi01, gw, device)
        diis = Diis(S, spin, device, notH=notH)
        E, E_term = energy(Exc, dm, itg_2e, spin, device, T, Vext)
        converged = False
        E_diff = []
        dms = []
        Es = []
        E_terms = []
        
        for iter in range(max_iter):
            # Vxc = contract('ri,rj,r,r->ij', phi, phi, vxc, gw)
            dm_new = scf_once(Vxc, n_elec, dm, S, J, spin, device, T, Vext, diis=diis)
            E_new, E_term_new = energy(Exc, dm_new, itg_2e, spin, device, T, Vext)
            E_diff.append(torch.abs(E_new-E).item())
            print('iter:', iter, '\tdiff E:', E_diff[-1])
            dms.append(dm)
            Es.append(E)
            E_terms.append(E_term)
            if torch.abs(E_new-E).item() <= criterion:
                converged = True
                break
            E = E_new
            E_term = E_term_new
            dm = dm_new
        if not converged:
            print('min diff E:', min(E_diff), 'at iter', E_diff.index(min(E_diff))+1, '\033[1;32mincrease criterion as %.11f\033[0m'%(min(E_diff)+1e-10))
            dm = dms[E_diff.index(min(E_diff))]
            E = Es[E_diff.index(min(E_diff))]
            E_term = E_terms[E_diff.index(min(E_diff))]
    E_term.update({'E': E.cpu().numpy()+E_N, 'dm': dm_new})
    return E_term

if __name__=="__main__":
    import pickle as pk
    from pyscf import gto, dft
    with open('/home/alfred/tree_regression/data/pkl/G2/h2o_0.0000_0_aug-cc-pvdz_train_valid.pkl', 'rb') as f:
        data= pk.load(f)
    mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in
            zip(data['atoms_charges'], data['atoms_coords'])], basis=data['basis'],
                    spin=data['spin'], charge=data['charge'], unit='bohr', verbose=0)
    ks = dft.KS(mol)
    grid = dft.gen_grid.Grids(mol)
    grid.coords = data['gc']
    grid.weights = data['gw']
    itg_2e = mol.intor('int2e_sph')

