import torch
torch.set_default_dtype(torch.float64)
from torch.nn import Module, Sequential, Linear, ELU, Conv3d, ReLU
import logging
import numpy as np
import matplotlib.pyplot as plt
from opt_einsum import contract
from pyscf import gto, dft
from pyscf.dft.numint import NumInt
from einsum import dm2rho, cal_dipole, dm2rho01_sep, dm2E1e, dm2K, dm2rho01, dm2J
from pyscf.grad.uks import Gradients as guks
from pyscf.grad.rks import Gradients as gks

def v2V(v_rho, v_gamma, phi01, gw, rho01):
    phi, phi1 = phi01[0], phi01[1:]
    if len(v_rho.shape) == 1:
        Vxc = contract('ri,r,r,rj->ij', phi, v_rho, gw, phi)
        Vxc += 4.0*contract('dri,r,dr,r,rj->ij', phi1, v_gamma, rho01[1:], gw, phi)
        # Vxc += 0.5*contract('dri,r,r,drj->ij', phi1, v_tau, gw, phi1)
        Vxc = (Vxc + Vxc.conj().T) / 2.0
        return Vxc
    else:
        rho1_a, rho1_b = rho01[0][1:], rho01[1][1:]
        Vxc_a = contract('ri,r,r,rj->ij', phi, v_rho[:, 0], gw, phi)
        Vxc_b = contract('ri,r,r,rj->ij', phi, v_rho[:, 1], gw, phi)
        Vxc_a += 4. * contract('dri,r,dr,r,rj->ij', phi1, v_gamma[:, 0], rho1_a, gw, phi)
        Vxc_a += 2. * contract('dri,r,dr,r,rj->ij', phi1, v_gamma[:, 1], rho1_b, gw, phi)
        
        Vxc_b += 4. * contract('dri,r,dr,r,rj->ij', phi1, v_gamma[:, 2], rho1_b, gw, phi)
        Vxc_b += 2. * contract('dri,r,dr,r,rj->ij', phi1, v_gamma[:, 1], rho1_a, gw, phi)
        # Vxc_a += 0.5 * contract('dri,r,r,drj->ij', phi1, v_tau[:, 0], gw, phi1)
        # Vxc_b += 0.5 * contract('dri,r,r,drj->ij', phi1, v_tau[:, 1], gw, phi1)
        Vxc_a = (Vxc_a + Vxc_a.conj().T) / 2.0
        Vxc_b = (Vxc_b + Vxc_b.conj().T) / 2.0
        return Vxc_a, Vxc_b

def v2V2(v_rho, v_gamma, v_tau, phi01, gw, rho01):
    phi, phi1 = phi01[0], phi01[1:]
    if len(v_rho.shape) == 1:
        Vxc = contract('ri,r,r,rj->ij', phi, v_rho, gw, phi)
        Vxc += 4.0*contract('dri,r,dr,r,rj->ij', phi1, v_gamma, rho01[1:], gw, phi)
        Vxc += 0.5*contract('dri,r,r,drj->ij', phi1, v_tau, gw, phi1)
        Vxc = (Vxc + Vxc.conj().T) / 2.0
        return Vxc
    else:
        rho1_a, rho1_b = rho01[0][1:], rho01[1][1:]
        Vxc_a = contract('ri,r,r,rj->ij', phi, v_rho[:, 0], gw, phi)
        Vxc_b = contract('ri,r,r,rj->ij', phi, v_rho[:, 1], gw, phi)
        Vxc_a += 4. * contract('dri,r,dr,r,rj->ij', phi1, v_gamma[:, 0], rho1_a, gw, phi)
        Vxc_a += 2. * contract('dri,r,dr,r,rj->ij', phi1, v_gamma[:, 1], rho1_b, gw, phi)
        
        Vxc_b += 4. * contract('dri,r,dr,r,rj->ij', phi1, v_gamma[:, 2], rho1_b, gw, phi)
        Vxc_b += 2. * contract('dri,r,dr,r,rj->ij', phi1, v_gamma[:, 1], rho1_a, gw, phi)
        Vxc_a += 0.5 * contract('dri,r,r,drj->ij', phi1, v_tau[:, 0], gw, phi1)
        Vxc_b += 0.5 * contract('dri,r,r,drj->ij', phi1, v_tau[:, 1], gw, phi1)
        Vxc_a = (Vxc_a + Vxc_a.conj().T) / 2.0
        Vxc_b = (Vxc_b + Vxc_b.conj().T) / 2.0
        return Vxc_a, Vxc_b

def solve_KS_mat_2step(F, S):
    s_cutoff = 1.0e-10
    eps = 1.0e-10
    s, U = torch.linalg.eigh(S)
    s, U = torch.where(s>s_cutoff, s, 0), torch.where(s>s_cutoff, U, 0)
    X = U * torch.unsqueeze(1.0/(torch.sqrt(s)+eps), 0)
    Fp = contract('ji,jk,kl->il', X, F, X)
    e, Cp = torch.linalg.eigh(Fp)
    C = contract('ij,jk->ik', X, Cp)
    return e, C

def F2dm(F, S, n_up, n_down, device):
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

class myRKS(dft.rks.RKS):
    def __init__(self, mol, xc='LDA,VWN', dm_target=None):
        self.dm_target = dm_target
        super().__init__(mol, xc)

    # get_veff = get_veff

class myUKS(dft.uks.UKS):
    def __init__(self, mol, xc='LDA,VWN', dm_target=None):
        self.dm_target = dm_target
        super().__init__(mol, xc)
        
    # get_veff = get_veff
    
def myKS(mol, xc='LDA,VWN', dm_target=None):
    if mol.spin == 0:
        return myRKS(mol, xc, dm_target)
    else:
        return myUKS(mol, xc, dm_target)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.args = args
        self.len_args = len(self.args)
    
    def __getitem__(self, index):
        return [self.args[i][index] for i in range(self.len_args)]
    
    def __len__(self):
        return len(self.args[-1])

class Model_e(Module):
    def __init__(self, feature_num=1331, in_channels=1, n_outs_node=4, device='cuda:7', **kwargs):
        super().__init__()
        self.device = device
        self.lin = Sequential(*([
        # Linear(feature_num, 256), ReLU(),
        # Linear(256, 256), ReLU(),
        # Linear(256, 128), ReLU(),
        # Linear(128, 128), ReLU(),
        # Linear(128, 64), ReLU(),
        # Linear(64, n_outs_node),
        Linear(feature_num, 40), ReLU(),
        Linear(40, 40), ReLU(),
        Linear(40, 40), ReLU(),
        Linear(40, n_outs_node),
        ]))
        self.loss_func = lambda a, b: torch.mean(torch.abs(a - b))
        self.optimizer = None
        self.n_outs_node = n_outs_node
    
    def forward_v(self, x):
        x = torch.tensor(x.reshape(len(x), -1), requires_grad=True)
        e = self.lin(x).reshape(len(x))
        if x.grad is not None:
            x.grad.zero_()
        v = torch.autograd.grad(torch.sum(e*(x[:,0]+x[:,1])), x, create_graph=True)[0][:,0:2]
        return e.reshape(-1), v.reshape(-1)

    def forward(self, x):
        x = x.reshape(len(x), -1)
        e = self.lin(x)
        if self.n_outs_node>1:
            return e
        else:
            return e.reshape(-1)
    
    def fit(self, x, e, valid_x, valid_e, coeff_train, coeff_test, model_path, lr=0.0001, batch_size=5120, max_iter=3000, save_every=1000, valid_every=100):
        self.to(self.device)
        coeff_train, coeff_test = torch.tensor(coeff_train).to(self.device), torch.tensor(coeff_test).to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False)
        x, e = torch.asarray(x).to(self.device), torch.asarray(e).to(self.device)
        valid_x, valid_e = torch.asarray(valid_x).to(self.device), torch.asarray(valid_e).to(self.device)
        dataset = Dataset(x, e, coeff_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        validloader = torch.utils.data.DataLoader(Dataset(valid_x, valid_e, coeff_test), batch_size=batch_size, shuffle=True)
        loss_record = []
        valid_loss_record = []
        for epoch in range(max_iter):
            epoch_loss = 0.
            for batch_x, batch_e, batch_coeff in dataloader:
                self.optimizer.zero_grad()
                batch_e_p = self.forward(batch_x)
                loss_e = self.loss_func(batch_e_p, batch_coeff)
                # loss_E = torch.mean(torch.abs(batch_e_p - batch_e)*batch_gw)
                # loss = loss_E
                epoch_loss += loss_e.item()*len(batch_e)
                loss_e.backward()
                self.optimizer.step()
            epoch_loss /= len(e)
            loss_record.append(epoch_loss)
            if (epoch+1)%valid_every==0:
                with torch.no_grad():
                    valid_loss = 0.
                    for batch_x, batch_e, batch_coeff in validloader:
                        batch_e_p = self.forward(batch_x)
                        loss_e = self.loss_func(batch_e_p, batch_coeff)
                        valid_loss += loss_e.item()*len(batch_e)
                    valid_loss /= len(valid_e)
                    valid_loss_record.append(valid_loss)

            if (epoch+1)%save_every==0:
                print(f'saving model at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
    
    def load(self, model_path):
        self.to(self.device)
        chk = torch.load(model_path)
        self.load_state_dict(chk['net'])
        self.optimizer = torch.optim.Adam(self.parameters(), foreach=False)
        self.optimizer.load_state_dict(chk['opt'])
    
class Model_e2(Module):
    def __init__(self, feature_num=5, n_outs_node=1, device='cuda:7', **kwargs):
        super().__init__()
        self.device = device
        self.lin = Sequential(*([
        Linear(feature_num, 256), ReLU(),
        Linear(256, 256), ReLU(),
        Linear(256, 128), ReLU(),
        Linear(128, 128), ReLU(),
        Linear(128, 64), ReLU(),
        Linear(64, n_outs_node),
        # Linear(feature_num, 40), ReLU(),
        # Linear(40, 40), ReLU(),
        # Linear(40, 40), ReLU(),
        # Linear(40, n_outs_node),
        ]))
        self.loss_func = lambda a, b: torch.mean(torch.abs(a - b))
        self.optimizer = None
        self.n_outs_node = n_outs_node
    

    def forward(self, x):
        x = x.reshape(len(x), -1)
        e = self.lin(x)
        if self.n_outs_node>1:
            return e
        else:
            return e.reshape(-1)
    
    def fit(self, x, e, valid_x, valid_e, model_path, lr=0.0001, batch_size=5120, max_iter=3000, save_every=1000, valid_every=100):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False)
        x, e = torch.asarray(x).to(self.device), torch.asarray(e).to(self.device)
        valid_x, valid_e = torch.asarray(valid_x).to(self.device), torch.asarray(valid_e).to(self.device)
        dataloader = torch.utils.data.DataLoader(Dataset(x, e), batch_size=batch_size, shuffle=True)
        validloader = torch.utils.data.DataLoader(Dataset(valid_x, valid_e), batch_size=batch_size, shuffle=False)
        loss_record = []
        valid_loss_record = []
        for epoch in range(max_iter):
            epoch_loss = 0.
            for batch_x, batch_e in dataloader:
                self.optimizer.zero_grad()
                batch_e_p = self.forward(batch_x)
                loss_e = self.loss_func(batch_e_p, batch_e)
                epoch_loss += loss_e.item()*len(batch_e)
                loss_e.backward()
                self.optimizer.step()
            epoch_loss /= len(e)
            loss_record.append(epoch_loss)
            if (epoch+1)%valid_every==0:
                with torch.no_grad():
                    valid_loss = 0.
                    for batch_x, batch_e in validloader:
                        batch_e_p = self.forward(batch_x)
                        loss_e = self.loss_func(batch_e_p, batch_e)
                        valid_loss += loss_e.item()*len(batch_e)
                    valid_loss /= len(valid_e)
                    valid_loss_record.append(valid_loss)
                    print('epoch', epoch, 'valid_loss', valid_loss)

            if (epoch+1)%save_every==0:
                print(f'saving model at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
    
    def load(self, model_path):
        self.to(self.device)
        chk = torch.load(model_path)
        self.load_state_dict(chk['net'])
        self.optimizer = torch.optim.Adam(self.parameters(), foreach=False)
        self.optimizer.load_state_dict(chk['opt'])

class ModelE(Module):
    def __init__(self, feature_num=1331, in_channels=1, n_outs_node=1, device='cpu', **kwargs):
        super().__init__()
        self.device = device
        self.lin = Sequential(*([
        # Linear(feature_num, 256), ReLU(),
        # Linear(256, 256), ReLU(),
        # Linear(256, 128), ReLU(),
        # Linear(128, 128), ReLU(),
        # Linear(128, 64), ReLU(),
        # Linear(64, n_outs_node),
        Linear(feature_num, 40), ReLU(),
        Linear(40, 256), ReLU(),
        
        Linear(256, 256), ReLU(),
        # Linear(256, 256), ReLU(),
        Linear(256, 256), ReLU(),
        
        Linear(256, 40), ReLU(),
        Linear(40, n_outs_node),
        ]))
        self.loss_func = lambda a, b: torch.mean(torch.abs(a - b))
        self.optimizer = None
        self.n_outs_node = n_outs_node
    
    def forward(self, x, off=False):
        x = x.reshape(len(x), -1)
        e = self.lin(x)
        if self.n_outs_node>1:
            return e*0 if off else e
        else:
            return e.reshape(-1)*0 if off else e.reshape(-1)
        
    def get_batches(self, train_pkls, batch_size, shuffle=True): 
        train_idx = np.arange(len(train_pkls))
        if shuffle:
            np.random.shuffle(train_idx)
        batches = []
        for i in range(0, len(train_pkls), batch_size):
            batches.append(train_pkls[train_idx[pos]] for pos in range(i, min(i+batch_size, len(train_pkls))))
        return batches
    
    def fit(self, train_pkls, valid_pkls, model_path, lr=0.0001, batch_size=3, max_iter=3000, save_every=100, valid_every=50, off=False):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False)
        for pkl in train_pkls+valid_pkls:
            for k in ['x', 'gw', 'rho_base']:
                if (type(pkl[k]) == np.ndarray) or (type(pkl[k][0]) == np.ndarray):
                    pkl[k] = torch.tensor(pkl[k]).to(self.device)
        # dataloader = torch.utils.data.DataLoader(Dataset(train_pkls), batch_size=batch_size, shuffle=True)
        # validloader = torch.utils.data.DataLoader(Dataset(valid_pkls), batch_size=batch_size, shuffle=False)
        loss_record = []
        valid_loss_record = []
        last_loss = np.inf
        for epoch in range(max_iter):
            epoch_loss = 0.
            for pkls in self.get_batches(train_pkls, batch_size):
                self.optimizer.zero_grad()
                loss_e = 0.
                for pkl in pkls:
                    batch_e_p = self.forward(pkl['x'], off=off)
                    loss_e += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                epoch_loss += loss_e.item()
                loss_e.backward()
                self.optimizer.step()
            epoch_loss /= len(train_pkls)
            loss_record.append(epoch_loss*627)
            if (epoch+1)%valid_every==0:
                with torch.no_grad():
                    valid_loss = 0.
                    for pkls in self.get_batches(valid_pkls, batch_size, shuffle=False):
                        loss_e = 0.
                        for pkl in pkls:
                            batch_e_p = self.forward(pkl['x'], off=off)
                            loss_e += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                        valid_loss += loss_e.item()
                    valid_loss /= len(valid_pkls)
                    valid_loss_record.append(valid_loss*627)
            if (epoch+1)%save_every==0 and (epoch_loss+valid_loss)*627<last_loss:
                last_loss = (epoch_loss+valid_loss)*627
                print(f'saving model_E at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
            if (epoch+1)%save_every==0:
                plt.plot(np.arange(1, len(loss_record)+1)[:], loss_record[:], label='train')
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[:], valid_loss_record[:], label='valid', c='r', s=5)
                plt.plot([chk["epoch"], chk["epoch"]], plt.ylim(), '-.', label='chosen epoch')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Mean absolute error of E_tot in kcal/mol')
                plt.savefig(f'{model_path[:-3]}_losses.png')
                plt.close()
    
    def load(self, model_path):
        self.to(self.device)
        chk = torch.load(model_path)
        self.load_state_dict(chk['net'])
        self.optimizer = torch.optim.Adam(self.parameters(), foreach=False)
        self.optimizer.load_state_dict(chk['opt'])

class ModelE_zigzag(ModelE):
    def __init__(self, feature_num=7, in_channels=1, n_outs_node=1, device='cuda:7', **kwargs):
        super().__init__(feature_num, in_channels, n_outs_node, device)
        self.ni = NumInt()
    
    def forward_xc(self, rho, spin, off=False, keep_graph=False):
        if len(rho) != 2:
            rho = 0.5*rho, 0.5*rho
        rho_up, rhox_up, rhoy_up, rhoz_up, tau_up, rho_down, rhox_down, rhoy_down, rhoz_down, tau_down = (*rho[0][:4], rho[0][5], *rho[1][:4], rho[1][5])
        
        guu = rhox_up**2+rhoy_up**2+rhoz_up**2
        gdd = rhox_down**2+rhoy_down**2+rhoz_down**2
        gud = rhox_up*rhox_down+rhoy_up*rhoy_down+rhoz_up*rhoz_down
        
        inp = np.vstack((rho_up, rho_down, guu, gud, gdd, tau_up, tau_down))
        inp = torch.tensor(inp, requires_grad=True).to(self.device)
        
        g = inp[2:5]/((inp[0]+inp[1]+1e-7)**(4/3))/(2*(3*torch.pi)**(1/3))
        t_unif = 3/10*(3*torch.pi**2)**(2/3)*(inp[0]+inp[1]+1e-7)**(5/3)
        t = inp[5:7]/(t_unif+1e-7)
        e = self.forward(torch.vstack((inp[0], inp[1], *g, *t)).T, off=off)
        grad, = torch.autograd.grad((e*(inp[0]+inp[1])).sum(), inp, create_graph=True)
        if spin:
            vrho = grad[0:2].T
            vgamma = grad[2:5].T
            vlapl = torch.zeros((len(rho[0][0]), 2))
            vtau = grad[5:7].T
        else:
            vrho = (grad[0] + grad[1]) / 2
            vgamma = (grad[2] + grad[3] + grad[4]) / 4
            vlapl = torch.zeros(len(rho[0][0]))
            vtau = (grad[5] + grad[6]) / 2
        if not keep_graph:
            v = (vrho.detach().cpu().numpy(), vgamma.detach().cpu().numpy(), vlapl.detach().cpu().numpy(), vtau.detach().cpu().numpy())
            e = e.detach().cpu().numpy()
        else:
            v = (vrho, vgamma, vlapl, vtau)
        return e, v
    
    def fit2(self, train_pkls, valid_pkls, model_path, lr=0.0001, batch_size=3, max_iter=3000, save_every=200, valid_every=100, off=False):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False)
        # for pkl in train_pkls+valid_pkls:
        #     for k in ['gw', 'rho_base', 'phi01', 'phi2', 'dm_base', 'F_base', 'F_target']:
        #         if type(pkl[k]) != torch.Tensor :
        #             pkl[k] = torch.tensor(pkl[k]).to(self.device)
        loss_record = []
        valid_loss_record = []
        last_loss = np.inf
        for epoch in range(max_iter):
            print(epoch)
            epoch_loss_E = 0.
            epoch_loss_F = 0.
            for pkls in self.get_batches(train_pkls, batch_size):
                self.optimizer.zero_grad()
                loss_E = 0.
                loss_F = 0.
                for pkl in pkls:
                    
                    temp = {}
                    for k in ['gw', 'rho_base', 'phi01', 'phi2', 'dm_base', 'F_base', 'F_target']:
                        if type(pkl[k]) != torch.Tensor :
                            temp[k] = torch.tensor(pkl[k]).to(self.device)
                    
                    batch_e_p, (v_model_rho, v_model_gamma, v_model_lapla, v_model_tau) = self.forward_xc(rho=pkl['x'], spin=pkl['spin'], off=off, keep_graph=True)
                    loss_E += torch.abs(torch.sum(batch_e_p*temp['rho_base']*temp['gw'])-(pkl['E_target']-pkl['E_base']))
                    V_pulay = torch.zeros(temp['dm_base'].shape[:-2]+(3,)+temp['dm_base'].shape[-2:]).to(self.device)
                    
                    phi01 = temp['phi01']
                    phi2 = temp['phi2']
                    sep_rho01 = dm2rho01_sep(temp['dm_base'], phi01)

                        
                    F = []
                    if pkl['spin']:
                        dme = contract('aij, aj, aj, akj->aik', pkl['mo_coeff'], pkl['mo_occ'], pkl['mo_energy'], pkl['mo_coeff'])
                        for _, _, start, end in pkl['offset']:
                            
                            V_pulay = - contract('dri,ra,r,rj->adij', phi01[1:,:,start:end], v_model_rho, temp['gw'], phi01[0])
                            
                            V_pulay += - 2*contract('dri,adr,ra,r,prj->apji', phi01[1:], sep_rho01[:,1:], v_model_gamma[:,[0,2]], temp['gw'], phi01[1:,:,start:end])
                            V_pulay += -2*contract('pdri,adr,ra,r,rj->apij', phi2[:,:,:,start:end], sep_rho01[:,1:], v_model_gamma[:,[0,2]], temp['gw'], phi01[0])
                            V_pulay += -contract('dri,adr,r,r,prj->apji', phi01[1:], sep_rho01[[1,0],1:], v_model_gamma[:,1], temp['gw'], phi01[1:,:,start:end])
                            V_pulay += -contract('pdri,adr,r,r,rj->apij', phi2[:,:,:,start:end], sep_rho01[[1,0], 1:], v_model_gamma[:,1], temp['gw'], phi01[0])
                            V_pulay += -0.5*contract('pdri,ra,r,drj->apij', phi2[:,:,:,start:end], v_model_tau, temp['gw'], phi01[1:])
                            mask = np.where( (np.arange(temp['dm_base'].shape[-1])<start)|(np.arange(temp['dm_base'].shape[-1])>=end) , 0., 1.)[:,None]
                            Sterm = contract('pij, aij->p', pkl['dS'], dme*mask)
                            F.append((contract('apij, aij->p', -V_pulay, temp['dm_base'][:,start:end]) + torch.tensor(Sterm).to(self.device) )*2.)
                    else:
                        dme = contract('ij, j, j, kj->ik', pkl['mo_coeff'], pkl['mo_occ'], pkl['mo_energy'], pkl['mo_coeff'])
                        for _, _, start, end in pkl['offset']:
                            V_pulay = - contract('dri,r,r,rj->dij', phi01[1:,:,start:end], v_model_rho, temp['gw'], phi01[0])
                            
                            V_pulay += - 2*contract('dri,dr,r,r,prj->pji', phi01[1:], sep_rho01[1:], v_model_gamma, temp['gw'], phi01[1:,:,start:end])
                            V_pulay += -2*contract('pdri,dr,r,r,rj->pij', phi2[:,:,:,start:end], sep_rho01[1:], v_model_gamma, temp['gw'], phi01[0])
                            V_pulay += -0.5*contract('pdri,r,r,drj->pij', phi2[:,:,:,start:end], v_model_tau, temp['gw'], phi01[1:])
                            Sterm = contract('pij, ij->p', pkl['dS'][:,start:end], dme[start:end])
                            F.append((contract('pij, ij->p', -V_pulay, temp['dm_base'][start:end]) + torch.tensor(Sterm).to(self.device))*2. )

                    loss_F += torch.abs(torch.vstack(F)-(temp['F_target']-temp['F_base'])).mean()
                    
                    del temp
                    # torch.cuda.memory_cached()
                    torch.cuda.memory_reserved()
                    
                epoch_loss_E += loss_E.item()
                epoch_loss_F += loss_F.item()
                (loss_E+loss_F*.5).backward()
                self.optimizer.step()
            epoch_loss_E /= len(train_pkls)
            epoch_loss_F /= len(train_pkls)
            loss_record.append({'E':epoch_loss_E*627, 'F':epoch_loss_F})
            if (epoch+1)%valid_every==0:
                valid_loss_E = 0.
                valid_loss_F = 0.
                for pkls in self.get_batches(valid_pkls, batch_size, shuffle=False):
                    loss_E = 0.
                    loss_F = 0.
                    for pkl in pkls:
                        
                        temp = {}
                        for k in ['gw', 'rho_base', 'phi01', 'phi2', 'dm_base', 'F_base', 'F_target']:
                            if type(pkl[k]) != torch.Tensor :
                                temp[k] = torch.tensor(pkl[k]).to(self.device)
                        
                        batch_e_p, (v_model_rho, v_model_gamma, v_model_lapla, v_model_tau) = self.forward_xc(rho=pkl['x'], spin=pkl['spin'], off=off, keep_graph=True)
                        with torch.no_grad():
                            loss_E += torch.abs(torch.sum(batch_e_p*temp['rho_base']*temp['gw'])-(pkl['E_target']-pkl['E_base']))
                            V_pulay = torch.zeros(temp['dm_base'].shape[:-2]+(3,)+temp['dm_base'].shape[-2:]).to(self.device)
                            
                            phi01 = temp['phi01']
                            phi2 = temp['phi2']
                            sep_rho01 = dm2rho01_sep(temp['dm_base'], phi01)
                            F = []
                            if pkl['spin']:
                                dme = contract('aij, aj, aj, akj->aik', pkl['mo_coeff'], pkl['mo_occ'], pkl['mo_energy'], pkl['mo_coeff'])
                                for _, _, start, end in pkl['offset']:
                                    V_pulay = - contract('dri,ra,r,rj->adij', phi01[1:,:,start:end], v_model_rho, temp['gw'], phi01[0])
                                    
                                    V_pulay += - 2*contract('dri,adr,ra,r,prj->apji', phi01[1:], sep_rho01[:,1:], v_model_gamma[:,[0,2]], temp['gw'], phi01[1:,:,start:end])
                                    V_pulay += -2*contract('pdri,adr,ra,r,rj->apij', phi2[:,:,:,start:end], sep_rho01[:,1:], v_model_gamma[:,[0,2]], temp['gw'], phi01[0])
                                    V_pulay += -contract('dri,adr,r,r,prj->apji', phi01[1:], sep_rho01[[1,0],1:], v_model_gamma[:,1], temp['gw'], phi01[1:,:,start:end])
                                    V_pulay += -contract('pdri,adr,r,r,rj->apij', phi2[:,:,:,start:end], sep_rho01[[1,0], 1:], v_model_gamma[:,1], temp['gw'], phi01[0])
                                    V_pulay += -0.5*contract('pdri,ra,r,drj->apij', phi2[:,:,:,start:end], v_model_tau, temp['gw'], phi01[1:])
                                    mask = np.where( (np.arange(temp['dm_base'].shape[-1])<start)|(np.arange(temp['dm_base'].shape[-1])>=end) , 0., 1.)[:,None]
                                    Sterm = contract('pij, aij->p', pkl['dS'], dme*mask)
                                    F.append((contract('apij, aij->p', -V_pulay, temp['dm_base'][:,start:end]) + torch.tensor(Sterm).to(self.device) )*2.)
                            else:
                                dme = contract('ij, j, j, kj->ik', pkl['mo_coeff'], pkl['mo_occ'], pkl['mo_energy'], pkl['mo_coeff'])
                                for _, _, start, end in pkl['offset']:
                                    V_pulay = - contract('dri,r,r,rj->dij', phi01[1:,:,start:end], v_model_rho, temp['gw'], phi01[0])
                                    
                                    V_pulay += - 2*contract('dri,dr,r,r,prj->pji', phi01[1:], sep_rho01[1:], v_model_gamma, temp['gw'], phi01[1:,:,start:end])
                                    V_pulay += -2*contract('pdri,dr,r,r,rj->pij', phi2[:,:,:,start:end], sep_rho01[1:], v_model_gamma, temp['gw'], phi01[0])
                                    V_pulay += -0.5*contract('pdri,r,r,drj->pij', phi2[:,:,:,start:end], v_model_tau, temp['gw'], phi01[1:])
                                    Sterm = contract('pij, ij->p', pkl['dS'][:,start:end], dme[start:end])
                                    F.append((contract('pij, ij->p', -V_pulay, temp['dm_base'][start:end]) + torch.tensor(Sterm).to(self.device))*2. )

                            loss_F += torch.abs(torch.vstack(F)-(temp['F_target']-temp['F_base'])).sum()
                            
                    valid_loss_E += loss_E.item()
                    valid_loss_F += loss_F.item()
                    
                valid_loss_E /= len(valid_pkls)
                valid_loss_F /= len(valid_pkls)
                valid_loss_record.append({'E':valid_loss_E*627, 'F':valid_loss_F})
            if (epoch+1)%save_every==0 and (epoch_loss_E+valid_loss_E)*627<last_loss:
                last_loss = (epoch_loss_E+valid_loss_E)*627
                print(f'saving model_E at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
            if (epoch+1)%save_every==0:
                plt.plot(np.arange(1, len(loss_record)+1)[200:], [itm['E'] for itm in loss_record[200:]], label='train E', alpha=0.1)
                plt.plot(np.arange(1, len(loss_record)+1)[200:], [itm['F'] for itm in loss_record[200:]], label='train F', alpha=0.1)
                
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[2:], [itm['E'] for itm in valid_loss_record][2:], label='valid E', c='r', s=5)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[2:], [itm['F'] for itm in valid_loss_record][2:], label='valid F', c='green', s=5)
                plt.plot([chk["epoch"], chk["epoch"]], plt.ylim(), '-.', label='chosen epoch')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Mean absolute error of E_tot in kcal/mol and F in a. u.')
                plt.savefig(f'{model_path[:-3]}_losses.png')
                plt.close()


    def eval_xc(self, xc_code='b3lypg', rho=None, spin=1, relativity=0, deriv=2, omega=None, verbose=None):
        e_b3, v_b3 = self.ni.eval_xc(xc_code, rho, spin, relativity=relativity, deriv=deriv, omega=omega, verbose=verbose)[:2]
        v_b3_rho, v_gamma_b3 = v_b3[0], v_b3[1]
        e_model, (v_model_rho, v_model_gamma, v_model_lapla, v_model_tau) = self.forward_xc(rho=rho, spin=spin, off=self.off)
        return e_b3+e_model, (v_b3_rho+v_model_rho, v_gamma_b3+v_model_gamma, v_model_lapla, v_model_tau), None, None

    def scf(self, data, basis, grid_level=3, hyb=0.2, off=False, dm_target=None, xc='b3lypg', xctype='MGGA'):
        self.off = off
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in 
            zip(data['atoms_charges'], data['atoms_coords'])], basis=basis, 
                    spin=data['spin'], charge=data['charge'], unit='bohr')
        # self.nu = mol.intor('int1e_grids_sph', grids=data['gc'])
        if dm_target is not None:
            df = myKS(mol, dm_target=dm_target)
        else:
            df = dft.KS(mol)
        df = myKS(mol, dm_target=dm_target)
        df.xc = xc
        df.grids.level = grid_level
        df.define_xc_(self.eval_xc, xctype=xctype, hyb=hyb)
        E = df.kernel()
        dm = df.make_rdm1()
        if 'phi01' in data:
            rho = dm2rho(dm, data['phi01'][0])
            dipole = cal_dipole(rho, data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
            F = -guks(df).kernel() if data['spin'] else -gks(df).kernel()
            return {'dm_scf': dm, 'E_scf': E, 'rho_scf':rho, 'dipole_scf':dipole, 'F_scf':F, 'converged': df.converged}
        return {'dm_scf': dm, 'E_scf': E, 'converged': df.converged}

class ModelE_pbe(ModelE):
    def __init__(self, feature_num=5, in_channels=1, n_outs_node=1, device='cuda:7', **kwargs):
        super().__init__(feature_num, in_channels, n_outs_node, device)

        self.device = device
        self.lin = Sequential(*([
        # Linear(feature_num, 256), ReLU(),
        # Linear(256, 256), ReLU(),
        # Linear(256, 128), ReLU(),
        # Linear(128, 128), ReLU(),
        # Linear(128, 64), ReLU(),
        # Linear(64, n_outs_node),
        Linear(feature_num, 256), ReLU(),
        Linear(256, 256), ReLU(),
        
        Linear(256, 256), ReLU(),
        Linear(256, 256), ReLU(),
        Linear(256, 256), ReLU(),
        
        Linear(256, 256), ReLU(),
        Linear(256, n_outs_node),
        ]))
        self.loss_func = lambda a, b: torch.mean(torch.abs(a - b))
        self.optimizer = None
        self.n_outs_node = n_outs_node


        self.ni = NumInt()
        self.to(self.device)
    
    def forward_xc(self, rho, spin, off=False, keep_graph=False):
        if len(rho) != 2:
            rho = 0.5*rho, 0.5*rho
        rho_up, rhox_up, rhoy_up, rhoz_up, rho_down, rhox_down, rhoy_down, rhoz_down = (*rho[0][:4], *rho[1][:4])
        
        guu = rhox_up**2+rhoy_up**2+rhoz_up**2
        gdd = rhox_down**2+rhoy_down**2+rhoz_down**2
        gud = rhox_up*rhox_down+rhoy_up*rhoy_down+rhoz_up*rhoz_down
        
        inp = np.vstack((rho_up, rho_down, guu, gud, gdd))
        inp = torch.tensor(inp, requires_grad=True).to(self.device)
        
        g = inp[2:5]/((inp[0]+inp[1]+1e-7)**(4/3))/(2*(3*torch.pi)**(1/3))
        e = self.forward(torch.vstack((inp[0], inp[1], *g,)).T, off=off)
        grad, = torch.autograd.grad((e*(inp[0]+inp[1])).sum(), inp, create_graph=True)
        if spin:
            vrho = grad[0:2].T
            vgamma = grad[2:5].T
            # vlapl = torch.zeros((len(rho[0][0]), 2))
        else:
            vrho = (grad[0] + grad[1]) / 2
            vgamma = (grad[2] + grad[3] + grad[4]) / 4
            # vlapl = torch.zeros(len(rho[0][0]))
        if not keep_graph:
            v = (vrho.detach().cpu().numpy(), vgamma.detach().cpu().numpy())
            e = e.detach().cpu().numpy()
        else:
            v = (vrho, vgamma)
        return e, v
    
    def fit(self, train_pkls, valid_pkls, model_path, lr=0.0001, batch_size=3, max_iter=3000, save_every=200, valid_every=100, off=False):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False)
        for pkl in train_pkls+valid_pkls:
            for k in ['gw', 'rho_base', 'rho_target', 'phi01', 'rho01', 'E_base', 'E_target', 'F0', 'S', 'J']:
                if type(pkl[k]) != torch.Tensor :
                    pkl[k] = torch.tensor(np.array(pkl[k])).to(self.device)
        loss_record = []
        valid_loss_record = []
        last_loss = np.inf
        for epoch in range(max_iter):
            print(epoch)
            epoch_loss_E = 0.
            epoch_loss_rho = 0.
            for pkls in self.get_batches(train_pkls, batch_size):
                self.optimizer.zero_grad()
                loss_E = 0.
                loss_rho = 0.
                for pkl in pkls:
                    if 'e_pbe' not in pkl:
                        e_pbe, (v_pbe_rho, v_pbe_gamma) = self.ni.eval_xc('pbe', rho=pkl['x'] if pkl['spin'] else (pkl['x'][0]+pkl['x'][1]), spin=pkl['spin'], deriv=1)[:2]
                        pkl['e_pbe'] = torch.tensor(e_pbe).to(self.device)
                        pkl['v_pbe_rho'] = torch.tensor(v_pbe_rho).to(self.device)
                        pkl['v_pbe_gamma'] = torch.tensor(v_pbe_gamma).to(self.device)
                    batch_e_p, (v_model_rho, v_model_gamma) = self.forward_xc(rho=pkl['x'], spin=pkl['spin'], off=off, keep_graph=True)
                    batch_e_p = batch_e_p + pkl['e_pbe']
                    v_model_rho = v_model_rho + pkl['v_pbe_rho']
                    v_model_gamma = v_model_gamma + pkl['v_pbe_gamma']
                    # loss_E += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                    Vxc = v2V(v_model_rho, v_model_gamma, pkl['phi01'], pkl['gw'], pkl['rho01'])
                    if len(Vxc) == 2:
                        F = pkl['F0'] + Vxc[0], pkl['F0'] + Vxc[1]
                    else:
                        F = pkl['F0'] + Vxc
                    n_down = (pkl['n_elec'] - pkl['spin'])//2
                    n_up = pkl['n_elec'] - n_down
                    dm_new = F2dm(F, pkl['S'], n_up, n_down, self.device)
                    rho_new = dm2rho(dm_new, pkl['phi01'][0])
                    if np.random.random()>.5:
                        E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*pkl['rho_base']*pkl['gw']) + pkl['E_N']
                    else:
                        E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*rho_new*pkl['gw']) + pkl['E_N']
                    loss_rho += torch.abs(rho_new-pkl['rho_target']).mean()
                    loss_E += torch.abs(E_new-pkl['E_target'])
                    
                epoch_loss_E += loss_E.item()
                epoch_loss_rho += loss_rho.item()
                (loss_E+loss_rho).backward()
                num_nan = 0
                for p in self.parameters():
                    num_nan += torch.sum(torch.isnan(p.grad.data))
                    # num_inf += torch.sum(torch.isinf(p.grad.data))
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=np.random.randn(), posinf=np.random.randn(), neginf=np.random.randn())
                    p.grad.data = torch.clip(p.grad.data, min=-10., max=10.)
                if num_nan:
                    print('epoch: %d num_nan'%epoch, num_nan)
                self.optimizer.step()
            epoch_loss_E /= len(train_pkls)
            epoch_loss_rho /= len(train_pkls)
            loss_record.append({'E':epoch_loss_E*1, 'rho':epoch_loss_rho, 'tot': epoch_loss_E+epoch_loss_rho})

            if (epoch+1)%valid_every==0:
                valid_loss_E = 0.
                valid_loss_rho = 0.
                for pkls in self.get_batches(valid_pkls, batch_size, shuffle=False):
                    loss_E = 0.
                    loss_rho = 0.
                    for pkl in pkls:
                        if 'e_pbe' not in pkl:
                            e_pbe, (v_pbe_rho, v_pbe_gamma) = self.ni.eval_xc('pbe', rho=pkl['x'] if pkl['spin'] else (pkl['x'][0]+pkl['x'][1]), spin=pkl['spin'], deriv=1)[:2]
                            pkl['e_pbe'] = torch.tensor(e_pbe).to(self.device)
                            pkl['v_pbe_rho'] = torch.tensor(v_pbe_rho).to(self.device)
                            pkl['v_pbe_gamma'] = torch.tensor(v_pbe_gamma).to(self.device)
                        batch_e_p, (v_model_rho, v_model_gamma) = self.forward_xc(rho=pkl['x'], spin=pkl['spin'], off=off, keep_graph=True)
                        batch_e_p = batch_e_p.detach()
                        v_model_rho = v_model_rho.detach()
                        v_model_gamma = v_model_gamma.detach()
                        batch_e_p = batch_e_p + pkl['e_pbe']
                        v_model_rho = v_model_rho + pkl['v_pbe_rho']
                        v_model_gamma = v_model_gamma + pkl['v_pbe_gamma']
                        with torch.no_grad():
                            # loss_E += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                            Vxc = v2V(v_model_rho, v_model_gamma, pkl['phi01'], pkl['gw'], pkl['rho01'])
                            if len(Vxc) == 2:
                                F = pkl['F0'] + Vxc[0], pkl['F0'] + Vxc[1]
                            else:
                                F = pkl['F0'] + Vxc
                            n_down = (pkl['n_elec'] - pkl['spin'])//2
                            n_up = pkl['n_elec'] - n_down
                            dm_new = F2dm(F, pkl['S'], n_up, n_down, self.device)
                            rho_new = dm2rho(dm_new, pkl['phi01'][0])
                            if np.random.random()>.5:
                                E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*pkl['rho_base']*pkl['gw']) + pkl['E_N']
                            else:
                                E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*rho_new*pkl['gw']) + pkl['E_N']
                            loss_rho += torch.abs(rho_new-pkl['rho_target']).mean()
                            loss_E += torch.abs(E_new-pkl['E_target'])
                    valid_loss_E += loss_E.item()
                    valid_loss_rho += loss_rho.item()
                    
                valid_loss_E /= len(valid_pkls)
                valid_loss_rho /= len(valid_pkls)
                valid_loss_record.append({'E':valid_loss_E*1, 'rho':valid_loss_rho, 'tot':valid_loss_E+valid_loss_rho})
            if (epoch+1)%save_every==0 and (epoch_loss_E+valid_loss_E+epoch_loss_rho+epoch_loss_rho)<last_loss:
                last_loss = epoch_loss_E+valid_loss_E+epoch_loss_rho+epoch_loss_rho
                print(f'saving model_E at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
            if (epoch+1)%save_every==0:
                plt.plot(np.arange(1, len(loss_record)+1)[200:], [itm['E'] for itm in loss_record[200:]], label='train E', alpha=0.1)
                plt.plot(np.arange(1, len(loss_record)+1)[200:], [itm['rho'] for itm in loss_record[200:]], label='train rho', alpha=0.1)
                plt.plot(np.arange(1, len(loss_record)+1)[200:], [itm['tot'] for itm in loss_record[200:]], label='train tot', alpha=0.1)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[2:], [itm['E'] for itm in valid_loss_record][2:], label='valid E', c='r', s=5)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[2:], [itm['rho'] for itm in valid_loss_record][2:], label='valid rho', c='green', s=5)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[2:], [itm['tot'] for itm in valid_loss_record][2:], label='valid tot', c='k', s=5)
                plt.plot([chk["epoch"], chk["epoch"]], plt.ylim(), '-.', label='chosen epoch')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Mean absolute error of E_tot and rho in a. u.')
                plt.savefig(f'{model_path[:-3]}_losses.png')
                plt.close()


    def eval_xc(self, xc_code='pbe', rho=None, spin=1, relativity=0, deriv=2, omega=None, verbose=None):
        e_b3, v_b3 = self.ni.eval_xc(xc_code, rho, spin, relativity=relativity, deriv=deriv, omega=omega, verbose=verbose)[:2]
        v_b3_rho, v_gamma_b3 = v_b3[0], v_b3[1]
        e_model, (v_model_rho, v_model_gamma) = self.forward_xc(rho=rho, spin=spin, off=self.off)
        return e_b3+e_model, (v_b3_rho+v_model_rho, v_gamma_b3+v_model_gamma, None, None), None, None

    def scf(self, data, basis, grid_level=3, hyb=0, off=False, dm_target=None, xc='pbe', xctype='GGA'):
        self.off = off
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in 
            zip(data['atoms_charges'], data['atoms_coords'])], basis=basis, 
                    spin=data['spin'], charge=data['charge'], unit='bohr')
        # self.nu = mol.intor('int1e_grids_sph', grids=data['gc'])
        if dm_target is not None:
            df = myKS(mol, dm_target=dm_target)
        else:
            df = dft.KS(mol)
        df = myKS(mol, dm_target=dm_target)
        df.xc = xc
        df.grids.level = grid_level
        df.define_xc_(self.eval_xc, xctype=xctype, hyb=hyb)
        E = df.kernel()
        dm = df.make_rdm1()
        if 'phi01' in data:
            rho = dm2rho(dm, data['phi01'][0])
            dipole = cal_dipole(rho, data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
            F = -guks(df).kernel() if data['spin'] else -gks(df).kernel()
            return {'dm_scf': dm, 'E_scf': E, 'rho_scf':rho, 'dipole_scf':dipole, 'F_scf':F, 'converged': df.converged}
        return {'dm_scf': dm, 'E_scf': E, 'converged': df.converged}

class ModelE_b3(ModelE_pbe):
    def __init__(self, feature_num=3, in_channels=1, n_outs_node=2, device='cuda:7', **kwargs):
        super().__init__(feature_num, in_channels, n_outs_node, device)
    
    def forward_xc(self, rho, spin, off=False, keep_graph=False):
        if len(rho) != 2:
            rho = 0.5*rho, 0.5*rho
        rho_up, rhox_up, rhoy_up, rhoz_up, rho_down, rhox_down, rhoy_down, rhoz_down = (*rho[0][:4], *rho[1][:4])
        
        guu = rhox_up**2+rhoy_up**2+rhoz_up**2
        gdd = rhox_down**2+rhoy_down**2+rhoz_down**2
        gud = rhox_up*rhox_down+rhoy_up*rhoy_down+rhoz_up*rhoz_down
        
        inp = np.vstack((rho_up, rho_down, guu, gud, gdd))
        inp = torch.tensor(inp, requires_grad=True).to(self.device)
        
        t=torch.empty((inp.shape[1],3), device=self.device)
        unif=(inp[0]+inp[1]+1e-7)**(1.0/3)
        t[:,0]=unif
        t[:,1]=((1+torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(4.0/3)+(1-torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(4.0/3))*0.5
        t[:,2]=((inp[2]+inp[4]+2*inp[3])**0.5+1e-7)/unif**4
        # ds=(1+torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(5.0/3)+(1-torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(5.0/3)
        # t[:,3]=(inp[5]+inp[6]+1e-7)/(unif**5*ds)
        
        res = self.forward(torch.log(t), off=off)
        e, ec = res[:,0], res[:,1]
        grad, = torch.autograd.grad((e*(inp[0]+inp[1])).sum(), inp, create_graph=True)
        if spin:
            vrho = grad[0:2].T
            vgamma = grad[2:5].T
            # vlapl = torch.zeros((len(rho[0][0]), 2))
        else:
            vrho = (grad[0] + grad[1]) / 2
            vgamma = (grad[2] + grad[3] + grad[4]) / 4
            # vlapl = torch.zeros(len(rho[0][0]))
        if not keep_graph:
            v = (vrho.detach().cpu().numpy(), vgamma.detach().cpu().numpy())
            e = e.detach().cpu().numpy() + ec.detach().cpu().numpy()
        else:
            v = (vrho, vgamma)
            e = e + ec
        return e, v

    def fit(self, train_pkls, valid_pkls, model_path, lr=0.0001, batch_size=3, max_iter=3000, save_every=200, valid_every=100, off=False):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False)
        for pkl in train_pkls+valid_pkls:
            pkl['rho01_old'] = np.copy(pkl['rho01'])
            pkl['F0_old'] = np.copy(pkl['F0'])
            pkl['J_old'] = np.copy(pkl['J'])
            pkl['x_old'] = np.copy(pkl['x'])
            pkl['K_old'] = np.copy(pkl['K'])
            pkl['rho_target_old'] = np.copy(pkl['rho_target'])
            pkl['E_target_old'] = pkl['E_target']
            pkl['phi01_old'] = np.copy(pkl['phi01'])
            for k in ['gc', 'gw', 'rho_base', 'rho_target', 'phi01', 'rho01', 'E_base', 'E_target', 'F0', 'S', 'J', 'K']:
                if type(pkl[k]) != torch.Tensor :
                    pkl[k] = torch.tensor(np.array(pkl[k])).to(self.device)
        loss_record = []
        valid_loss_record = []
        last_loss = np.inf
        scf_every = max_iter//5

        for epoch in range(max_iter):
            print(epoch)
            if ((epoch+1)%scf_every==0) and (epoch+1<max_iter):
                # do SCF and update x, J, K, F0, rho_target
                print('redo scf')
                for pkl in train_pkls+valid_pkls:
                    res = self.scf(pkl, pkl['basis'], grid_level=3, xc='b3lypg', xctype='GGA', hyb=.2, init=pkl['dm_base'])
                    pkl['E_target_last'] = pkl['E_target_old']
                    rho01 = dm2rho01_sep(res['dm_scf'], pkl['phi01_old'])
                    pkl['x'] = rho01
                    F0 = pkl['T'] + pkl['V_ext'] + dm2J(res['dm_scf'], pkl['itg_2e'])
                    pkl['F0'] = torch.tensor(F0).to(self.device)
                    pkl['J'] = torch.tensor(F0 - pkl['T'] - pkl['V_ext']).to(self.device)
                    K = np.array([dm2K(res['dm_scf'][0], pkl['itg_2e']), dm2K(res['dm_scf'][1], pkl['itg_2e'])]) if pkl['spin'] else dm2K(res['dm_scf'], pkl['itg_2e'])
                    pkl['K'] = torch.tensor(K).to(self.device)
                    pkl['rho_target'] = torch.tensor(dm2rho(res['dm_scf'], pkl['phi01_old'][0])).to(self.device)# TODO maybe not update?
            
            epoch_loss_E = 0.
            epoch_loss_rho = 0.
            for pkls in self.get_batches(train_pkls, batch_size):
                self.optimizer.zero_grad()
                loss_E = 0.
                loss_rho = 0.
                for pkl in pkls:
                    random = np.random.random()<(1-epoch/max_iter)
                    # if random:
                    pkl['E_target'] = pkl['E_target_old']
                    pkl['rho_target'] = torch.tensor(pkl['rho_target_old']).to(self.device)
                    if 'e_b3' not in pkl:
                        e_b3, (v_b3_rho, v_b3_gamma) = self.ni.eval_xc('b3lypg', rho=pkl['x'] if pkl['spin'] else (pkl['x'][0]+pkl['x'][1]), spin=pkl['spin'], deriv=1)[:2]
                        pkl['e_b3'] = torch.tensor(e_b3).to(self.device)
                        pkl['v_b3_rho'] = torch.tensor(v_b3_rho).to(self.device)
                        pkl['v_b3_gamma'] = torch.tensor(v_b3_gamma).to(self.device)
                    batch_e_p, (v_model_rho, v_model_gamma) = self.forward_xc(rho=pkl['x'], spin=pkl['spin'], off=off, keep_graph=True)
                    batch_e_p = batch_e_p + pkl['e_b3']
                    v_model_rho = v_model_rho + pkl['v_b3_rho']
                    v_model_gamma = v_model_gamma + pkl['v_b3_gamma']
                    # loss_E += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                    Vxc = v2V(v_model_rho, v_model_gamma, pkl['phi01'], pkl['gw'], pkl['rho01'])

                    if len(Vxc) == 2:
                        F = pkl['F0'] + Vxc[0] - 0.2*pkl['K'][0], pkl['F0'] + Vxc[1] - 0.2*pkl['K'][1]
                    else:
                        F = pkl['F0'] + Vxc - 0.5*0.2*pkl['K']
                    n_down = (pkl['n_elec'] - pkl['spin'])//2
                    n_up = pkl['n_elec'] - n_down
                    dm_new = F2dm(F, pkl['S'], n_up, n_down, self.device)
                    rho_new = dm2rho(dm_new, pkl['phi01'][0])

                    if len(Vxc) == 2:
                        E_K = 0.5 * 0.2 * (dm2E1e(dm_new[0], pkl['K'][0]) + dm2E1e(dm_new[1], pkl['K'][1]))
                    else:
                        E_K = 0.25 * 0.2 * dm2E1e(dm_new, pkl['K'])
                    # if np.random.random()>.5:
                    #     E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*pkl['rho_base']*pkl['gw']) + pkl['E_N'] -E_K
                    # else:
                    E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*rho_new*pkl['gw']) + pkl['E_N'] - E_K
                    # E_new = dm2E1e(dm_new, pkl['T']) + pkl['E_ext_J'] + torch.sum(batch_e_p*pkl['rho_target']*pkl['gw']) + pkl['E_N'] - E_K
                    # loss_rho += torch.abs((rho_new-pkl['rho_target'])*pkl['gw']).mean()
                    loss_rho += (((rho_new-pkl['rho_target'])**2)*pkl['gw']).sum()
                    # loss_rho += torch.abs(rho_new-pkl['rho_target']).mean()
                    # loss_rho += torch.linalg.norm(torch.sum((rho_new-pkl['rho_target'])*pkl['gc'].T*pkl['gw'], axis=1))#dipole
                    
                    loss_E += torch.abs(E_new-pkl['E_target'])

                    # else:
                    #TODO
                    # K = get_K(F0.detach().cpu().numpy(), Vxc.detach().cpu().numpy(), pkl['itg_2e'], pkl['dm_base'])
                    # pkl['K'] = torch.tensor(K, device=self.device)
                    pkl['E_target'] = pkl.get('E_target_last', pkl['E_target_old'])
                    pkl['rho_target'] = pkl.get('rho_target_last', torch.tensor(pkl['rho_target_old']).to(self.device))
                    res = self.scf(pkl, pkl['basis'], grid_level=3, xc='b3lypg', xctype='GGA', hyb=.2, init=pkl['dm_base'])
                    F0 = pkl['T'] + pkl['V_ext'] + dm2J(res['dm_scf'], pkl['itg_2e'])
                    pkl['F0'] = torch.tensor(F0).to(self.device)
                    pkl['J'] = torch.tensor(F0 - pkl['T'] - pkl['V_ext']).to(self.device)
                    K = np.array([dm2K(res['dm_scf'][0], pkl['itg_2e']), dm2K(res['dm_scf'][1], pkl['itg_2e'])]) if pkl['spin'] else dm2K(res['dm_scf'], pkl['itg_2e'])
                    pkl['K'] = torch.tensor(K).to(self.device)
                    rho01 = dm2rho01_sep(res['dm_scf'], pkl['phi01_old'])
                    e_b3, (v_b3_rho, v_b3_gamma) = self.ni.eval_xc('b3lypg', rho=rho01, spin=pkl['spin'], deriv=1)[:2]
                    batch_e_p, (v_model_rho, v_model_gamma) = self.forward_xc(rho=rho01, spin=pkl['spin'], off=off, keep_graph=True)
                    batch_e_p = batch_e_p + torch.tensor(e_b3).to(self.device)
                    v_model_rho = v_model_rho + torch.tensor(v_b3_rho).to(self.device)
                    v_model_gamma = v_model_gamma + torch.tensor(v_b3_gamma).to(self.device)
                    Vxc = v2V(v_model_rho, v_model_gamma, pkl['phi01'], pkl['gw'], torch.tensor(rho01).to(self.device))

                    if len(Vxc) == 2:
                        F = pkl['F0'] + Vxc[0] - 0.2*pkl['K'][0], pkl['F0'] + Vxc[1] - 0.2*pkl['K'][1]
                    else:
                        F = pkl['F0'] + Vxc - 0.5*0.2*pkl['K']
                    n_down = (pkl['n_elec'] - pkl['spin'])//2
                    n_up = pkl['n_elec'] - n_down
                    dm_new = F2dm(F, pkl['S'], n_up, n_down, self.device)
                    rho_new = dm2rho(dm_new, pkl['phi01'][0])

                    if len(Vxc) == 2:
                        E_K = 0.5 * 0.2 * (dm2E1e(dm_new[0], pkl['K'][0]) + dm2E1e(dm_new[1], pkl['K'][1]))
                    else:
                        E_K = 0.25 * 0.2 * dm2E1e(dm_new, pkl['K'])
                    # if np.random.random()>.5:
                    #     E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*pkl['rho_base']*pkl['gw']) + pkl['E_N'] -E_K
                    # else:
                    E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*rho_new*pkl['gw']) + pkl['E_N'] - E_K
                    # E_new = dm2E1e(dm_new, pkl['T']) + pkl['E_ext_J'] + torch.sum(batch_e_p*pkl['rho_target']*pkl['gw']) + pkl['E_N'] - E_K
                    # loss_rho += torch.abs((rho_new-pkl['rho_target'])*pkl['gw']).mean()
                    loss_rho += (((rho_new-pkl['rho_target'])**2)*pkl['gw']).sum()
                    # loss_rho += torch.abs(rho_new-pkl['rho_target']).mean()
                    # loss_rho += torch.linalg.norm(torch.sum((rho_new-pkl['rho_target'])*pkl['gc'].T*pkl['gw'], axis=1))#dipole
                    
                    loss_E += torch.abs(E_new-pkl['E_target'])
                    # pkl['dm_new'] = (dm_new[0].detach().cpu().numpy(), dm_new[1].detach().cpu().numpy()) if pkl['spin'] else dm_new.detach().cpu().numpy()

                    pkl['E_target_last'] = E_new.detach()
                    pkl['rho_target_last'] = rho_new.detach()
                epoch_loss_E += loss_E.item()
                epoch_loss_rho += loss_rho.item()
                (loss_E+loss_rho*100).backward()# pre2 1.0, pre3 0.1,#pre4 0.1 no clip, pre5 5. no clip, pre6.no clip rho*gw*1e5 inp rhocc, pre8 update K using dm_new
                # pre9 +loss dipole, 
                # pre10 +loss dipole 4000 epoches clip no K update, 
                # pre11 +loss dipole 4000 epoches clip K update,
                # ->pre12 +loss dipole 4000 epoches clip K rho01 and Vxc update, 
                # pre13 loss rho no gw +loss dipole 4000 epoches clip K rho01 and Vxc update
                # pre14 to pre9, follow nagai's 3 features
                # pre15 to pre9, follow nagai's 4 features
                num_nan = 0
                for p in self.parameters():
                    num_nan += torch.sum(torch.isnan(p.grad.data))
                    # num_inf += torch.sum(torch.isinf(p.grad.data))
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=np.random.randn()*0., posinf=np.random.randn()*0., neginf=np.random.randn()*0.)
                    # p.grad.data = torch.clip(p.grad.data, min=-10., max=10.)
                if num_nan:
                    print('epoch: %d num_nan'%epoch, num_nan, pkl['fn'])
                    print((pkl['E_base']-pkl['E_target']).detach(), (((pkl['rho_base']-pkl['rho_target'])**2)*pkl['gw']).sum().detach() )
                self.optimizer.step()
                
                # for pkl in pkls:    
                #     if epoch>400:
                #         dm = pkl['dm_new']
                #         K = np.array([dm2K(dm[0], pkl['itg_2e']), dm2K(dm[1], pkl['itg_2e'])]) if pkl['spin'] else dm2K(dm, pkl['itg_2e'])
                #         pkl['K'] = torch.tensor(K).to(self.device)
                        
                #         if np.random.random()>.5:
                #             #TODO update rho01, F0, J, K, x using dm_new
                #             pkl['rho01'] = torch.tensor(dm2rho01_sep(dm, pkl['phi01'])).to(self.device)
                #     # #         pkl['rho_base'] = torch.tensor(dm2rho(dm, pkl['phi01'][0])).to(self.device)
                #             F0 = pkl['T'] + pkl['V_ext'] + dm2J(dm, pkl['itg_2e'])
                #             pkl['F0'] = torch.tensor(F0).to(self.device)
                #             pkl['J'] = torch.tensor(F0 - pkl['T'] - pkl['V_ext']).to(self.device)
                #             if len(dm) != 2:
                #                 dm = 0.5*dm, 0.5*dm
                #             assert len(dm) == 2
                #             pkl['x'] = dm2rho01(dm[0], pkl['phi01']), dm2rho01(dm[1], pkl['phi01'])
                #             pkl['E_target'] = E_new.detach()
                #             pkl['rho_target'] = rho_new.detach()
                #         else:
                #             pkl['rho01'] = torch.tensor(pkl['rho01_old'], self.device)
                #             pkl['F0'] = torch.tensor(pkl['F0_old'], self.device)
                #             pkl['J'] = torch.tensor(pkl['J_old']).to(self.device)
                #             pkl['x'] = torch.tensor(pkl['x_old']).to(self.device)
                #             pkl['E_target'] = pkl['E_target_old']
                #             pkl['rho_target'] = torch.tensor(pkl['rho_target_old']).to(self.device)

            epoch_loss_E /= len(train_pkls)
            epoch_loss_rho /= len(train_pkls)
            loss_record.append({'E':epoch_loss_E*1, 'rho':epoch_loss_rho, 'tot': epoch_loss_E+epoch_loss_rho})
            print('loss E', epoch_loss_E, 'loss rho', epoch_loss_rho)
            if (epoch+1)%valid_every==0:
                valid_loss_E = 0.
                valid_loss_rho = 0.
                for pkls in self.get_batches(valid_pkls, batch_size, shuffle=False):
                    loss_E = 0.
                    loss_rho = 0.
                    for pkl in pkls:
                        if 'e_b3' not in pkl:
                            e_b3, (v_b3_rho, v_b3_gamma) = self.ni.eval_xc('b3lypg', rho=pkl['x'] if pkl['spin'] else (pkl['x'][0]+pkl['x'][1]), spin=pkl['spin'], deriv=1)[:2]
                            pkl['e_b3'] = torch.tensor(e_b3).to(self.device)
                            pkl['v_b3_rho'] = torch.tensor(v_b3_rho).to(self.device)
                            pkl['v_b3_gamma'] = torch.tensor(v_b3_gamma).to(self.device)
                        batch_e_p, (v_model_rho, v_model_gamma) = self.forward_xc(rho=pkl['x'], spin=pkl['spin'], off=off, keep_graph=True)
                        batch_e_p = batch_e_p.detach()
                        v_model_rho = v_model_rho.detach()
                        v_model_gamma = v_model_gamma.detach()
                        batch_e_p = batch_e_p + pkl['e_b3']
                        v_model_rho = v_model_rho + pkl['v_b3_rho']
                        v_model_gamma = v_model_gamma + pkl['v_b3_gamma']
                        with torch.no_grad():
                            # loss_E += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                            Vxc = v2V(v_model_rho, v_model_gamma, pkl['phi01'], pkl['gw'], pkl['rho01'])
                            if len(Vxc) == 2:
                                F = pkl['F0'] + Vxc[0] - 0.2*pkl['K'][0], pkl['F0'] + Vxc[1] - 0.2*pkl['K'][1]
                            else:
                                F = pkl['F0'] + Vxc - 0.5*0.2*pkl['K']
                            n_down = (pkl['n_elec'] - pkl['spin'])//2
                            n_up = pkl['n_elec'] - n_down
                            dm_new = F2dm(F, pkl['S'], n_up, n_down, self.device)
                            rho_new = dm2rho(dm_new, pkl['phi01'][0])
                            if len(Vxc) == 2:
                                E_K = 0.5 * 0.2 * (dm2E1e(dm_new[0], pkl['K'][0]) + dm2E1e(dm_new[1], pkl['K'][1]))
                            else:
                                E_K = 0.25 * 0.2 * dm2E1e(dm_new, pkl['K'])
                            # if np.random.random()>.5:
                            #     E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*pkl['rho_base']*pkl['gw']) + pkl['E_N'] -E_K
                            # else:
                            E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*rho_new*pkl['gw']) + pkl['E_N'] - E_K
                            # E_new = dm2E1e(dm_new, pkl['T']) + pkl['E_ext_J'] + torch.sum(batch_e_p*pkl['rho_target']*pkl['gw']) + pkl['E_N'] - E_K
                            # loss_rho += torch.abs((rho_new-pkl['rho_target'])*pkl['gw']).mean()
                            loss_rho += (((rho_new-pkl['rho_target'])**2)*pkl['gw']).sum()
                            # loss_rho += torch.abs(rho_new-pkl['rho_target']).mean()
                            # loss_rho += torch.linalg.norm(torch.sum((rho_new-pkl['rho_target'])*pkl['gc'].T*pkl['gw'], axis=1))#dipole
                            loss_E += torch.abs(E_new-pkl['E_target'])
                    valid_loss_E += loss_E.item()
                    valid_loss_rho += loss_rho.item()
                    
                valid_loss_E /= len(valid_pkls)
                valid_loss_rho /= len(valid_pkls)
                valid_loss_record.append({'E':valid_loss_E*1, 'rho':valid_loss_rho, 'tot':valid_loss_E+valid_loss_rho})
            if (epoch+1)%save_every==0 and (epoch_loss_E+valid_loss_E+epoch_loss_rho+valid_loss_rho)<last_loss:
                last_loss = epoch_loss_E+valid_loss_E+epoch_loss_rho+valid_loss_rho
                print(f'saving model_E at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
            if (epoch+1)%save_every==0:
                plt.plot(np.arange(1, len(loss_record)+1)[:], [itm['E'] for itm in loss_record[:]], label='train E', alpha=0.1)
                plt.plot(np.arange(1, len(loss_record)+1)[:], [itm['rho'] for itm in loss_record[:]], label='train rho', alpha=0.1)
                plt.plot(np.arange(1, len(loss_record)+1)[:], [itm['tot'] for itm in loss_record[:]], label='train tot', alpha=0.1)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[:], [itm['E'] for itm in valid_loss_record][:], label='valid E', c='r', s=5)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[:], [itm['rho'] for itm in valid_loss_record][:], label='valid rho', c='green', s=5)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[:], [itm['tot'] for itm in valid_loss_record][:], label='valid tot', c='k', s=5)
                plt.plot([chk["epoch"], chk["epoch"]], plt.ylim(), '-.', label='chosen epoch')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Mean absolute error of E_tot and rho in a. u.')
                plt.savefig(f'{model_path[:-3]}_losses.png')
                plt.close()


    def eval_xc(self, xc_code='b3lypg', rho=None, spin=1, relativity=0, deriv=2, omega=None, verbose=None):
        e_b3, v_b3 = self.ni.eval_xc(xc_code, rho, spin, relativity=relativity, deriv=deriv, omega=omega, verbose=verbose)[:2]
        v_b3_rho, v_gamma_b3 = v_b3[0], v_b3[1]
        e_model, (v_model_rho, v_model_gamma) = self.forward_xc(rho=rho, spin=spin, off=self.off)
        return e_b3+e_model, (v_b3_rho+v_model_rho, v_gamma_b3+v_model_gamma, None, None), None, None
    
    def scf(self, data, basis, grid_level=3, hyb=0.2, off=False, dm_target=None, xc='b3lypg', xctype='GGA', init=None):
        self.off = off
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in 
            zip(data['atoms_charges'], data['atoms_coords'])], basis=basis, 
                    spin=data['spin'], charge=data['charge'], unit='bohr')
        # self.nu = mol.intor('int1e_grids_sph', grids=data['gc'])
        if dm_target is not None:
            df = myKS(mol, dm_target=dm_target)
        else:
            df = dft.KS(mol)
        df = myKS(mol, dm_target=dm_target)
        df.xc = xc
        df.grids.level = grid_level
        df.define_xc_(self.eval_xc, xctype=xctype, hyb=hyb)
        try:
            E = df.kernel(dm_init=init)
        except:
            print('===scf Internal Error, redo it===')
            E = df.kernel()
        dm = df.make_rdm1()
        if ('phi01' in data) and (type(data['phi01'])!=torch.Tensor):
            rho = dm2rho(dm, data['phi01'][0])
            rho01 = dm2rho01_sep(dm, data['phi01'])
            dipole = cal_dipole(rho, data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
            F = -guks(df).kernel() if data['spin'] else -gks(df).kernel()
            return {'dm_scf': dm, 'E_scf': E, 'rho_scf':rho, 'rho01_scf':rho01, 'dipole_scf':dipole, 'F_scf':F, 'converged': df.converged}
        return {'dm_scf': dm, 'E_scf': E, 'converged': df.converged}

class ModelE_MGGA(ModelE_pbe):
    def forward_xc(self, rho, spin, off=False, keep_graph=False):
        if len(rho) != 2:
            rho = 0.5*rho, 0.5*rho
        rho_up, rhox_up, rhoy_up, rhoz_up, rho_down, rhox_down, rhoy_down, rhoz_down = (*rho[0][:4], *rho[1][:4])
        tau_up, tau_down = rho[0][5], rho[1][5]
        guu = rhox_up**2+rhoy_up**2+rhoz_up**2
        gdd = rhox_down**2+rhoy_down**2+rhoz_down**2
        gud = rhox_up*rhox_down+rhoy_up*rhoy_down+rhoz_up*rhoz_down
        
        inp = np.vstack((rho_up, rho_down, guu, gud, gdd, tau_up, tau_down))
        inp = torch.tensor(inp, requires_grad=True).to(self.device)
        
        # t=torch.empty((inp.shape[1],4), device=self.device)
        # unif=(inp[0]+inp[1]+1e-7)**(1.0/3)
        # t[:,0]=unif
        # t[:,1]=((1+torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(4.0/3)+(1-torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(4.0/3))*0.5
        # t[:,2]=((inp[2]+inp[4]+2*inp[3])**0.5+1e-7)/unif**4
        # ds=(1+torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(5.0/3)+(1-torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(5.0/3)
        # t[:,3]=(inp[5]+inp[6]+1e-7)/(unif**5*ds)
        # e = self.forward(torch.log(t), off=off)

        t=torch.empty((inp.shape[1],7), device=self.device)
        unif=(inp[0]+inp[1]+1e-7)**(1.0/3)
        t[:,0]=inp[0]+1e-7
        t[:,1]=inp[1]+1e-7
        t[:,2]=((inp[2]+1e-7)**0.5)/unif**4
        t[:,3]=((inp[2]+inp[4]+2*inp[3])**0.5+1e-7)/unif**4
        t[:,4]=((inp[4]+1e-7)**0.5)/unif**4
        

        ds=(1+torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(5.0/3)+(1-torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(5.0/3)
        t[:,5]=(inp[5]+1e-7)/(unif**5*ds)
        t[:,6]=(inp[6]+1e-7)/(unif**5*ds)
        
        e = self.forward(torch.log(t), off=off)
        grad, = torch.autograd.grad((e*(inp[0]+inp[1])).sum(), inp, create_graph=True)
        if spin:
            vrho = grad[0:2].T
            vgamma = grad[2:5].T
            vtau = grad[5:7].T
        else:
            vrho = (grad[0] + grad[1]) / 2
            vgamma = (grad[2] + grad[3] + grad[4]) / 4
            vtau = (grad[5] + grad[6]) / 2
        if not keep_graph:
            v = (vrho.detach().cpu().numpy(), vgamma.detach().cpu().numpy(), vtau.detach().cpu().numpy())
            e = e.detach().cpu().numpy()
        else:
            v = (vrho, vgamma, vtau)
        return e, v

    def fit(self, train_pkls, valid_pkls, model_path, lr=0.0001, batch_size=3, max_iter=3000, save_every=200, valid_every=100, off=False):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False)
        for pkl in train_pkls+valid_pkls:
            for k in ['gc', 'gw', 'rho_base', 'rho_target', 'phi01', 'rho01', 'E_base', 'E_target', 'F0', 'S', 'J', 'K']:
                if type(pkl[k]) != torch.Tensor :
                    pkl[k] = torch.tensor(np.array(pkl[k])).to(self.device)
        loss_record = []
        valid_loss_record = []
        last_loss = np.inf
        for epoch in range(max_iter):
            self.optimizer.zero_grad()
            print(epoch)
            epoch_loss_E = 0.
            epoch_loss_rho = 0.
            for pkls in self.get_batches(train_pkls, batch_size):
                # self.optimizer.zero_grad()
                loss_E = 0.
                loss_rho = 0.
                rec = []
                for pkl in pkls:
                    rec.append(pkl)
                    if 'e_b3' not in pkl:
                        e_b3, (v_b3_rho, v_b3_gamma) = self.ni.eval_xc('b3lypg', rho=pkl['x'] if pkl['spin'] else (pkl['x'][0]+pkl['x'][1]), spin=pkl['spin'], deriv=1)[:2]
                        pkl['e_b3'] = torch.tensor(e_b3).to(self.device)
                        pkl['v_b3_rho'] = torch.tensor(v_b3_rho).to(self.device)
                        pkl['v_b3_gamma'] = torch.tensor(v_b3_gamma).to(self.device)
                    batch_e_p, (v_model_rho, v_model_gamma, v_model_tau) = self.forward_xc(rho=pkl['x'], spin=pkl['spin'], off=off, keep_graph=True)
                    batch_e_p = batch_e_p + pkl['e_b3']
                    v_model_rho = v_model_rho + pkl['v_b3_rho']
                    v_model_gamma = v_model_gamma + pkl['v_b3_gamma']
                    # loss_E += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                    
                    # print(len(pkl['gw']))
                    if len(pkl['gw'])<100000:
                        Vxc = v2V2(v_model_rho, v_model_gamma, v_model_tau, pkl['phi01'], pkl['gw'], pkl['rho01'])
                    else:
                        Vxc = v2V(v_model_rho, v_model_gamma, pkl['phi01'], pkl['gw'], pkl['rho01'])
                    if len(Vxc) == 2:
                        F = pkl['F0'] + Vxc[0] - 0.2*pkl['K'][0], pkl['F0'] + Vxc[1] - 0.2*pkl['K'][1]
                    else:
                        F = pkl['F0'] + Vxc - 0.5*0.2*pkl['K']
                    n_down = (pkl['n_elec'] - pkl['spin'])//2
                    n_up = pkl['n_elec'] - n_down
                    dm_new = F2dm(F, pkl['S'], n_up, n_down, self.device)
                    rho_new = dm2rho(dm_new, pkl['phi01'][0])
                    if len(Vxc) == 2:
                        E_K = 0.5 * 0.2 * (dm2E1e(dm_new[0], pkl['K'][0]) + dm2E1e(dm_new[1], pkl['K'][1]))
                    else:
                        E_K = 0.25 * 0.2 * dm2E1e(dm_new, pkl['K'])
                    # if np.random.random()>.5:
                    #     E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*pkl['rho_base']*pkl['gw']) + pkl['E_N'] -E_K
                    # else:
                    E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*rho_new*pkl['gw']) + pkl['E_N'] - E_K
                    # loss_rho += torch.abs((rho_new-pkl['rho_target'])*pkl['gw']).mean()
                    loss_rho += (torch.abs((rho_new-pkl['rho_target']))**(1/3)).sum()
                    # loss_rho += torch.abs(rho_new-pkl['rho_target']).mean()
                    # loss_rho += torch.linalg.norm(torch.sum((rho_new-pkl['rho_target'])*pkl['gc'].T*pkl['gw'], axis=1))#dipole
                    loss_E += (E_new-pkl['E_target'])**2
                    # loss_E += (E_new-pkl['E_target'])**2
                    pkl['dm_new'] = (dm_new[0].detach().cpu().numpy(), dm_new[1].detach().cpu().numpy()) if pkl['spin'] else dm_new.detach().cpu().numpy()
                
                epoch_loss_E += torch.abs(E_new.detach()-pkl['E_target']).item()
                epoch_loss_rho += (((rho_new.detach()-pkl['rho_target'])**2)*pkl['gw']).sum().item()
                (loss_E+loss_rho*0.).backward()# pre2 1.0, pre3 0.1,#pre4 0.1 no clip, pre5 5. no clip, pre6.no clip rho*gw*1e5 inp rhocc, pre8 update K using dm_new
                # pre9 +loss dipole, 
                # pre10 +loss dipole 4000 epoches clip no K update, 
                # pre11 +loss dipole 4000 epoches clip K update,
                # ->pre12 +loss dipole 4000 epoches clip K rho01 and Vxc update, 
                # pre13 loss rho no gw +loss dipole 4000 epoches clip K rho01 and Vxc update
                # pre14 to pre9, follow nagai's 3 features
                # !!!! never update before  pre15 to pre9, follow nagai's 4 features, update K, tau only E use target
                # pre 15rep follow nagai's 4 features, update, tau, all use target
                num_nan = 0
                for p in self.parameters():
                    num_nan += torch.sum(torch.isnan(p.grad.data))
                    # num_inf += torch.sum(torch.isinf(p.grad.data))
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=np.random.randn()*0., posinf=np.random.randn()*0., neginf=np.random.randn()*0.)
                    p.grad.data = torch.clip(p.grad.data, min=-10., max=10.)
                if num_nan:
                    print('epoch: %d num_nan'%epoch, num_nan, pkl['fn'])
                # self.optimizer.step()
                
                # for pkl in rec:
                #     if (epoch>400) and (np.random.random()>0.5):
                #         #TODO update rho01, F0, J, K, x using dm_new
                #         dm = pkl['dm_new']
                #         # pkl['rho01'] = torch.tensor(dm2rho01_sep(dm, pkl['phi01'])).to(self.device)
                # # #         pkl['rho_base'] = torch.tensor(dm2rho(dm, pkl['phi01'][0])).to(self.device)
                #         # F0 = pkl['T'] + pkl['V_ext'] + dm2J(dm, pkl['itg_2e'])
                #         # pkl['F0'] = torch.tensor(F0).to(self.device)
                #         # pkl['J'] = torch.tensor(F0 - pkl['T'] - pkl['V_ext']).to(self.device)
                #         # K = np.array([dm2K(dm[0], pkl['itg_2e']), dm2K(dm[1], pkl['itg_2e'])]) if pkl['spin'] else dm2K(dm, pkl['itg_2e'])
                #         # pkl['K'] = torch.tensor(K).to(self.device)
                #         if len(dm) != 2:
                #             dm = 0.5*dm, 0.5*dm
                #         # assert len(dm) == 2
                #         # pkl['x'] = dm2rho01(dm[0], pkl['phi01']), dm2rho01(dm[1], pkl['phi01'])
                #         phi01 = (pkl['phi01']).detach().cpu().numpy()
                #         # rho01 = dm2rho01(dm[0], phi01), dm2rho01(dm[1], phi01)
                        
                #         tau = 0.5*contract('ij, dri, drj->r', dm[0], phi01[1:], phi01[1:]), 0.5*contract('ij, dri, drj->r', dm[1], phi01[1:], phi01[1:])
                #         # pkl['x'] = np.array([*(rho01[0]), np.zeros_like(tau[0]), tau[0]]), np.array([*(rho01[1]), np.zeros_like(tau[1]), tau[1]])
                #         pkl['x'][0][-1] = tau[0]
                #         pkl['x'][1][-1] = tau[1]
            self.optimizer.step()
            epoch_loss_E /= len(train_pkls)
            epoch_loss_rho /= len(train_pkls)
            loss_record.append({'E':epoch_loss_E*1, 'rho':epoch_loss_rho, 'tot': epoch_loss_E+epoch_loss_rho})
            print('loss E', epoch_loss_E, 'loss rho', epoch_loss_rho)
            if (epoch+1)%valid_every==0:
                valid_loss_E = 0.
                valid_loss_rho = 0.
                for pkls in self.get_batches(valid_pkls, batch_size, shuffle=False):
                    loss_E = 0.
                    loss_rho = 0.
                    for pkl in pkls:
                        if 'e_b3' not in pkl:
                            e_b3, (v_b3_rho, v_b3_gamma) = self.ni.eval_xc('b3lypg', rho=pkl['x'] if pkl['spin'] else (pkl['x'][0]+pkl['x'][1]), spin=pkl['spin'], deriv=1)[:2]
                            pkl['e_b3'] = torch.tensor(e_b3).to(self.device)
                            pkl['v_b3_rho'] = torch.tensor(v_b3_rho).to(self.device)
                            pkl['v_b3_gamma'] = torch.tensor(v_b3_gamma).to(self.device)
                        batch_e_p, (v_model_rho, v_model_gamma, v_model_tau) = self.forward_xc(rho=pkl['x'], spin=pkl['spin'], off=off, keep_graph=True)
                        batch_e_p = batch_e_p.detach()
                        v_model_rho = v_model_rho.detach()
                        v_model_gamma = v_model_gamma.detach()
                        batch_e_p = batch_e_p + pkl['e_b3']
                        v_model_rho = v_model_rho + pkl['v_b3_rho']
                        v_model_gamma = v_model_gamma + pkl['v_b3_gamma']
                        with torch.no_grad():
                            # loss_E += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                            if len(pkl['gw'])<100000:
                                Vxc = v2V2(v_model_rho, v_model_gamma, v_model_tau, pkl['phi01'], pkl['gw'], pkl['rho01'])
                            else:
                                Vxc = v2V(v_model_rho, v_model_gamma, pkl['phi01'], pkl['gw'], pkl['rho01'])
                            if len(Vxc) == 2:
                                F = pkl['F0'] + Vxc[0] - 0.2*pkl['K'][0], pkl['F0'] + Vxc[1] - 0.2*pkl['K'][1]
                            else:
                                F = pkl['F0'] + Vxc - 0.5*0.2*pkl['K']
                            n_down = (pkl['n_elec'] - pkl['spin'])//2
                            n_up = pkl['n_elec'] - n_down
                            dm_new = F2dm(F, pkl['S'], n_up, n_down, self.device)
                            rho_new = dm2rho(dm_new, pkl['phi01'][0])
                            if len(Vxc) == 2:
                                E_K = 0.5 * 0.2 * (dm2E1e(dm_new[0], pkl['K'][0]) + dm2E1e(dm_new[1], pkl['K'][1]))
                            else:
                                E_K = 0.25 * 0.2 * dm2E1e(dm_new, pkl['K'])
                            # if np.random.random()>.5:
                            #     E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*pkl['rho_base']*pkl['gw']) + pkl['E_N'] -E_K
                            # else:
                            E_new = dm2E1e(dm_new, pkl['F0']-pkl['J']) + 0.5*dm2E1e(dm_new, pkl['J']) + torch.sum(batch_e_p*rho_new*pkl['gw']) + pkl['E_N'] - E_K
                            # loss_rho += torch.abs((rho_new-pkl['rho_target'])*pkl['gw']).mean()
                            loss_rho += (((rho_new-pkl['rho_target'])**2)*pkl['gw']).sum()
                            # loss_rho += torch.abs(rho_new-pkl['rho_target']).mean()
                            # loss_rho += torch.linalg.norm(torch.sum((rho_new-pkl['rho_target'])*pkl['gc'].T*pkl['gw'], axis=1))#dipole
                            loss_E += torch.abs(E_new-pkl['E_target'])
                            # loss_E += (E_new-pkl['E_target'])**2
                    valid_loss_E += loss_E.item()
                    valid_loss_rho += loss_rho.item()
                    
                valid_loss_E /= len(valid_pkls)
                valid_loss_rho /= len(valid_pkls)
                valid_loss_record.append({'E':valid_loss_E*1, 'rho':valid_loss_rho, 'tot':valid_loss_E+valid_loss_rho})
            if (epoch+1)%save_every==0 and (epoch_loss_E+valid_loss_E+epoch_loss_rho+valid_loss_rho)<last_loss:
                last_loss = epoch_loss_E+valid_loss_E+epoch_loss_rho+valid_loss_rho
                print(f'saving model_E at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
            if (epoch+1)%save_every==0:
                plt.plot(np.arange(1, len(loss_record)+1)[:], [itm['E'] for itm in loss_record[:]], label='train E', alpha=0.1)
                plt.plot(np.arange(1, len(loss_record)+1)[:], [itm['rho'] for itm in loss_record[:]], label='train rho', alpha=0.1)
                plt.plot(np.arange(1, len(loss_record)+1)[:], [itm['tot'] for itm in loss_record[:]], label='train tot', alpha=0.1)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[:], [itm['E'] for itm in valid_loss_record][:], label='valid E', c='r', s=5)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[:], [itm['rho'] for itm in valid_loss_record][:], label='valid rho', c='green', s=5)
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[:], [itm['tot'] for itm in valid_loss_record][:], label='valid tot', c='k', s=5)
                plt.plot([chk["epoch"], chk["epoch"]], plt.ylim(), '-.', label='chosen epoch')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Mean absolute error of E_tot and rho in a. u.')
                plt.savefig(f'{model_path[:-3]}_losses.png')
                plt.close()


    def eval_xc(self, xc_code='b3lypg', rho=None, spin=1, relativity=0, deriv=2, omega=None, verbose=None):
        e_b3, v_b3 = self.ni.eval_xc(xc_code, rho, spin, relativity=relativity, deriv=deriv, omega=omega, verbose=verbose)[:2]
        v_b3_rho, v_gamma_b3 = v_b3[0], v_b3[1]
        e_model, (v_model_rho, v_model_gamma, v_model_tau) = self.forward_xc(rho=rho, spin=spin, off=self.off)
        return e_b3+e_model, (v_b3_rho+v_model_rho, v_gamma_b3+v_model_gamma, np.zeros_like(v_b3_rho), v_model_tau), None, None
    
    def scf(self, data, basis, grid_level=3, hyb=0.2, off=False, dm_target=None, xc='b3lypg', xctype='MGGA'):
        self.off = off
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in 
            zip(data['atoms_charges'], data['atoms_coords'])], basis=basis, 
                    spin=data['spin'], charge=data['charge'], unit='bohr')
        # self.nu = mol.intor('int1e_grids_sph', grids=data['gc'])
        if dm_target is not None:
            df = myKS(mol, dm_target=dm_target)
        else:
            df = dft.KS(mol)
        df = myKS(mol, dm_target=dm_target)
        df.xc = xc
        df.grids.level = grid_level
        df.define_xc_(self.eval_xc, xctype=xctype, hyb=hyb)
        E = df.kernel()
        dm = df.make_rdm1()
        if 'phi01' in data:
            rho = dm2rho(dm, data['phi01'][0])
            rho01 = dm2rho01_sep(dm, data['phi01'])
            dipole = cal_dipole(rho, data['gc'], data['gw'], data['atoms_charges'], data['atoms_coords'])
            F = -guks(df).kernel() if data['spin'] else -gks(df).kernel()
            return {'dm_scf': dm, 'E_scf': E, 'rho_scf':rho, 'rho01_scf':rho01, 'dipole_scf':dipole, 'F_scf':F, 'converged': df.converged}
        return {'dm_scf': dm, 'E_scf': E, 'converged': df.converged}

class ModelE_post(ModelE):
    def forward(self, rho, off=False):
        if len(rho) != 2:
            rho = 0.5*rho, 0.5*rho
        rho_up, rhox_up, rhoy_up, rhoz_up, rho_down, rhox_down, rhoy_down, rhoz_down = (*rho[0][:4], *rho[1][:4])
        
        guu = rhox_up**2+rhoy_up**2+rhoz_up**2
        gdd = rhox_down**2+rhoy_down**2+rhoz_down**2
        gud = rhox_up*rhox_down+rhoy_up*rhoy_down+rhoz_up*rhoz_down
        
        inp = np.vstack((rho_up, rho_down, guu, gud, gdd))
        inp = torch.tensor(inp).to(self.device)
        
        g = inp[2:5]/((inp[0]+inp[1]+1e-7)**(4/3))/(2*(3*torch.pi)**(1/3))
        e = self.lin(torch.vstack((inp[0], inp[1], *g,)).T)
        if self.n_outs_node>1:
            return e*0 if off else e
        else:
            return e.reshape(-1)*0 if off else e.reshape(-1)
    def fit(self, train_pkls, valid_pkls, model_path, lr=0.0001, batch_size=3, max_iter=3000, save_every=100, valid_every=50, off=False):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False)
        for pkl in train_pkls+valid_pkls:
            for k in ['gw', 'rho_base']:
                if (type(pkl[k]) == np.ndarray) or (type(pkl[k][0]) == np.ndarray):
                    pkl[k] = torch.tensor(pkl[k]).to(self.device)
        # dataloader = torch.utils.data.DataLoader(Dataset(train_pkls), batch_size=batch_size, shuffle=True)
        # validloader = torch.utils.data.DataLoader(Dataset(valid_pkls), batch_size=batch_size, shuffle=False)
        loss_record = []
        valid_loss_record = []
        last_loss = np.inf
        for epoch in range(max_iter):
            epoch_loss = 0.
            for pkls in self.get_batches(train_pkls, batch_size):
                self.optimizer.zero_grad()
                loss_e = 0.
                for pkl in pkls:
                    batch_e_p = self.forward(pkl['x'], off=off)
                    loss_e += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                epoch_loss += loss_e.item()
                loss_e.backward()
                self.optimizer.step()
            epoch_loss /= len(train_pkls)
            loss_record.append(epoch_loss*627)
            if (epoch+1)%valid_every==0:
                with torch.no_grad():
                    valid_loss = 0.
                    for pkls in self.get_batches(valid_pkls, batch_size, shuffle=False):
                        loss_e = 0.
                        for pkl in pkls:
                            batch_e_p = self.forward(pkl['x'], off=off)
                            loss_e += torch.abs(torch.sum(batch_e_p*pkl['rho_base']*pkl['gw'])-(pkl['E_target']-pkl['E_base']))
                        valid_loss += loss_e.item()
                    valid_loss /= len(valid_pkls)
                    valid_loss_record.append(valid_loss*627)
            if (epoch+1)%save_every==0 and (epoch_loss+valid_loss)*627<last_loss:
                last_loss = (epoch_loss+valid_loss)*627
                print(f'saving model_E at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
            if (epoch+1)%save_every==0:
                plt.plot(np.arange(1, len(loss_record)+1)[:], loss_record[:], label='train')
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[:], valid_loss_record[:], label='valid', c='r', s=5)
                plt.plot([chk["epoch"], chk["epoch"]], plt.ylim(), '-.', label='chosen epoch')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Mean absolute error of E_tot in kcal/mol')
                plt.savefig(f'{model_path[:-3]}_losses.png')
                plt.close()
                
class ModelE_post_MGGA(ModelE_post):
    def forward(self, rho, off=False):
        if len(rho) != 2:
            rho = 0.5*rho, 0.5*rho
        rho_up, rhox_up, rhoy_up, rhoz_up, rho_down, rhox_down, rhoy_down, rhoz_down = (*rho[0][:4], *rho[1][:4])
        tau_up, tau_down = rho[0][5], rho[1][5]
        guu = rhox_up**2+rhoy_up**2+rhoz_up**2
        gdd = rhox_down**2+rhoy_down**2+rhoz_down**2
        gud = rhox_up*rhox_down+rhoy_up*rhoy_down+rhoz_up*rhoz_down
        
        inp = np.vstack((rho_up, rho_down, guu, gud, gdd, tau_up, tau_down))
        inp = torch.tensor(inp).to(self.device)
        
        t=torch.empty((inp.shape[1],4), device=self.device)
        unif=(inp[0]+inp[1]+1e-7)**(1.0/3)
        t[:,0]=unif
        t[:,1]=((1+torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(4.0/3)+(1-torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(4.0/3))*0.5
        t[:,2]=((inp[2]+inp[4]+2*inp[3])**0.5+1e-7)/unif**4
        ds=(1+torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(5.0/3)+(1-torch.div((inp[0]-inp[1]),(inp[0]+inp[1]+1e-7)))**(5.0/3)
        t[:,3]=(inp[5]+inp[6]+1e-7)/(unif**5*ds)
        
        e = self.lin(torch.log(t))
        if self.n_outs_node>1:
            return e*0 if off else e
        else:
            return e.reshape(-1)*0 if off else e.reshape(-1)

class ModelD(ModelE):
    def __init__(self, feature_num=1331, in_channels=1, n_outs_node=1, device='cuda:7', **kwargs):
        super().__init__(feature_num, in_channels, n_outs_node, device)
        #overwrite the model architecture
        self.lin = Sequential(*([
        # Linear(feature_num, 256), ReLU(),
        # Linear(256, 256), ReLU(),
        # Linear(256, 128), ReLU(),
        # Linear(128, 128), ReLU(),
        # Linear(128, 64), ReLU(),
        # Linear(64, n_outs_node),
        Linear(feature_num, 40), ReLU(),
        Linear(40, 40), ReLU(),
        
        Linear(40, 256), ReLU(),
        Linear(256, 256), ReLU(),
        Linear(256, 40), ReLU(),
        
        Linear(40, n_outs_node),
        ]))
    

        
    def forward(self, x, off=False):
        x = x.reshape(len(x), -1)
        d = self.lin(torch.log(torch.abs(x)+1e-3))**3
        if self.n_outs_node>1:
            return d*0+1 if off else d+1
        else:
            return d.reshape(-1)*0+1 if off else d.reshape(-1)+1

    def fit(self, train_pkls, valid_pkls, model_path, lr=0.0001, batch_size=3, max_iter=3000, save_every=1000, valid_every=100, off=False):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False, weight_decay=0.)
        for pkl in train_pkls+valid_pkls:
            for k in ['x', 'rho_base', 'rho_target', 'gc', 'gw', 'dipole_target', 'dipole_base']:
                if type(pkl[k]) == np.ndarray:
                    pkl[k] = torch.tensor(pkl[k]).to(self.device)
        loss_record = []
        valid_loss_record = []
        last_loss = np.inf
        for epoch in range(max_iter):
            epoch_loss = 0.
            for pkls in self.get_batches(train_pkls, batch_size):
                self.optimizer.zero_grad()
                loss_rho = 0.
                for pkl in pkls:
                    batch_rho_p = self.forward(pkl['x'], off=off)

                    # rho = torch.relu(batch_rho_p*pkl['rho_base'])
                    # diff_rho = rho - pkl['rho_base']
                    # pos_idx = diff_rho>0
                    # ratio = -torch.sum((diff_rho*pkl['gw'])[pos_idx])/(torch.sum((diff_rho*pkl['gw'])[~pos_idx]))
                    # diff_rho[~pos_idx] *= ratio
                    # rho = pkl['rho_base'] + diff_rho
                    # loss_rho += torch.sum((torch.sum((rho-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2)**.5

                    loss_rho += (torch.sum(batch_rho_p*pkl['rho_base']*pkl['gw'])-torch.sum(pkl['rho_target']*pkl['gw']))**2*100#TODO maybe I value loss more fair
                    loss_rho += torch.sum((torch.sum((batch_rho_p*pkl['rho_base']-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2)**.5
                epoch_loss += loss_rho.item()
                loss_rho.backward()
                self.optimizer.step()
            epoch_loss /= len(train_pkls)
            loss_record.append(epoch_loss)
            if (epoch+1)%valid_every==0:
                with torch.no_grad():
                    valid_loss = 0.
                    for pkls in self.get_batches(valid_pkls, batch_size, shuffle=False):
                        loss_rho = 0.
                        for pkl in pkls:
                            batch_rho_p = self.forward(pkl['x'], off=off)
                            # rho = torch.relu(batch_rho_p*pkl['rho_base'])
                            # diff_rho = rho - pkl['rho_base']
                            # pos_idx = diff_rho>0
                            # ratio = -torch.sum((diff_rho*pkl['gw'])[pos_idx])/torch.sum((diff_rho*pkl['gw'])[~pos_idx])
                            # diff_rho[~pos_idx] *= ratio
                            # rho = pkl['rho_base'] + diff_rho
                            # loss_rho += torch.sum((torch.sum((rho-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2)**.5

                            loss_rho += (torch.sum(batch_rho_p*pkl['rho_base']*pkl['gw'])-torch.sum(pkl['rho_target']*pkl['gw']))**2*100
                            loss_rho += torch.sum((torch.sum((batch_rho_p*pkl['rho_base']-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2)**.5
                            
                            # print('loss dip', torch.sum((torch.sum((batch_rho_p*pkl['rho_base']-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2))
                            # loss_rho += torch.sum(batch_rho_p*pkl['gw'])**2
                            # print('rho', torch.sum(batch_rho_p*pkl['gw'])**2)
                            # loss_rho += torch.sum((torch.sum(batch_rho_p*pkl['gw']*(pkl['gc'].T), axis=1)-(pkl['dipole_target']-pkl['dipole_base']))**2)*0.1
                            # print('dipole', torch.sum((torch.sum(batch_rho_p*pkl['gw']*(pkl['gc'].T), axis=1)-(pkl['dipole_target']-pkl['dipole_base']))**2)*0.1)
                            # loss_rho += torch.abs(torch.linalg.norm(torch.sum(batch_rho_p*pkl['gw']*(pkl['gc'].T), axis=1)+pkl['dipole_base'])-torch.linalg.norm(pkl['dipole_target']))
                        valid_loss += loss_rho.item()
                    valid_loss /= len(valid_pkls)
                    valid_loss_record.append(valid_loss)
            if (epoch+1)%save_every==0 and (epoch_loss+valid_loss)<last_loss:
                last_loss = (epoch_loss+valid_loss)
                print(f'saving model_D at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
            if (epoch+1)%save_every==0:
                print('loss', (epoch_loss+valid_loss), last_loss)
                plt.plot(np.arange(1, len(loss_record)+1)[500:], loss_record[500:], label='train')
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[4:], valid_loss_record[4:], label='valid', c='r', s=5)
                plt.plot([chk["epoch"], chk["epoch"]], plt.ylim(), '-.', label='chosen epoch')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Mean of loss / a. u.')
                plt.savefig(f'{model_path[:-3]}_losses.png')
                plt.close()

class ModelF(ModelE):
    def __init__(self, feature_num=1331, in_channels=1, n_outs_node=1, device='cuda:7', **kwargs):
        super().__init__(feature_num, in_channels, n_outs_node, device)
        #overwrite the model architecture
        self.lin = Sequential(*([
        # Linear(feature_num, 256), ReLU(),
        # Linear(256, 256), ReLU(),
        # Linear(256, 128), ReLU(),
        # Linear(128, 128), ReLU(),
        # Linear(128, 64), ReLU(),
        # Linear(64, n_outs_node),
        Linear(feature_num, 40), ReLU(),
        Linear(40, 40), ReLU(),
        Linear(40, 40), ReLU(),
        Linear(40, n_outs_node),
        ]))
    

        
    def forward(self, x, off=False):
        x = x.reshape(len(x), -1)
        d = self.lin(torch.log(torch.abs(x)+1e-3))**3
        if self.n_outs_node>1:
            return d*0 if off else d
        else:
            return d.reshape(-1)*0 if off else d.reshape(-1)

    def fit(self, train_pkls, valid_pkls, model_path, lr=0.0001, batch_size=3, max_iter=3000, save_every=1000, valid_every=100, off=False):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False, weight_decay=0.)
        for pkl in train_pkls+valid_pkls:
            for k in ['x', 'rho_base', 'F_base', 'F_target', 'gc', 'gw', 'atoms_charges', 'atoms_coords']:
                if type(pkl[k]) == np.ndarray:
                    pkl[k] = torch.tensor(pkl[k]).to(self.device)
        loss_record = []
        valid_loss_record = []
        last_loss = np.inf
        for epoch in range(max_iter):
            epoch_loss = 0.
            for pkls in self.get_batches(train_pkls, batch_size):
                self.optimizer.zero_grad()
                loss_F = 0.
                for pkl in pkls:
                    d3 = torch.sum((pkl['atoms_coords'][:,None] - pkl['gc'])**2, axis=-1)**1.5
                    # r = (pkl['atoms_coords'][:,None] - pkl['gc']).transpose(2,0,1)
                    r = (pkl['atoms_coords'][:,None] - pkl['gc']).transpose(1,2).transpose(0,1)
                    batch_rho_p = self.forward(pkl['x'], off=off)
                    # def get_force_ele(atoms_charges, atoms_coords, rho, gc, gw):
                    #     d3 = np.sum((atoms_coords[:,None] - gc)**2, axis=-1)**1.5
                    #     r = (atoms_coords[:,None] - gc).transpose(2,0,1)
                    #     force = -r/d3*np.einsum('i,j,j->ij', atoms_charges, rho, gw)
                    #     return force.sum(axis=2).T
                    force = (-r/d3*contract('i,j,j->ij', pkl['atoms_charges'], batch_rho_p*pkl['rho_base'], pkl['gw'])).sum(axis=2).T
                    # force = force - force.mean(axis=0)
                    # torque = torch.cross(pkl['atoms_coords'], force).sum(axis=0)
                    
                    loss_F += torch.sum((pkl['F_base'] + force - pkl['F_target'])**2)**.5

                epoch_loss += loss_F.item()
                loss_F.backward()
                self.optimizer.step()
            epoch_loss /= len(train_pkls)
            loss_record.append(epoch_loss)
            if (epoch+1)%valid_every==0:
                with torch.no_grad():
                    valid_loss = 0.
                    for pkls in self.get_batches(valid_pkls, batch_size, shuffle=False):
                        loss_F = 0.
                        for pkl in pkls:
                            d3 = torch.sum((pkl['atoms_coords'][:,None] - pkl['gc'])**2, axis=-1)**1.5
                            # r = (pkl['atoms_coords'][:,None] - pkl['gc']).transpose(2,0,1)
                            r = (pkl['atoms_coords'][:,None] - pkl['gc']).transpose(1,2).transpose(0,1)
                            batch_rho_p = self.forward(pkl['x'], off=off)
                            force = (-r/d3*contract('i,j,j->ij', pkl['atoms_charges'], batch_rho_p*pkl['rho_base'], pkl['gw'])).sum(axis=2).T
                            # force = force - force.mean(axis=0)
                            loss_F += torch.sum((pkl['F_base'] + force - pkl['F_target'])**2)**.5
                        valid_loss += loss_F.item()
                    valid_loss /= len(valid_pkls)
                    valid_loss_record.append(valid_loss)
            if (epoch+1)%save_every==0 and (epoch_loss+valid_loss)<last_loss:
                last_loss = (epoch_loss+valid_loss)
                print(f'saving model_F at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
            if (epoch+1)%save_every==0:
                plt.plot(np.arange(1, len(loss_record)+1)[500:], loss_record[500:], label='train')
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[4:], valid_loss_record[4:], label='valid', c='r', s=5)
                plt.plot([chk["epoch"], chk["epoch"]], plt.ylim(), '-.', label='chosen epoch')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Mean of loss / a. u.')
                plt.savefig(f'{model_path[:-3]}_losses.png')
                plt.close()

class UV(ModelE):
    def __init__(self, feature_num=1331, in_channels=1, n_outs_node=1, device='cuda:7', **kwargs):
        super().__init__(feature_num, in_channels, n_outs_node, device)
        #overwrite the model architecture
        self.lin = Sequential(*([
        # Linear(feature_num, 256), ReLU(),
        # Linear(256, 256), ReLU(),
        # Linear(256, 128), ReLU(),
        # Linear(128, 128), ReLU(),
        # Linear(128, 64), ReLU(),
        # Linear(64, n_outs_node),
        Linear(feature_num, 40), ReLU(),
        Linear(40, 40), ReLU(),
        Linear(40, 40), ReLU(),
        Linear(40, n_outs_node),
        ]))
    
    def eval_uv(self, feature, keep_grad=True):
        with torch.enable_grad():
            u = self.model(feature)
            u = u if keep_grad else u.cpu().data.numpy()
            if feature.grad is not None:
                feature.grad.zero_()
            grad = torch.autograd.grad(torch.sum(u*(feature[:,0])), feature, create_graph=keep_grad)[0]
        v = grad if keep_grad else grad.cpu().data.numpy()
        return u, v
    
    def eval_UV(self, feature, phi01, gw, keep_grad=True):
        tensor = torch.tensor(feature, requires_grad=True, device=self.device)
        tensor.retain_grad()
        u, v = self.eval_uv(tensor, keep_grad)
        if keep_grad:
            phi, phi1 = torch.tensor(phi01[0]).to(self.device), torch.tensor(phi01[1:]).to(self.device)
            gw = torch.tensor(gw).to(self.device)
            # rho1 = torch.tensor(data['rho01_ccsd'][1:]).to(self.device)
        else:
            phi, phi1 = phi01[0], phi01[1:]
            # rho1 = data['rho01_ccsd'][1:]
        V = contract('ri,r,r,rj->ij', phi, v[:,0], gw, phi)
        gnorm_phi = contract('dri, dri->ri', phi1, phi1)**0.5
        V += 2. * contract('ri,r,r,rj->ij', gnorm_phi, v[:,1], gw, phi)
        V = (V + V.conj().T) / 2.0
        U = contract('r,r,r->', u, tensor[:,0], gw)
        return U, V
        
    def forward(self, x, off=False):
        x = x.reshape(len(x), -1)
        d = self.lin(torch.log(torch.abs(x)+1e-3))**3
        if self.n_outs_node>1:
            return d*0+1 if off else d+1
        else:
            return d.reshape(-1)*0+1 if off else d.reshape(-1)+1

    def fit(self, train_pkls, valid_pkls, model_path, lr=0.0001, batch_size=3, max_iter=3000, save_every=1000, valid_every=100, off=False):
        self.to(self.device)
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr, foreach=False, weight_decay=0.)
        for pkl in train_pkls+valid_pkls:
            for k in ['x', 'rho_base', 'rho_target', 'gc', 'gw', 'dipole_target', 'dipole_base']:
                if type(pkl[k]) == np.ndarray:
                    pkl[k] = torch.tensor(pkl[k]).to(self.device)
        loss_record = []
        valid_loss_record = []
        last_loss = np.inf
        for epoch in range(max_iter):
            epoch_loss = 0.
            for pkls in self.get_batches(train_pkls, batch_size):
                self.optimizer.zero_grad()
                loss_rho = 0.
                for pkl in pkls:
                    batch_rho_p = self.forward(pkl['x'], off=off)

                    # rho = torch.relu(batch_rho_p*pkl['rho_base'])
                    # diff_rho = rho - pkl['rho_base']
                    # pos_idx = diff_rho>0
                    # ratio = -torch.sum((diff_rho*pkl['gw'])[pos_idx])/torch.sum((diff_rho*pkl['gw'])[~pos_idx])
                    # diff_rho[~pos_idx] *= ratio
                    # rho = pkl['rho_base'] + diff_rho
                    # loss_rho += torch.sum((torch.sum((rho-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2)**.5

                    loss_rho += (torch.sum(batch_rho_p*pkl['rho_base']*pkl['gw'])-torch.sum(pkl['rho_target']*pkl['gw']))**2*100#TODO maybe I value loss more fair
                    loss_rho += torch.sum((torch.sum((batch_rho_p*pkl['rho_base']-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2)**.5
                    # loss_rho += torch.abs(torch.linalg.norm(torch.sum(batch_rho_p*pkl['gw']*(pkl['gc'].T), axis=1)+pkl['dipole_base'])-torch.linalg.norm(pkl['dipole_target']))
                epoch_loss += loss_rho.item()
                loss_rho.backward()
                self.optimizer.step()
            epoch_loss /= len(train_pkls)
            loss_record.append(epoch_loss)
            if (epoch+1)%valid_every==0:
                with torch.no_grad():
                    valid_loss = 0.
                    for pkls in self.get_batches(valid_pkls, batch_size, shuffle=False):
                        loss_rho = 0.
                        for pkl in pkls:
                            batch_rho_p = self.forward(pkl['x'], off=off)
                            # rho = torch.relu(batch_rho_p*pkl['rho_base'])
                            # diff_rho = rho - pkl['rho_base']
                            # pos_idx = diff_rho>0
                            # ratio = -torch.sum((diff_rho*pkl['gw'])[pos_idx])/torch.sum((diff_rho*pkl['gw'])[~pos_idx])
                            # diff_rho[~pos_idx] *= ratio
                            # rho = pkl['rho_base'] + diff_rho
                            # loss_rho += torch.sum((torch.sum((rho-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2)**.5

                            # print(batch_rho_p.mean())
                            loss_rho += (torch.sum(batch_rho_p*pkl['rho_base']*pkl['gw'])-torch.sum(pkl['rho_target']*pkl['gw']))**2*100
                            loss_rho += torch.sum((torch.sum((batch_rho_p*pkl['rho_base']-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2)**.5
                            # print('loss rho', torch.abs(torch.sum(batch_rho_p*pkl['rho_base']*pkl['gw'])-torch.sum(pkl['rho_target']*pkl['gw']))*100)
                            
                            # print('loss dip', torch.sum((torch.sum((batch_rho_p*pkl['rho_base']-pkl['rho_target'])*pkl['gw']*(pkl['gc'].T), axis=1))**2))
                            # loss_rho += torch.sum(batch_rho_p*pkl['gw'])**2
                            # print('rho', torch.sum(batch_rho_p*pkl['gw'])**2)
                            # loss_rho += torch.sum((torch.sum(batch_rho_p*pkl['gw']*(pkl['gc'].T), axis=1)-(pkl['dipole_target']-pkl['dipole_base']))**2)*0.1
                            # print('dipole', torch.sum((torch.sum(batch_rho_p*pkl['gw']*(pkl['gc'].T), axis=1)-(pkl['dipole_target']-pkl['dipole_base']))**2)*0.1)
                            # loss_rho += torch.abs(torch.linalg.norm(torch.sum(batch_rho_p*pkl['gw']*(pkl['gc'].T), axis=1)+pkl['dipole_base'])-torch.linalg.norm(pkl['dipole_target']))
                        valid_loss += loss_rho.item()
                    valid_loss /= len(valid_pkls)
                    valid_loss_record.append(valid_loss)
            if (epoch+1)%save_every==0 and (epoch_loss+valid_loss)<last_loss:
                last_loss = (epoch_loss+valid_loss)
                print(f'saving model_D at {epoch+1}th epoch.')
                chk = {'net': self.state_dict(),
                'opt': self.optimizer.state_dict(),
                'epoch': epoch+1,
                'losses':loss_record,
                'valid_losses':valid_loss_record}
                torch.save(chk, model_path)
            if (epoch+1)%save_every==0:
                plt.plot(np.arange(1, len(loss_record)+1)[500:], loss_record[500:], label='train')
                plt.scatter(np.arange(valid_every, (len(valid_loss_record)+1)*valid_every, valid_every)[4:], valid_loss_record[4:], label='valid', c='r', s=5)
                plt.plot([chk["epoch"], chk["epoch"]], plt.ylim(), '-.', label='chosen epoch')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Mean of loss / a. u.')
                plt.savefig(f'{model_path[:-3]}_losses.png')
                plt.close()
    


if __name__ == '__main__':
    train_list, valid_list = None, None
    model_e = ModelE(feature_num=len(train_list[0]['x'][0]))
    model_e.fit(train_list, valid_list, '/home/alfred/tree_regression/model/ef626c6c7e_b3lypg_to_ccsdt_1_eT2_False_e.pt')
    
    model_d = ModelD(feature_num=len(train_list[0]['x'][0]))
    model_d.fit(train_list, valid_list, '/home/alfred/tree_regression/model/ef626c6c7e_b3lypg_to_ccsdt_1_eT2_False_d.pt')

