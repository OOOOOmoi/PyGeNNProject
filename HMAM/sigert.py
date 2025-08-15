import numpy as np
import os
import sys
from scipy import optimize
import matplotlib.pyplot as plt

from theory_helpers import nu0_fb
from copy import copy
from copy import deepcopy

def convert_syn_weight(W, neuron_params):
    """
    Convert the amplitude of the PSC into mV.

    Parameters
    ----------
    W : float
        Synaptic weight defined as the amplitude of the post-synaptic current.
    neuron_params : dict
        Parameters of the neuron.
    """
    tau_syn_ex = neuron_params['tau_syn_ex']
    C_m = neuron_params['C_m']
    PSP_transform = tau_syn_ex / C_m

    return PSP_transform * W


def mu_sigma(rates, K_matrix, J_matrix, rate_ext, single_neuron_params, DC):
    rates = np.hstack((rates, rate_ext))
    C_m =  single_neuron_params['C_m']  # pF
    tau_m = single_neuron_params['tau_m']  # ms
    # 转换时间常数为秒
    tau_syn_ex = single_neuron_params['tau_syn_ex'] * 1e-3  # s

    # 计算平均电流 μ
    # mu = Σ_j (K_ij * J_ij * r_j) + 外部输入
    KJ = K_matrix * J_matrix
    J2 = J_matrix * J_matrix
    KJ2 = KJ * J_matrix
    mu = tau_m * 1e-3 * np.dot(KJ, rates) + tau_m / C_m * DC
    sigma2 = tau_m * 1e-3 * np.dot(KJ2, rates)
    sigma = np.sqrt(sigma2)
    return mu, sigma

class network1D:
    def __init__(self, params):
        self.label = '1D'
        self.params = {'input_params': params['input_params'],
                       'neuron_params': {'single_neuron_dict': copy(single_neuron_dict)},
                       'connection_params': {'replace_cc': None,
                                             'replace_cc_input_source': None}
                       }
        nested_update(self.params, params)
        self.add_DC_drive = np.zeros(1)
        self.structure = {'A': {'E'}}
        self.structure_vec = ['A-E']
        self.area_list = ['A']
        if 'K_stable' in params.keys():
            self.K_matrix = np.array([[params['K_stable'], params['K']]])
        else:
            self.K_matrix = np.array([[params['K'], params['K']]])

        self.W_matrix = np.array([[params['W'], params['W']]])
        self.J_matrix = convert_syn_weight(self.W_matrix,
                                           self.params['neuron_params']['single_neuron_dict'])

    def Phi(self, rate):
        mu, sigma = mu_sigma(rate, K_matrix=self.K_matrix,
                             J_matrix=self.J_matrix,
                             input_params=self.params['input_params'],
                             single_neuron_params=self.params['neuron_params']['single_neuron_dict'],
                             DC=self.add_DC_drive)
        NP = self.params['neuron_params']['single_neuron_dict']
        return list(map(lambda mu, sigma: nu0_fb(mu, sigma,
                                                    1.e-3 * NP['tau_m'],
                                                    1.e-3 * NP['tau_syn_ex'],
                                                    1.e-3 * NP['t_ref'],
                                                    NP['V_th'] - NP['E_L'],
                                                    NP['V_reset'] - NP['E_L']),
                        mu, sigma))

    def Phi_noisefree(self, rate):
        mu, sigma = mu_sigma(rate, K_matrix=self.K_matrix,
                             J_matrix=self.J_matrix,
                             input_params=self.params['input_params'],
                             single_neuron_params=self.params['neuron_params']['single_neuron_dict'],
                             DC=self.add_DC_drive)
        NP = self.params['neuron_params']['single_neuron_dict']
        th_shift = NP['V_th'] - NP['E_L']
        if mu > th_shift:
            T = 1e-3 * NP['tau_m'] * \
                np.log(mu[0] / (mu[0] - th_shift))
            return (1 / T)
        else:
            return 0.

    def fsolve(self, rates_init):
        def f(rate):
            return self.Phi(rate) - rate
        result = optimize.fsolve(f, rates_init, full_output=1)
        mu, sigma = mu_sigma(result[0], K_matrix=self.K_matrix,
                             J_matrix=self.J_matrix,
                             input_params=self.params['input_params'],
                             single_neuron_params=self.params['neuron_params']['single_neuron_dict'],
                             DC=self.add_DC_drive)
        result_dic = {'rates': np.array([result[0]]), 'mus': np.array(
            [mu]), 'sigmas': np.array([sigma]), 'eps': result[-1], 'time': np.array([0])}
        return result_dic


"""
Network class for the 2D case:
2 excitatory populations with recurrent connectivity and external
stimulation.
"""


class network2D:
    def __init__(self, params):
        self.label = '2D'
        self.params = {'input_params': params['input_params'],
                       'neuron_params': {'single_neuron_dict': copy(single_neuron_dict)},
                       'connection_params': {'replace_cc': None,
                                             'replace_cc_input_source': None}
                       }
        nested_update(self.params, params)
        self.add_DC_drive = np.zeros(1)
        self.structure = {'A': {'E1', 'E2'}}
        self.structure_vec = ['A-E1', 'A-E2']
        self.area_list = ['A']
        if 'K_stable' in params.keys():
            self.K_matrix = np.array(
                [[params['K_stable'] / 2., params['K_stable'] / 2., params['K']]])
        else:
            self.K_matrix = np.array(
                [[params['K'] / 2., params['K'] / 2., params['K']]])

        self.W_matrix = np.array([[params['W'], params['W'], params['W']]])
        self.J_matrix = convert_syn_weight(self.W_matrix,
                                           self.params['neuron_params']['single_neuron_dict'])

    def Phi(self, rate):
        mu, sigma = self.theory.mu_sigma(rate)
        NP = self.params['neuron_params']['single_neuron_dict']
        return list(map(lambda mu, sigma: nu0_fb(mu, sigma,
                                                 1.e-3 * NP['tau_m'],
                                                 1.e-3 * NP['tau_syn_ex'],
                                                 1.e-3 * NP['t_ref'],
                                                 NP['V_th'] - NP['E_L'],
                                                 NP['V_reset'] - NP['E_L']),
                        mu, sigma))

    def fsolve(self, rates_init):
        def f(rate):
            return self.Phi(rate) - rate
        result = optimize.fsolve(f, rates_init, full_output=1)
        mu, sigma = mu_sigma(result[0], K_matrix=self.K_matrix,
                             J_matrix=self.J_matrix,
                             input_params=self.params['input_params'],
                             single_neuron_params=self.params['neuron_params']['single_neuron_dict'],
                             DC=self.add_DC_drive)
        result_dic = {'rates': np.array([result[0]]), 'mus': np.array(
            [mu]), 'sigmas': np.array([sigma]), 'eps': result[-1], 'time': np.array([0])}
        return result_dic

    def vector_field(self, x_vec, y_vec):
        NP = self.params['neuron_params']['single_neuron_dict']
        vector_matrix_x = np.zeros((len(y_vec), len(x_vec)))
        vector_matrix_y = np.zeros((len(y_vec), len(x_vec)))
        for i, x in enumerate(y_vec):
            for j, y in enumerate(x_vec):
                mu, sigma = mu_sigma([x, y], K_matrix=self.K_matrix,
                             J_matrix=self.J_matrix,
                             input_params=self.params['input_params'],
                             single_neuron_params=self.params['neuron_params']['single_neuron_dict'],
                             DC=self.add_DC_drive)
                new_rates = np.array(
                    list(map(lambda mu, sigma: nu0_fb(mu, sigma,
                                                      1.e-3 * NP['tau_m'],
                                                      1.e-3 * NP['tau_syn_ex'],
                                                      1.e-3 * NP['t_ref'],
                                                      NP['V_th'] - NP['E_L'],
                                                      NP['V_reset'] - NP['E_L']),
                             mu, sigma)))
                vector_matrix_x[i, j] = (new_rates[1] - y)
                vector_matrix_y[i, j] = (new_rates[0] - x)
        x, y = np.meshgrid(x_vec, y_vec)
        return x, y, vector_matrix_x, vector_matrix_y

    def nullclines_x0(self, x0_vec):
        NP = self.params['neuron_params']['single_neuron_dict']

        def nullcline(x0, x1):
            rates = np.zeros(2)
            rates[0] = x0
            rates[1] = x1
            mu, sigma = mu_sigma(rates, K_matrix=self.K_matrix,
                             J_matrix=self.J_matrix,
                             input_params=self.params['input_params'],
                             single_neuron_params=self.params['neuron_params']['single_neuron_dict'],
                             DC=self.add_DC_drive)
            new_rates = np.array(
                list(map(lambda mu, sigma: nu0_fb(mu, sigma,
                                                  1.e-3 * NP['tau_m'],
                                                  1.e-3 * NP['tau_syn_ex'],
                                                  1.e-3 * NP['t_ref'],
                                                  NP['V_th'] - NP['E_L'],
                                                  NP['V_reset'] - NP['E_L']), mu, sigma)))[0]
            return new_rates - x0

        nullcline_x0 = []
        for x0 in x0_vec:
            result = optimize.fsolve(
                lambda x: nullcline(x0, x), 0, full_output=1)
            nullcline_x0.append(result[0][0])
        return nullcline_x0
plt.figure(figsize=(8, 6))
network_params = {'K': 420.,
                  'W': 10.}
input_params = {'rate_ext': 160.}
network_params.update({'input_params': input_params})
x = np.arange(0, 150., 1.)
net = network1D(network_params)
y = np.fromiter([net.Phi(x[j])[0] for j in range(len(x))], dtype=np.float64)
plt.plot(x, y, label='Original params', color='blue')

x_long = np.arange(0, 100000., 500.)
y_long = np.fromiter([net.Phi(x_long[j])[0]
                      for j in range(len(x_long))], dtype=np.float64)
plt.plot(x_long, y_long, label='Long x range', color='red')

input_params = {'rate_ext': 160.}
network_params2 = deepcopy(network_params)
network_params2.update(
    {'neuron_params': {'single_neuron_dict': {'t_ref': 0.}}})
net = network1D(network_params2)
y = np.fromiter([net.Phi(x[j])[0] for j in range(len(x))], dtype=np.float64)
plt.plot(x, y, label='t_ref=0', color='green')
plt.xlabel("Rate input")
plt.ylabel("Phi output")
plt.legend()
plt.grid(True)
plt.title("Phi function for different settings")
plt.xlim(0, 150)
plt.ylim(0, 150)
plt.savefig('network1D_Phi_all.png', dpi=300)