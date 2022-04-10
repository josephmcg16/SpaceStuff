import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from celestial_bodies_data import mu_bodies
from utils import *

plt.rcParams['text.usetex'] = True


class TwoBodyProblemPropogator:
    """class for orbit propogation of two body problem trajectory"""

    def __init__(self, state0, t_span,
                 central_body='earth', compute_elems=True):
        self.state0 = state0  # [km, km, km, km/s, km/s, km/s]
        self.states = np.expand_dims(state0, 0)
        if compute_elems:
            self.orbitelems0 = cartesian_to_keplerian(self.states)[0]
            self.orbitelems = np.expand_dims(self.orbitelems0, 0)
            self.orbitelems_names = ['a', 'e', 'i', 'Omega', 'argp', 'nu']

        self.t_span = t_span  # [s]
        self.epochs = [self.t_span[0]]

        self.central_body = central_body.lower()

    @staticmethod
    def ode(epoch, state_vec, mu_body):
        pos_vec = state_vec[:3]
        vel_vec = state_vec[3:]
        acc_vec = -mu_body * (pos_vec/(np.linalg.norm(pos_vec) ** 3))
        return np.hstack([vel_vec, acc_vec])

    def propogate(self, t_eval=None, method='RK45', max_step=np.inf, tol=1e-6):
        """intgrate deterministic dynamics '__ode'"""
        sol = solve_ivp(
            fun=self.ode,
            t_span=self.t_span,
            y0=self.state0,
            method=method,
            t_eval=t_eval, max_step=max_step,
            rtol=tol, atol=tol,
            args=(mu_bodies[self.central_body],)
        )
        if sol.status == -1:
            raise RuntimeError(sol.message)

        self.states = sol.y.transpose()
        self.orbitelems = cartesian_to_keplerian(self.states)

        self.epochs = sol.t
        return self.states


class MonteCarloPropogator():
    def __init__(self, state0, t_span, n_samples=50,
                 central_body='earth', initial_pdf='normal'):
        self.state0 = state0  # [km, km, km, km/s, km/s, km/s]
        self.orbitelems0 = cartesian_to_keplerian(self.state0)[0]

        self.orbitelems_names = ['a', 'e', 'i', 'Omega', 'argp', 'nu']

        self.t_span = t_span  # [s]

        self.central_body = central_body.lower()

        self.n_samples = n_samples

    @staticmethod
    def initial_probability_distributions(
            self, n_samples, noise_std=None):
        if noise_std is None:
            noise_std = dict(zip(self.orbitelems_names, [
                30e-3, 30e-3, 36*4.84814e-6,
                36*4.84814e-6, 36*4.84814e-6, 36*4.84814e-6]))
        self.noise_std = noise_std

        pdf_orbitalelems0 = []
        for (element_name, element_value) in zip(self.orbitelems_names,
                                                 self.orbitelems0):
            pdf_orbitalelems0.append(
                np.random.normal(loc=element_value,
                                 scale=noise_std[element_name],
                                 size=(n_samples)))
        pdf_orbitalelems0 = np.asarray(pdf_orbitalelems0).transpose()
        return pdf_orbitalelems0

    def propogate(self, n_samples=None, noise_std=None, t_eval=None,
                  method='RK45', max_step=np.inf, tol=1e-7):
        if n_samples is not None:
            self.n_samples = n_samples
        self.pdf_orbitalelems0 = self.initial_probability_distributions(
            self, n_samples, noise_std)
        self.pdf_state0 = keplerian_to_cartesian(self.pdf_orbitalelems0)

        state_flows = []
        for k, state0 in enumerate(self.pdf_state0):
            tbp_obj = TwoBodyProblemPropogator(
                state0, self.t_span, self.central_body, compute_elems=False)
            tbp_obj.propogate(t_eval=t_eval, method=method,
                              max_step=max_step, tol=tol)
            state_flows.append(tbp_obj.states)

        # set up interpolation for algorithm chosen sol.t ???
        if t_eval is None:
            self.state_flows = state_flows
            self.means = None
            self.covs = None
        else:
            self.state_flows = np.asarray(state_flows)
            self.means = np.mean(self.state_flows, 0)
            self.covs = build_covariance_tensor(self.state_flows)
        return self.means, self.covs


if __name__ == "__main__":
    np.random.seed(123)
    # epochs [s]
    T0 = 0
    TF = 3600
    DT = 5
    epochs = np.arange(T0, TF, DT)

    a_bounds = [1000, 2000]  # semi-major axis [km]
    e_bounds = [0, 0.5]  # eccentricity [-]
    i_bounds = [0, np.pi]  # inclination [rad]
    Omega_bounds = [0, 2*np.pi]  # right ascension [rad]
    argp_bounds = [0, 2*np.pi]  # argument of periapsis [rad]
    nu_bounds = [0, 2*np.pi]  # true anomaly [rad]

    keplerian_bounds = np.asarray(
        [a_bounds, e_bounds, i_bounds, Omega_bounds, argp_bounds, nu_bounds]
    )

    N_ICS = 50
    cart_ics, kepler_ics = generate_deterministic_initial_conditions(
        N_ICS, keplerian_bounds)

    # loop over ics and get moments for each trajectory
    mean_tensor = []
    cov_tensor = []
    for i, state0 in enumerate(cart_ics):
        mc_tbp = MonteCarloPropogator(state0, [T0, TF])
        means, covs = mc_tbp.propogate(n_samples=20, t_eval=epochs, tol=1e-10)
        mean_tensor.append(means)
        cov_tensor.append(covs)
    mean_tensor = np.asarray(mean_tensor)
    cov_tensor = np.asarray(cov_tensor)
