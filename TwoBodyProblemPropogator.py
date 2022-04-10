import numpy as np
from scipy.integrate import solve_ivp
from celestial_bodies_data import mu_bodies
from utils import cartesian_to_keplerian, keplerian_to_cartesian

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


class TwoBodyProblemPropogator:
    def __init__(self, state0, tspan, dt=np.inf, central_body='earth', pertubations=None):
        self.state0 = state0  # [km, km, km, km/s, km/s, km/s]
        self.orbitelems = cartesian_to_keplerian(self.state0)

        self.t_span = tspan  # [s]
        self.dt = dt  # [s]

        self.central_body = central_body.lower()

    def __ODE(self, t, state_vec, mu):
        pos_vec = state_vec[:3]
        vel_vec = state_vec[3:]
        acc_vec = -mu * (pos_vec/(np.linalg.norm(pos_vec) ** 3))  # [km/s^2]
        
        return np.hstack([vel_vec, acc_vec])

    def propogate(self, t_eval=None, method='RK45', tol=1e-6):
        sol = solve_ivp(
            fun=self.__ODE, t_span=self.t_span, y0=self.state0, 
            method=method, t_eval=t_eval, max_step=self.dt, rtol=tol, atol=tol,
            args=(mu_bodies[self.central_body],)
            )
        if sol.status == -1:
            raise RuntimeError(sol.message)

        self.states = sol.y.transpose()
        self.orbitelems = cartesian_to_keplerian(self.states)

        self.epochs = sol.t
        return self.states

    def plot_trajectory(self, disp_orbitelems=False):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        ax.scatter(0, 0, 0)
        ax.plot(self.states[0, 0], self.states[0, 1],
                self.states[0, 2], marker='*', markersize=7)
        ax.plot(self.states[:, 0], self.states[:, 1], self.states[:, 2])

        if disp_orbitelems:
            ax.text2D(0.05, 0.95, 
            "$a$={0:3.1f}km, $e$={1:.3f}, $i$={2:.3f}rad\n $\Omega$={3:.3f}rad, $\omega$={4:.3f}rad, $\\nu$={5:.3f}rad,".format(
                *self.orbitelems[-1]), transform=ax.transAxes)

        ax.set_xlabel(r'$\mathbf{r}_1$ [km]')
        ax.set_ylabel(r'$\mathbf{r}_2$ [km]')
        ax.set_zlabel(r'$\mathbf{r}_3$ [km]')
        return fig, ax


if __name__ == "__main__":
    # epochs [s]cl
    T_0 = 0
    T_F = 3600 #*24*7  # 1 week

    state_0 = [1e3, 1.2e3, 2e3, 6, 5, 4]  # [km], [km/s]
    tbp_prop = TwoBodyProblemPropogator(
        state0=state_0, tspan=[T_0, T_F], central_body='earth'
        )

    tbp_prop.propogate(tol=1e-10)
    states = tbp_prop.states
    orbitelems = tbp_prop.orbitelems
    epochs = tbp_prop.epochs

    tbp_prop.plot_trajectory()
    plt.show()
