import numpy as np
from scipy.integrate import solve_ivp
from celestial_bodies_data import mu_bodies

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


class TwoBodyProblemPropogator:
    def __init__(self, state_0, tspan, dt, central_body='earth', pertubations=None):
        self.state_0 = state_0  # [km, km, km, km/s, km/s, km/s]

        self.tspan = tspan  # [s]
        self.dt = dt  # [s]

        self.central_body = central_body.lower()

    def __ODE(self, t, state_vec, mu):
        pos_vec = state_vec[:3]
        vel_vec = state_vec[3:]
        acc_vec = -mu * (pos_vec/(np.linalg.norm(pos_vec) ** 3))  # [km/s^2]
        
        return np.hstack([vel_vec, acc_vec])

    def propogate(self, method='RK45', rtol=1e-6, atol=1e-6):
        sol = solve_ivp(
            fun=self.__ODE, t_span=self.tspan, y0=self.state_0, 
            method=method, max_step=self.dt, rtol=rtol, atol=atol, 
            args=(mu_bodies[self.central_body],)
            )
        if sol.status == -1:
            raise RuntimeError(sol.message)

        self.state_history = sol.y.transpose()
        return self.state_history


def plot(state_history, show_plot=True):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(0, 0, 0)
    ax.plot(state_history[0, 0], state_history[0, 1], state_history[0, 2], marker='*', markersize=7)
    ax.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2])

    ax.set_xlabel(r'$\mathbf{r}_1$ [km]')
    ax.set_ylabel(r'$\mathbf{r}_2$ [km]')
    ax.set_zlabel(r'$\mathbf{r}_3$ [km]')

    if show_plot:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    T_0 = 0
    T_F = 3600
    DT = 60
    epochs = np.arange(T_0, T_F, DT)  # [s]

    state_0 = [1e3, 1.2e3, 2e3, 6, 5, 4]  # [km], [km/s]

    tbp_prop = TwoBodyProblemPropogator(
        state_0=state_0, tspan=[epochs[0], epochs[-1]], dt=DT, central_body='earth'
        )
    states = tbp_prop.propogate()
    plot(states)
