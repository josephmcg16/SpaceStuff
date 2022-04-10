import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from celestial_bodies_data import mu_bodies
from utils import cartesian_to_keplerian

plt.rcParams['text.usetex'] = True


class TwoBodyProblemPropogator:
    """class for orbit propogation of two body problem trajectory"""
    def __init__(self, state0, t_span,
                 central_body='earth'):
        self.state0 = state0  # [km, km, km, km/s, km/s, km/s]
        self.states = np.expand_dims(state0, 0)
        self.orbitelems = cartesian_to_keplerian(self.states)

        self.t_span = t_span  # [s]
        self.epochs = [self.t_span[0]]

        self.central_body = central_body.lower()

    @staticmethod
    def __ode(epoch, state_vec, mu_body):
        pos_vec = state_vec[:3]
        vel_vec = state_vec[3:]
        acc_vec = -mu_body * (pos_vec/(np.linalg.norm(pos_vec) ** 3))
        return np.hstack([vel_vec, acc_vec])

    def propogate(self, t_eval=None, method='RK45', max_step=np.inf, tol=1e-6):
        """intgrate deterministic dynamics '__ode'"""
        sol = solve_ivp(
            fun=self.__ode,
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
        self.orbitelems = dict(zip(
            ['a', 'e', 'i', 'Omega', 'argp', 'nu'],
            cartesian_to_keplerian(self.states)))

        self.epochs = sol.t
        return self.states

    def plot_trajectory(self, show_orbitelems=False):
        """3d plot of a single trajectory"""
        fig, axes = plt.subplots(subplot_kw={'projection': '3d'})

        axes.scatter(0, 0, 0)
        axes.plot(self.states[0, 0], self.states[0, 1], self.states[0, 2],
                  marker='*', markersize=7)
        axes.plot(self.states[:, 0], self.states[:, 1], self.states[:, 2])

        if show_orbitelems:
            axes.text2D(0.05, 0.95,
                        f"a={self.orbitelems['a']},"
                        f"e={self.orbitelems['e']},"
                        f"i={self.orbitelems['i']}\n"
                        f"Omega={self.orbitelems['Omega']},"
                        f"nu={self.orbitelems['nu']}")

        axes.set_xlabel(r'$\mathbf{r}_1$ [km]')
        axes.set_ylabel(r'$\mathbf{r}_2$ [km]')
        axes.set_zlabel(r'$\mathbf{r}_3$ [km]')
        return fig, axes


if __name__ == "__main__":
    # epochs [s]cl
    T_0 = 0
    T_F = 360  # 1 week

    state_0 = [1e3, 1.2e3, 2e3, 6, 5, 4]  # [km], [km/s]
    tbp_prop = TwoBodyProblemPropogator(
        state0=state_0, t_span=[T_0, T_F], central_body='earth'
        )

    tbp_prop.propogate(tol=1e-10)
    states = tbp_prop.states
    orbitelems = tbp_prop.orbitelems
    epochs = tbp_prop.epochs

    tbp_prop.plot_trajectory(show_orbitelems=True)
    plt.show()
