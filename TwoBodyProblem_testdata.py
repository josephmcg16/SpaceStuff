"""Create test data for TwoBodyProblem"""

from tkinter.ttk import Progressbar
import numpy as np
from tqdm import tqdm

from twobodyproblem.twobodypropogator import keplerian_ics_doe, \
                                             TwoBodyProblemPropogator


if __name__ == "__main__":
    DATA_PREFIX = 'TwoBodyProblem'
    NUM_OF_ICS = 1000  # number of initial condition in each file
    NUM_OF_TEST = 5  # number of testing data files

    # TIME STEPS [s]
    T_0 = 0
    T_F = 3600
    DT = 5
    EPOCHS = np.arange(T_0, T_F, DT)

    # GENERATE INITIAL CONDITIONS
    def generate_initial_conditions(n_ics):
        a_bounds = [1000, 2000]  # semi-major axis [km]
        e_bounds = [0, 0.5]  # eccentricity [-]
        i_bounds = [0, np.pi]  # inclination [rad]
        Omega_bounds = [0, 2*np.pi]  # right ascension [rad]
        argp_bounds = [0, 2*np.pi]  # argument of periapsis [rad]
        nu_bounds = [0, 2*np.pi]  # true anomaly [rad]

        bounds = np.asarray(
            [a_bounds, e_bounds, i_bounds,
             Omega_bounds, argp_bounds, nu_bounds]
        )

        return keplerian_ics_doe(n_ics, bounds, 'earth')[0]

    # LOOP OVER TESTING FILES
    for test_num in tqdm(range(NUM_OF_TEST), desc="Files     "):
        initial_conditions = generate_initial_conditions(NUM_OF_ICS)

        # PROPOGATE TRAJECTORIES
        states = []
        for state0 in tqdm(initial_conditions, leave=False, desc="Trajectory"):
            tbp = TwoBodyProblemPropogator(state0, [T_0, T_F], 'earth')
            states.append(tbp.propogate(EPOCHS, tol=1e-10))

        states = np.asarray(states)

        # SAVE TRAJECTORIES AS .NPY DATA FILE
        DATA_SET = f'train{test_num}_x'
        np.save(f'data\\{DATA_PREFIX}_{DATA_SET}.npy', states)
