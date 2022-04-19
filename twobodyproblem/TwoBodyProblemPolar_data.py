import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from twobodypropogator import TwoBodyProblemPropogator, keplerian_ics_doe
from celestial_bodies_data import mu_bodies


def observation(state0, T0, t, tol):
    tbp = TwoBodyProblemPropogator(state0, [T0, t], ode='polar_ode')
    final_state = tbp.propogate(tol=tol)[-1]
    return final_state


if __name__ == '__main__':
    # SCRIPT PARAMETERS
    N_ICS_TRAINING_FILE = 600
    N_ICS_TESTING_FILE = 600
    N_ICS_VALIDATION_FILE = 5000
    NUM_TRAINING_FILES = 20
    NUM_TESTING_FILES = 5

    SAVE_PREFIX = "./data/TwoBodyProblemPolar"

    N_STEPS = 501  # num of time steps in a trajectory
    TOL = 1e-7

    # KEPLERIAN BOUNDS
    a_bounds = [7500, 9000]  # semi-major axis [km]
    e_bounds = [0, 0.05]  # eccentricity [-]
    i_bounds = [0, 0]  # inclination [rad]
    Omega_bounds = [0, 2*np.pi]  # right ascension [rad]
    argp_bounds = [0, 0]  # argument of periapsis [rad]
    nu_bounds = [0, 2*np.pi]  # true anomaly [rad]

    keplerian_bounds = np.asarray([
        a_bounds, e_bounds, i_bounds,
        Omega_bounds, argp_bounds, nu_bounds
    ])

    # TIME STEPS
    T0 = 0
    TF = 2*np.pi * np.sqrt(a_bounds[-1]**3 / mu_bodies['earth'])
    TIME = np.linspace(T0, TF, N_STEPS)

    # TRAINING FILES
    DATA_FILE_PREFIX = f'{SAVE_PREFIX}_train'

    print('\nGenerating Training Files ...\n------------------------------\n')
    for train_num in tqdm(range(NUM_TRAINING_FILES), desc='Files'):
        # generate initial conditions
        ics = keplerian_ics_doe(
            N_ICS_TRAINING_FILE, keplerian_bounds, coords='polar')[0]

        # propogate each trajectory
        states = []
        for state0 in tqdm(ics, desc='Trajectory', leave=False):
            states.append(
                np.asarray(
                    [observation(state0, T0, t, TOL)
                     for t in TIME]
                )
            )
        states = np.asarray(states)

        # save file
        np.save(f'{DATA_FILE_PREFIX}_{train_num+1}_x.npy', states)
    print(f'Training Files saved to {DATA_FILE_PREFIX}...\n')

    # TESTING FILES
    DATA_FILE_PREFIX = f'{SAVE_PREFIX}_test'

    print('\nGenerating Testing Files ...\n------------------------------\n')
    for test_num in tqdm(range(NUM_TESTING_FILES), desc='Files'):
        # generate initial conditions
        ics = keplerian_ics_doe(
            N_ICS_TESTING_FILE, keplerian_bounds, coords='polar')[0]

        # propogate each trajectory
        states = []
        for state0 in tqdm(ics, desc='Trajectory', leave=False):
            states.append(
                np.asarray(
                    [observation(state0, T0, t, TOL)
                     for t in TIME]
                )
            )
        states = np.asarray(states)

        # save file
        np.save(f'{DATA_FILE_PREFIX}_{test_num+1}_x.npy', states)
    print(f'Testing Files saved to {DATA_FILE_PREFIX}...\n')

    # VALIDATION FILE
    DATA_FILE_PREFIX = f'{SAVE_PREFIX}_val'

    print('\nGenerating Validation File ...\n------------------------------\n')

    # generate initial conditions
    ics = keplerian_ics_doe(
        N_ICS_VALIDATION_FILE, keplerian_bounds, coords='polar')[0]

    # propogate each trajectory
    states = []
    for state0 in tqdm(ics, desc='Trajectory', leave=False):
        states.append(
            np.asarray(
                [observation(state0, T0, t, TOL)
                    for t in TIME]
            )
        )
    states = np.asarray(states)

    # save file
    np.save(f'{DATA_FILE_PREFIX}_x.npy', states)
    print(f'Validation File saved to {DATA_FILE_PREFIX}...\n')
    print('Complete.')
