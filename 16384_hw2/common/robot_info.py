import numpy as np
import os

def robot_info():
    robot_info = {}
    # length of the links
    robot_info['link_lengths'] = np.array([0.3280, 0.3160])  # It is your responsibility to measure the links
    # masses [kg]
    # (density of link tube is .1213 kg/m, with additional .165 kg of hardware at ends)
    robot_info['link_masses'] = robot_info['link_lengths'] * 0.1213 + np.array([.165, .165])
    robot_info['joint_masses'] = np.array([0.347, 0.3])
    robot_info['end_effector_mass'] = 0  # Nothing for the end effector for this lab!
    return robot_info

def load_data():
    """Joint angles and joint velocities recorded from a run of the RR arm."""
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data.csv'), delimiter=',')
    return {
        'theta': data[:, 0:2],
        'theta_dot': data[:, 2:4],
    }

# Column order of solution.csv, matching the keys returned by load_solution().
SOLUTION_COLUMNS = [
    'com1_x', 'com1_y', 'com1_x_dot', 'com1_y_dot',
    'end1_x', 'end1_y', 'end1_x_dot', 'end1_y_dot',
    'com2_x', 'com2_y', 'com2_x_dot', 'com2_y_dot',
    'end2_x', 'end2_y', 'end2_x_dot', 'end2_y_dot',
    'tau_force_1', 'tau_force_2',
    'tau_grav_comp_1', 'tau_grav_comp_2',
]

def load_solution():
    """Reference answers for every row of data.csv, keyed by column name.

    Used by local_check.py so you can verify your own work before submitting.
    """
    path = os.path.join(os.path.dirname(__file__), 'solution.csv')
    data = np.loadtxt(path, delimiter=',')
    return {name: data[:, i] for i, name in enumerate(SOLUTION_COLUMNS)}
