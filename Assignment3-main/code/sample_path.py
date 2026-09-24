import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot

def sample_path(log_name=None):
    # Load log data (allow for sample data as well as arbitrary logs)
    if log_name is None:
        # Default to sample-log
        log = np.genfromtxt('sample_ground_truth.csv', delimiter=',', skip_header=1)
        theta = log[:, 0:2]
        gt_x = log[:, 2]
        gt_y = log[:, 3]
        show_ground_truth = True
    else:
        # For arbitrary logs, you'll need to implement a way to load them
        theta = np.genfromtxt(log_name, delimiter=',', skip_header=1)
        show_ground_truth = False

    # Calculate end effector path through entire log
    n = len(theta)
    x = np.zeros(n)
    y = np.zeros(n)
    robot = Robot(np.array([[0.55], [0.4]]), np.array([[1], [1]]), np.array([[1], [1]]), 0)

    # calculate end effector x/y for each timestep
    for i in range(n):
        frames = robot.fk(theta[i,:].reshape(-1,1))
        x[i] = frames[0, 2, -1]
        y[i] = frames[1, 2, -1]

    # Save calculated path
    calculated_path = np.column_stack((x, y))
    np.savetxt('calculated_path.csv', calculated_path, delimiter=',', header='x,y', comments='')

    d = 1  # Limit the range to +/- 1 meter by default

    if show_ground_truth:
        # Save ground truth
        ground_truth_path = np.column_stack((gt_x, gt_y))
        np.savetxt('ground_truth_path.csv', ground_truth_path, delimiter=',', header='x,y', comments='')

        # Side-by-side subplots so the two paths are easy to compare, instead of
        # overlapping on a single plot.
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Plot of end effector position over a sample run.')

        ax1.plot(x, y, 'k-', linewidth=1)
        ax1.set_title('Your Kinematics')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.axis('equal')
        ax1.set_xlim([-d, d])
        ax1.set_ylim([-d, d])

        ax2.plot(gt_x, gt_y, 'g--', linewidth=1)
        ax2.set_title('Correct Kinematics')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('y [m]')
        ax2.axis('equal')
        ax2.set_xlim([-d, d])
        ax2.set_ylim([-d, d])
    else:
        plt.figure()
        plt.plot(x, y, 'k-', linewidth=1)
        plt.title('Plot of end effector position over a sample run.')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.xlim([-d, d])
        plt.ylim([-d, d])

    plt.show()

sample_path()
