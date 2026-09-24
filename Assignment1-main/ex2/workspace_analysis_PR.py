import numpy as np
#import matplotlib.pyplot as plt


# Example usage
# joints = [0.1, 0.3] (m, rad)
# linkLengths = [0.310, 0.1] (m, m)
def endEffector_PR(joints, linkLengths):
    p1 = joints[0]
    theta2 = joints[1]

    l1 = linkLengths[0]
    l2 = linkLengths[1]

    x = l1 + p1 + l2 * np.cos(theta2)
    y = l2 * np.sin(theta2)
    return x, y



# NOTE: for consistency, linklengths [0] here refers to the base link length 
# of the prismatic joint.
def workspace_analysis_PR(
        nSamples = 50, minJoint=[0, 0], 
        maxJoint=[0.1, 2*np.pi], 
        linkLengths=[0.3, 0.1]):

    p_1 = np.linspace(minJoint[0], maxJoint[0], nSamples)
    theta_2 = np.linspace(minJoint[1], maxJoint[1], nSamples)

    xs = np.zeros(nSamples**2)
    ys = np.zeros(nSamples**2)

    for i in range(nSamples):
        for j in range(nSamples):
            joints = [p_1[i], theta_2[j]]
            index = i * nSamples + j
            xs[index], ys[index] = endEffector_PR(joints, linkLengths)

    return xs, ys

