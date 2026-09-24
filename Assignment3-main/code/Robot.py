import numpy as np

class Robot:
    """
    Represents a general fixed-base kinematic chain.
    """

    def __init__(self, link_lengths, link_masses, joint_masses, end_effector_mass):
        """
        Constructor: Makes a brand new robot with the specified parameters.
        """
        # Validate input
        if link_lengths.shape[1] != 1:
            raise ValueError(f"Invalid link_lengths: Should be a column vector, is {link_lengths.shape[0]}x{link_lengths.shape[1]}.")
        
        if link_masses.shape[1] != 1:
            raise ValueError(f"Invalid link_masses: Should be a column vector, is {link_masses.shape[0]}x{link_masses.shape[1]}.")
        
        if joint_masses.shape[1] != 1:
            raise ValueError("Invalid joint_masses: Should be a column vector.")
        
        if not isinstance(end_effector_mass, (int, float)):
            raise ValueError("Invalid end_effector_mass: Should be a number.")
        
        self.dof = link_lengths.shape[0]
        
        if link_masses.shape[0] != self.dof:
            raise ValueError("Invalid number of link masses: should match number of link lengths.")
        
        if joint_masses.shape[0] != self.dof:
            raise ValueError("Invalid number of joint masses: should match number of link lengths. Did you forget the base joint?")
        
        self.link_lengths = link_lengths
        self.link_masses = link_masses
        self.joint_masses = joint_masses
        self.end_effector_mass = end_effector_mass

    def forward_kinematics(self, thetas):
        """
        Returns the forward kinematic map for each frame, one for the base of
        each link, and one for the end effector. Link i is given by
        frames[:,:,i], and the end effector frame is frames[:,:,-1].
        """
        if thetas.shape[1] != 1:
            raise ValueError("Expecting a column vector of joint angles.")
        
        if thetas.shape[0] != self.dof:
            raise ValueError(f"Invalid number of joints: {thetas.shape[0]} found, expecting {self.dof}")
        
        # Allocate a variable containing the transforms from each frame
        # to the base frame.
        frames = np.zeros((3, 3, self.dof + 1))
        n = self.dof
        
        # TODO:  FILL IN 3x3 HOMOGENEOUS TRANSFORM FOR n + 1 FRAMES
        # You'll need to implement the forward kinematics calculation here
        # base frame
        theta = thetas[0, 0]
        length = self.link_lengths[0, 0]

        frames[:, :, 1] = np.array([
            [np.cos(theta), -np.sin(theta), length * np.cos(theta)],
            [np.sin(theta),  np.cos(theta), length * np.sin(theta)],
            [0,              0,              1]
        ])

        # middle frames
        for i in range(1, n - 1):
            theta = thetas[i, 0]
            length = self.link_lengths[i, 0]

            H = np.array([
                [np.cos(theta), -np.sin(theta), length * np.cos(theta)],
                [np.sin(theta),  np.cos(theta),  length * np.sin(theta)],
                [0,              0,               1]
            ])

            frames[:, :, i + 1] = frames[:, :, i] @ H

        # end effector frame
        theta = thetas[n - 1, 0]
        length = self.link_lengths[n - 1, 0]

        H = np.array([
            [np.cos(theta), -np.sin(theta), length * np.cos(theta)],
            [np.sin(theta),  np.cos(theta),  length * np.sin(theta)],
            [0,              0,               1]
        ])

        frames[:, :, n] = frames[:, :, n - 1] @ H
        
        return frames

    def fk(self, thetas):
        """
        Shorthand for returning the forward kinematics.
        """
        return self.forward_kinematics(thetas)

    def end_effector(self, thetas):
        """
        Returns [x; y; theta] for the end effector given a set of joint angles.
        """
        # Find the transform to the end-effector frame.
        frames = self.fk(thetas)
        H_0_ee = frames[:,:,-1]
        
        # Extract the components of the end_effector position and orientation.
        x = H_0_ee[0,2]
        y = H_0_ee[1,2]
        th = np.arctan2(H_0_ee[1,0], H_0_ee[0,0])
        
        # Pack them up nicely.
        return np.array([x, y, th])

    def ee(self, thetas):
        """
        Shorthand for returning the end effector position and orientation.
        """
        return self.end_effector(thetas)