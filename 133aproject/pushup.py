'''hw6p5.py

   This is the skeleton code for HW6 Problem 5.

   This explores the singularity handing while following a circle
   outside the workspace.

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from hw6code.GeneratorNode     import GeneratorNode
from hw6code.KinematicChain    import KinematicChain
from hw5code.TransformHelpers  import *


def spline(t, T, p0, pf):
    p = p0 + (pf-p0) * (3*t**2/T**2 - 3*t**3/T**3)
    v =      (pf-p0) * (6*t   /T**2 - 6*t**2/T**3)
    return (p, v)

class Q:
    def __init__(self, joint_names) -> None:
        self.joint_names = joint_names
        self.joint_values = dict([(joint_name, 0) for joint_name in joint_names])
        
    def setAll(self, value):
        self.joint_values = dict([(joint_name, value) for joint_name in self.joint_names])

    def setSome(self, joints, values):
        if isinstance(values, np.ndarray):
            values = values.flatten()

        for (i, joint) in enumerate(joints):
            if (joint not in self.joint_values):
                raise IndexError("Invalid Jointname pass into set")
            self.joint_values[joint] = values[i]

    def retSome(self, joints):
        return np.array([self.joint_values[joint] for joint in joints]).reshape((-1, 1))

    def retAll(self):
        return self.retSome(self.joint_names)

    def __len__(self):
        return len(self.joint_names)
        
    def __str__(self):
        return str(self.joint_values)

class M(Q):
    def getMatrix(self, joint_names):
        return np.diag(np.array([self.joint_values[name] for name in joint_names]))

class Jacobian():
    def __init__(self, joints, chain) -> None:
        Jv = chain.Jv()
        Jw = chain.Jw()
        J = np.vstack((Jv, Jw))
        self.Jacobian = J
        self.joints = joints
        self.columns = dict([(joint, J[:, i]) for (i, joint) in enumerate(self.joints)])

    def merge(Js, joints):
        columns = len(joints)
        rows = sum([J.Jacobian.shape[0] for J in Js])
        mergedJ = np.zeros((rows, columns))

        rowInput = 0
        for J in Js:
            size = J.Jacobian.shape[0]
            for (columnNo, joint) in enumerate(joints):
                if joint in J.columns:
                    mergedJ[rowInput:rowInput + size, columnNo] = J.columns[joint]
                
            rowInput += size
                
        return mergedJ

class X():
    def calculateError(self, currentValues, desiredValues):
        # desiredValues = [(chain1p, chain1r), (chain2p, chain2r), ...]
        errors = [(ep(desP, tipP), eR(desR, tipR)) for ((tipP, tipR), (desP, desR)) in zip(currentValues, desiredValues)]
        flatErrors = [item for sublist in errors for item in sublist]
        return np.vstack(tuple(flatErrors))

class SetBounds():
    def __init__(self, min, max, jointName, exp=4) -> None:
        self.jointName = jointName
        self.min = min
        self.max = max
        self.exp = exp

    def calcQDot(self, capQ):
        # Need to pass in Q object 
        q = capQ.retSome([self.jointName])[0]
        qDotSec = Q(capQ.joint_names)
        qDotSec.setAll(0.0)
        qDotParticular = self.exp * (self.T(q) ** (self.exp - 1)) * self.TPrime(q)
        qDotSec.setSome([self.jointName], np.array(qDotParticular))

        return qDotSec.retAll()

    def T(self, q):
        return (q - (self.min + self.max)/2) / ((self.max - self.min) / 2)

    def TPrime(self, q):
        return (q) / ((self.max - self.min) / 2)

    def Error(self, capQ):
        q = capQ.retSome([self.jointName])[0]
        return self.T(q)

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        
        self.chain_world_pelvis = KinematicChain(node, 'world', 'pelvis', self.jointnames('world_pelvis'))
        self.chain_world_chest = KinematicChain(node, 'world', 'chest', self.jointnames('world_chest'))
        
        self.chain_left_arm = KinematicChain(node, 'world', 'l_hand', self.jointnames('left_arm'))
        self.chain_right_arm = KinematicChain(node, 'world', 'r_hand', self.jointnames('right_arm'))
        self.chain_left_leg = KinematicChain(node, 'world', 'l_foot', self.jointnames('left_leg'))
        self.chain_right_leg = KinematicChain(node, 'world', 'r_foot', self.jointnames('right_leg'))
        
        # Initialize the current joint position and chain data.
        # TODO Initialize all the chains 

        self.Q = Q(self.jointnames())
        self.Qdot = Q(self.jointnames())
        
        self.Qdot.setAll(0)
        self.Q.setAll(0)
        self.Q.setSome(['r_arm_shx', 'l_arm_shx', 'r_arm_shz', 'l_arm_shz', 'rotate_y', 'mov_z'], np.array([0.25, -0.25, np.pi/2, -np.pi/2, 0.95, 0.51]))
        
        self.chain_left_arm.setjoints(self.Q.retSome(self.jointnames('left_arm')))
        self.chain_right_arm.setjoints(self.Q.retSome(self.jointnames('right_arm')))
        self.chain_left_leg.setjoints(self.Q.retSome(self.jointnames('left_leg')))
        self.chain_right_leg.setjoints(self.Q.retSome(self.jointnames('right_leg')))
        
        self.chain_world_chest.setjoints(self.Q.retSome(self.jointnames('world_chest')))
        self.chain_world_pelvis.setjoints(self.Q.retSome(self.jointnames('world_pelvis')))

        self.gamma = 0.05

        self.M = M(self.jointnames())
        self.M.setAll(1)
        
        # Also zero the task error.
        self.err = np.zeros((30, 1))

        # Pick the convergence bandwidth.
        self.lam = 30
        self.X = X()

        self.Bounds = [SetBounds(-2.35, 0, 'r_arm_elx'), SetBounds(0, 2.35, 'l_arm_elx'), SetBounds(0, 3.14, 'r_arm_ely'), SetBounds(0, 3.14, 'l_arm_ely')]

    # Declare the joint names.
    def jointnames(self, which_chain='all'):
        # Return a list of joint names FOR THE EXPECTED URDF!
        # TODO Implement better method to find the jointnames
        
        joints = {
        
        'left_arm':['back_bkz', 'back_bky', 'back_bkx', 'l_arm_shz','l_arm_shx', 'l_arm_ely', 'l_arm_elx', 'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2'],
        
         'right_arm': ['back_bkz', 'back_bky', 'back_bkx', 'r_arm_shz','r_arm_shx', 'r_arm_ely', 'r_arm_elx', 'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2'],
         
         'left_leg': ['l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky', 'l_leg_akx'] , 
         
         'right_leg':['r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx'], 
         
         'world_pelvis':['mov_x', 'mov_y', 'mov_z', 'rotate_x', 'rotate_y', 'rotate_z'],
         'world_chest':['mov_x', 'mov_y', 'mov_z', 'rotate_x', 'rotate_y', 'rotate_z', 'back_bkz', 'back_bky', 'back_bkx']
         
        }
         
        if which_chain == 'all':
            return joints['left_arm'] + joints['right_arm'][3:] + joints['left_leg'] + joints['right_leg'] + joints['world_pelvis']
         
        if which_chain == 'world_pelvis':
            return joints['world_pelvis']
        if which_chain == 'world_chest':
            return joints['world_chest']
             
        return joints['world_pelvis'] + joints[which_chain]
        
    def quat_to_angle(self, quat):
        q0 = quat[0]
        q1 = quat[1]
        q2 = quat[2]
        q3 = quat[3]
        
        r00 = 2 * (q0 ** 2 + q1**2) - 1
        r01 = 2*(q1*q2 - q0*q3)
        r02 = 2*(q1*q3 + q0*q2)
        
        r10 = 2*(q1*q2 + q0*q3)
        r11 = 2*(q0**2 + q2**2) - 1
        r12 = 2*(q2*q3 - q0*q1)
        
        r20 = 2*(q1*q3 - q0*q2)
        r21 = 2*(q2*q3 + q0*q1)
        r22 = 2*(q0**2 + q3**3) - 1
        
        return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    
    def chest_pos(self, t, period):
    	# add explicit length of pushup 
        s = 0.3 * np.cos((np.pi/period)* t)
        orient = R_from_quat(np.array([0.889, 0, 0.4573, 0]))
        return (np.array([0.4661, 0, 0.8587 - 0.3 + s]).reshape((-1,1)), orient)
    
    def chest_vel(self, t, period):
        sdot = - 0.3 * (np.pi/period) * np.sin((np.pi/period) * t)
        return (np.array([0, 0, sdot]).reshape((-1,1)), np.array([0, 0, 0]).reshape((-1,1)))
    	
    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        self.period = 2
    
        leftLegPos = (np.array([-0.704, 0.115, 0.0085]).reshape((-1, 1)), R_from_quat(np.array([0.8892, 0, 0.4573, 0])))
        leftArmPos = (np.array([0.704, 0.226, 0.00474]).reshape((-1, 1)), R_from_quat(np.array([0.583, -0.399, 0.399, -0.583])))
        rightLegPos = (np.array([-0.704, -0.115, 0.0085]).reshape((-1, 1)), R_from_quat(np.array([0.8892, 0, 0.4573, 0])))
        rightArmPos = (np.array([0.704, -0.226, 0.0047]).reshape((-1, 1)), R_from_quat(np.array([0.583, 0.399, 0.399, 0.583])))
        chestPos = self.chest_pos(t, self.period) # changing
        
        # Grab the last joint value and task error.
        
        q = self.Q.retAll()
        err = self.err
        
        # Compute the inverse kinematics

        J_left_arm = Jacobian(self.jointnames('left_arm'), self.chain_left_arm)
        J_right_arm = Jacobian(self.jointnames('right_arm'), self.chain_right_arm)
        J_left_leg = Jacobian(self.jointnames('left_leg'), self.chain_left_leg)
        J_right_leg = Jacobian(self.jointnames('right_leg'), self.chain_right_leg)
        J_chest = Jacobian(self.jointnames('world_chest'), self.chain_world_chest)

        JMerged = Jacobian.merge([J_left_arm, J_right_arm, J_left_leg, J_right_leg, J_chest], self.jointnames())

        xdot = np.zeros((30, 1))
        xdot[24:27, :] = self.chest_vel(t, self.period)[0]
        
        M = self.M.getMatrix(self.jointnames())
        MSI = np.linalg.inv(M @ M) # M squared inverse
        JInv = MSI @ JMerged.T @ np.linalg.inv(JMerged @ MSI @ JMerged.T + self.gamma**2 * np.eye(30))
        qdot = JInv @ (xdot + self.lam * err)

        # Integrate the joint position and update the kin chain data.
        qsec = Q(self.jointnames())
        qsec.setSome(self.jointnames(), q)
        qsec.setSome(['l_leg_kny', 'r_leg_kny'], [0.45, 0.45]) # keep the knees sort of bent
        
        # sdot = (np.identity(len(q)) - JInv @ JMerged) @ (0.05 * (qsec.retAll() - q))
        # qdot += sdot

        # Centering the joints (Pushing from limits) TODO FIX 

        # qdotSec = sum([sb.calcQDot(self.Q) for sb in self.Bounds])
        # sdot = (np.identity(len(q)) - JInv @ JMerged) @ ((0.3 * qdotSec))

        # print(f"Errors {[sb.Error(self.Q) for sb in self.Bounds]}")
    
        # qdot = qdot + sdot

        q = q + dt * qdot
        self.Q.setSome(self.jointnames(), q)
        self.Qdot.setAll(qdot)

        self.chain_left_arm.setjoints(self.Q.retSome(self.jointnames('left_arm')))
        self.chain_right_arm.setjoints(self.Q.retSome(self.jointnames('right_arm')))
        self.chain_left_leg.setjoints(self.Q.retSome(self.jointnames('left_leg')))
        self.chain_right_leg.setjoints(self.Q.retSome(self.jointnames('right_leg')))
        #self.chain_world_pelvis.setjoints(self.Q.retSome(self.jointnames('world_pelvis')))
        self.chain_world_chest.setjoints(self.Q.retSome(self.jointnames('world_chest')))
        
        # Compute the resulting task error (to be used next cycle).

        chains = [self.chain_left_arm, self.chain_right_arm, self.chain_left_leg, self.chain_right_leg, self.chain_world_chest]
        tipPositions = [(c.ptip(), c.Rtip()) for c in chains]

        err = self.X.calculateError(tipPositions, [leftArmPos, rightArmPos, leftLegPos, rightLegPos, chestPos])
        # err = np.zeros((24, 1))


        # Save the joint value and task error for the next cycle.
        self.err = err

        # Return the position and velocity as python lists.
        # return (q.flatten().tolist(), qdot.flatten().tolist())       
        # test code
        q_all = self.Q.retAll()
        qdot_all = self.Qdot.retAll()

        wristValues = self.Q.retSome(['l_arm_ely', 'l_arm_elx', 'r_arm_ely', 'r_arm_elx'])
        print(f"Wrist Values: Left, Right {wristValues}")

        self.Q_still = Q(self.jointnames())
        self.Qdot_still = Q(self.jointnames())
        
        self.Qdot_still.setAll(0.0)
        self.Q_still.setAll(0.0)
        self.Q_still.setSome(['r_arm_shx', 'l_arm_shx', 'r_arm_shz', 'l_arm_shz', 'rotate_y', 'mov_z'], np.array([0.25, -0.25, np.pi/2, -np.pi/2, 0.95, 0.51]))
    
        # q_all = self.Q_still.retAll()
        # qdot_all = self.Qdot_still.retAll()
        
        return (q_all.flatten().tolist(), qdot_all.flatten().tolist())

        


#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the generator node (100Hz) for the Trajectory.
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

