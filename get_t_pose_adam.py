import math
import time

import numpy as np
import pybullet as p
import pybullet_data

robot_urdf = "/home/jianrenw/skild-gym/skild_gym/env/assets/adam/urdf/adam_v2.urdf"

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
robot_start_pos = [0, 0, 0]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot = p.loadURDF(robot_urdf, robot_start_pos, robot_start_orientation)


num_joints = p.getNumJoints(robot)

joint_names = []

# Iterate through the joints and print their names
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot, joint_index)
    joint_name = joint_info[1].decode(
        "utf-8"
    )  # Decode the byte string to get the joint name
    joint_names.append(joint_name)

print(joint_names)
print(len(joint_names))

t_pose_angles = np.zeros(len(joint_names))

time_step = 1.0 / 240.0
p.setTimeStep(time_step)


def create_joint_frame_visualization(robot, joint_index, frame_size=0.1):
    joint_position, joint_orientation = p.getLinkState(robot, joint_index)[4:6]

    # Calculate the rotation matrix from Euler angles
    rot_matrix = p.getMatrixFromQuaternion(joint_orientation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Extract the rotated axes
    x_axis_end = np.dot(rot_matrix, [frame_size, 0, 0]) + joint_position
    y_axis_end = np.dot(rot_matrix, [0, frame_size, 0]) + joint_position
    z_axis_end = np.dot(rot_matrix, [0, 0, frame_size]) + joint_position

    p.addUserDebugLine(
        joint_position,
        x_axis_end,
        [1, 0, 0],
        parentObjectUniqueId=-1,
        parentLinkIndex=-1,
    )

    # Y-axis (green)
    p.addUserDebugLine(
        joint_position,
        y_axis_end,
        [0, 1, 0],
        parentObjectUniqueId=-1,
        parentLinkIndex=-1,
    )

    # Z-axis (blue)
    p.addUserDebugLine(
        joint_position,
        z_axis_end,
        [0, 0, 1],
        parentObjectUniqueId=-1,
        parentLinkIndex=-1,
    )


# Function to set the T-pose for the robot
def set_t_pose(robot, t_pose_angles):
    # You need to know the joint names and the target angles for the T-pose.
    # Replace these joint names and angles with your robot's configuration.

    for joint_index, angle in enumerate(t_pose_angles):
        if joint_index == 27:
            p.resetJointState(robot, joint_index, -0.3)
        else:
            p.resetJointState(robot, joint_index, angle)
        create_joint_frame_visualization(robot, joint_index)

    joint_position_right_shoulder, _ = p.getLinkState(
        robot, joint_names.index("shoulderRoll_Right")
    )[4:6]
    joint_position_right_elbow, _ = p.getLinkState(
        robot, joint_names.index("elbow_Right")
    )[4:6]
    joint_dis = np.linalg.norm(
        np.array(joint_position_right_shoulder) - np.array(joint_position_right_elbow)
    )
    print("arm", joint_dis)

    joint_position_right_hip, _ = p.getLinkState(
        robot, joint_names.index("hipRoll_Right")
    )[4:6]
    joint_position_right_knee, _ = p.getLinkState(
        robot, joint_names.index("kneePitch_Right")
    )[4:6]
    joint_dis = np.linalg.norm(
        np.array(joint_position_right_hip) - np.array(joint_position_right_knee)
    )
    print("leg", joint_dis)


# Set the robot to the T-pose
set_t_pose(robot, t_pose_angles)

# Run the simulation for a few seconds to visualize the T-pose
time.sleep(1000)
