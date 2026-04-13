import math
import numpy as np
import scipy

def forward_kinematics(theta1, theta2, theta3):
    def rotation_x(angle):
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])

    def rotation_y(angle):
        return np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])

    def rotation_z(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def translation(x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    T_0_1 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta1)
    T_1_2 = rotation_y(-1.57080) @ rotation_z(theta2)
    T_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta3)
    T_3_ee = translation(0.06231, -0.06216, 0.01800)
    T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
    return T_0_ee[:3, 3]

def get_error_leg(theta, desired_position):
    return np.linalg.norm(desired_position - forward_kinematics(theta[0], theta[1], theta[2]))

def inverse_kinematics_with_optimizer(target):

    inital_guess = [0,0,0]

    res = scipy.optimize.minimize(
        fun=get_error_leg,
        x0=inital_guess,
        args=(target,),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12}
    )

    if not res.success:
        res = scipy.optimize.minimize(
                fun=get_error_leg,
                x0=inital_guess,
                args=(target,),
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10}
        )


    return res.x


def inverse_kinematics_with_gradient(target):
    joint_angles = np.array([0.0,0.0,0.0])
    learning_rate = 13.85
    max_steps = 500
    tolerance = 1e-10

    for step in range(max_steps):
        C, mean_err = get_cost(joint_angles, target)

        if mean_err < tolerance:
            break

        C_grad = get_gradient(joint_angles, target)

        joint_angles -= learning_rate * C_grad

    return joint_angles



def get_cost(joint_angles, target_ee):
    current_pos = forward_kinematics(joint_angles[0], joint_angles[1], joint_angles[2])
    error = target_ee - current_pos
    C = np.sum(np.square(error))
    mean_err = np.mean(np.abs(error))
    
    return C, mean_err


def get_gradient(joint_angles, target_ee):
    epsilon = 1e-6
    grad = np.zeros(3)

    for i in range(3):
        angles_plus = np.copy(joint_angles)
        angles_minus = np.copy(joint_angles)
        
        angles_plus[i] += epsilon
        angles_minus[i] -= epsilon
        
        cost_plus, _ = get_cost(angles_plus, target_ee)
        cost_minus, _ = get_cost(angles_minus, target_ee)
        
        grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
        
    return grad
