# import numpy as np

# def compute_discrete_function_terms_single_step_euler_2(
#     previous_state_measurement, 
#     previous_control_input, 
#     autonomous_function, 
#     input_function
# ):
#     """
#     Single step Euler integration for input map g(u) and autonomous map f(x).
#     Computes: x_{k+1} = x_k + dt*f(x_k) + dt*g(u_k)
    
#     Args:
#         previous_state_measurement (np.ndarray): Previous state (x_k-1).
#         previous_control_input (np.ndarray): Previous control input (u_k-1).
#         autonomous_function: Function f(x) for the autonomous map.
#         input_function: Function g(u) for the input map.

#     Returns:
#         tuple: (f_discrete, g_discrete, x_discrete)
#             - f_discrete (np.ndarray): Discrete autonomous map f(x_k).
#             - g_discrete (np.ndarray): Discrete input map g(u_k).
#             - x_discrete (np.ndarray): Updated state x_{k+1}.
#     """
    
#     integration_timestep = 0.01

#     # Initialize variables
#     x_discrete = np.zeros_like(previous_state_measurement)
#     f_discrete = np.zeros_like(previous_state_measurement)
#     g_discrete = np.zeros_like(previous_state_measurement)

#     # Compute f_discrete
#     f_discrete = previous_state_measurement + integration_timestep * autonomous_function
    
#     # Compute g_discrete
#     g_discrete = integration_timestep * input_function

#     # Compute x_discrete
#     x_discrete = f_discrete + g_discrete

#     return f_discrete, g_discrete, x_discrete

import numpy as np

def compute_discrete_function_terms_single_step_euler_2(
    previous_state_measurement, 
    previous_control_input, 
    autonomous_function, 
    input_function
):
    """
    Single step Euler integration for discrete dynamics.
    Computes: x_{k+1} = x_k + dt*f(x_k) + dt*g(u_k)*[1, 1, 1]
    
    Args:
        previous_state_measurement (np.ndarray): State vector x_k (3x1).
        previous_control_input (np.ndarray): Control input vector u_k (2x1).
        integration_timestep (float): Timestep dt for Euler integration.
        autonomous_function (np.ndarray): Autonomous dynamics f(x_k) (3x1).
        input_function (np.ndarray): Input map g(u_k) (3x2).

    Returns:
        tuple: (f_discrete, g_discrete, x_discrete)
            - f_discrete (np.ndarray): Discrete autonomous map f(x_k), shape (3x1).
            - g_discrete (np.ndarray): Discrete input map g(u_k), shape (3x2).
            - x_discrete (np.ndarray): Updated state x_{k+1}, shape (3x1).
    """
    integration_timestep= 0.01

    # Compute f_discrete
    f_discrete = previous_state_measurement + integration_timestep * autonomous_function

    # Compute g_discrete
    g_discrete = integration_timestep * input_function

    # Compute x_discrete: add f_discrete and g_discrete summed along its columns
    ones_vector = np.ones(2)  # Vector [1, 1, 1] (3x1)
    x_discrete = f_discrete + g_discrete @ ones_vector.reshape(-1, 1)

    return f_discrete, g_discrete, x_discrete
