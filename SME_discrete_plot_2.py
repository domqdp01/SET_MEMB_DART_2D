import pandas as pd
import numpy as np
from scipy.optimize import linprog
from funct_fin import steer_angle, rolling_friction, motor_force, F_friction_due_to_steering, slip_angles, lateral_tire_forces, friction
from discretization_function_2 import compute_discrete_function_terms_single_step_euler_2
from continuos_matrix_2 import continuous_matrices_2
from Fucntion_2_D import compute_vertices
from direction import generate_orthogonal_directions
from underapproximate_convex_politope import underapproximate_convex_polytope
import matplotlib.pyplot as plt


# # # # ============================================================================================== # # #
# # # #                                  IMPORTING AND LOADING DATA                                    # # #
# # # #                                    IN CONTINUOUS TIME                                          # # #
# # # # ============================================================================================== # # #


# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_43_23.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_18_2024_14_52_47.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_35_50.csv'  # noise_up = 0.05
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_39_57.csv'  # noise_up = 0.5
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_14_50_47.csv'  # noise_up = 1
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_17_07_52.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_10_2025_17_12_50.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_16_41_15.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_16_58_58.csv' # unifrom noise
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_04_25.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_10_42.csv' # uniform nosi not centered
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_13_40.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_12_16_2024_16_59_04.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_14_2025_18_20_50.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_04_25.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_17_2025_14_54_47.csv'
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_20_2025_10_46_12.csv'
df = pd.read_csv(file_path)

# Load data
vy = df['velocity y'].values
x = df['vicon x'].values
y = df['vicon y'].values
vx = df['vel encoder'].values
t = df['elapsed time sensors'].values
yaw = df['vicon yaw'].values
w = df['W (IMU)'].values
tau = df['throttle'].values
steering_input = df['steering'].values

# Define dynamic model parameter
l = 0.175
lr = 0.54 * l
lf = l - lr
m = 1.67
m_front_wheel = 0.847
m_rear_wheel = 0.733
Cf = m_front_wheel / m
Cr = m_rear_wheel / m
Jz = 0.006513

# Parameters needed for SME evaluation
n_state = 3                       # number of states
I_n = np.eye(n_state, dtype=int)  # dim (3,3)
H = np.vstack([I_n, -I_n])        # dim (6,3)

d_up = 0.003                 # noise upper bound
d_low = - d_up                    # noise lower bound
h_d = np.concatenate([
    np.full((n_state, 1), d_up),
    np.full((n_state, 1), -d_low)
])

# Number of parameter the SME has to estimate
n_param = 2

# Offline direction evaluation
direction =  generate_orthogonal_directions(n_param)
# print(direction)

# Initialiaze the plot 
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0.7, 1.4)
ax.set_ylim(0.7, 1.4)
ax.set_title('')
ax.set_xlabel('μ_1')
ax.set_ylabel('μ_2')

mu_1_values = []
mu_2_values = []



starting_instant = 50
ending_instant = len(df)
for index in range(starting_instant, ending_instant):
    
    # # # # ============================================================================================== # # #
    # # # #                                  UNFALSIFIED PARAMETER SET                                     # # #
    # # # #                                          𝚫_k                                                   # # #
    # # # # ============================================================================================== # # #
    
    F_0_minus2, G_0_minus2 = continuous_matrices_2(index - 2, steering_input, vx, vy, w, tau) # Maps F and G in continuos time
    delta_minus_2 = steer_angle(steering_input[index - 2])                                    # Steering angle
    x_cont_minus_2 = np.array([[vx[index -2]], [vy[index -2]], [w[index -2]]])                # state vector in continuous time
    u_cont_minus_2 = np.array([[tau[index -2]], [delta_minus_2]])                             # imput vector in continuous time
    
    # Istant i minus 1
    F_0_minus1, G_0_minus1 = continuous_matrices_2(index - 1, steering_input, vx, vy, w, tau)
    delta_minus_1 = steer_angle(steering_input[index - 1])
    x_cont_minus_1 = np.array([[vx[index -1]], [vy[index -1]], [w[index -1]]]) 
    u_cont_minus_1 = np.array([[tau[index -1]], [delta_minus_2]])

    # Changing names
    autonomous_func_minus_2 = F_0_minus2
    input_func_minus_2 = G_0_minus2
    autonomous_func_minus_1 = F_0_minus1
    input_func_minus_1 = G_0_minus1

    # Maps in discrete time
    f_dicr_minus_2, g_discr_minus_2, state_discr_minus_2 = compute_discrete_function_terms_single_step_euler_2(x_cont_minus_2, u_cont_minus_2, autonomous_func_minus_2, input_func_minus_2)
    f_dicr_minus_1, g_discr_minus_1, state_discr_minus_1 = compute_discrete_function_terms_single_step_euler_2(x_cont_minus_1, u_cont_minus_1, autonomous_func_minus_1, input_func_minus_1)

    ## The inequality to solve is: - H * G * mu <= h_d - H * x_discr + H * F
    ## Grouping the terms: A = - H * G and b = h_d - H * x_discr + H * F
    ## Finally:  A * mu <= blen(df)

    if index == starting_instant:
        A_i_minus2 = - H @ g_discr_minus_2
        b_i_minus2 = h_d - H @ state_discr_minus_2 + H @ f_dicr_minus_2
    else:
        pass

    A_i_minus1 = - H @ g_discr_minus_1
    b_i_minus1 = h_d - H @ state_discr_minus_1 + H @ f_dicr_minus_1

    # print(A_i_minus1.shape)
    A = np.concatenate([A_i_minus1, A_i_minus2])
    b = np.concatenate([b_i_minus1, b_i_minus2])

    # print(f"A_shape = {A.shape}, b_shape = {b.shape}")

    vertex = compute_vertices(A, b)
    
    # Cicles for checking vetex vector dimension 

    if len(vertex) == 0:
        # print(f"Warning: No vertex found at iteration {index}. Skipping this iteration.")
        continue  # Skip this iteration to avoid errors
    vertex = np.array(vertex)

    if vertex.shape[1] == 0:
        # print(f"⚠️  Iteration {index}: Vertex is empty (shape={vertex.shape}), skipping iteration.")
        continue

    if vertex.ndim > 2:
        vertex = vertex.squeeze(-1)
    # print(f"Debug: vertex.shape = {vertex.shape}")

    Hp, hp = underapproximate_convex_polytope(vertex, direction)

    # Controllo per evitare errori di indice e divisione per zero
    mu_1_up = mu_1_low = mu_2_up = mu_2_low = None  

    if Hp.shape[0] > 0 and hp.shape[0] > 0:
        mu_1_up = hp[0] / Hp[0, 0] if Hp[0, 0] != 0 else None  # Evita inf

    if Hp.shape[0] > 1 and hp.shape[0] > 1:
        mu_1_low = hp[1] / Hp[1, 0] if Hp[1, 0] != 0 else None  # Evita inf

    if Hp.shape[0] > 2 and hp.shape[0] > 2:
        mu_2_up = hp[2] / Hp[2, 1] if Hp[2, 1] != 0 else None  # Evita inf

    if Hp.shape[0] > 3 and hp.shape[0] > 3:
        mu_2_low = hp[3] / Hp[3, 1] if Hp[3, 1] != 0 else None  # Evita None

    print(f"Iteration {index} --> mu_1 = [{mu_1_low}, {mu_1_up}], mu_2 = [{mu_2_low}, {mu_2_up}]")
    
    if None not in [mu_1_low, mu_1_up, mu_2_low, mu_2_up]:
        ax.clear()
        ax.set_xlabel(r'$\mu_1$')
        ax.set_ylabel(r'$\mu_2$')
        ax.set_xlim(0.7, 1.4)
        ax.set_ylim(0.7, 1.4)
        ax.set_title(
                    rf"$\mu_1 \in [{mu_1_low:.3f}, {mu_1_up:.3f}]$" "\n" 
                    rf"$\mu_2 \in [{mu_2_low:.3f}, {mu_2_up:.3f}]$"
                    )

        ax.scatter(vertex[:, 0], vertex[:, 1], color='blue', label='Vertices')

        rect_x = [mu_1_low, mu_1_up, mu_1_up, mu_1_low, mu_1_low]
        rect_y = [mu_2_low, mu_2_low, mu_2_up, mu_2_up, mu_2_low]
        ax.plot(rect_x, rect_y, 'g-')
        ax.fill_betweenx([mu_2_low, mu_2_up], mu_1_low, mu_1_up, color='green', alpha=0.5, label=r'$\theta_k$')

        ax.legend()
        plt.pause(0.005)  


    A_i_minus2 = Hp
    b_i_minus2 = hp
    b_i_minus2 = np.atleast_2d(b_i_minus2).T

plt.ioff()
plt.show()

