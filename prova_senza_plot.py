import pandas as pd
import numpy as np
from scipy.optimize import linprog
from funct_fin import steer_angle, rolling_friction, motor_force, F_friction_due_to_steering, slip_angles, lateral_tire_forces, friction
from discretization_function_2 import compute_discrete_function_terms_single_step_euler_2
from continuos_matrix_2 import continuous_matrices_2
from sympy import symbols, Matrix,And, solve, reduce_inequalities, simplify
from Fucntion_2_D import compute_vertices
from direction import generate_orthogonal_directions
from underapproximate_convex_politope import underapproximate_convex_polytope
# from continuos_matrix_copy import continuous_matrices


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
file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_14_2025_18_20_50.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_13_2025_17_04_25.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_17_2025_14_54_47.csv'
# file_path = '/home/domenico/DART_QDP/src/racecar_pkg/DATA/car_1_Datarecording_01_20_2025_10_46_12.csv'
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

d_up = 0.001                 # noise upper bound
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

starting_instant = 400
ending_instant = 460


F_0_minus2, G_0_minus2 = continuous_matrices_2(starting_instant - 2, steering_input, vx, vy, w, tau) # Maps F and G in continuos time
delta_minus_2 = steer_angle(steering_input[starting_instant - 2])                                    # Steering angle
x_cont_minus_2 = np.array([[vx[starting_instant -2]], [vy[starting_instant -2]], [w[starting_instant -2]]])                # state vector in continuous time
u_cont_minus_2 = np.array([[tau[starting_instant -2]], [delta_minus_2]])                             # imput vector in continuous time
autonomous_func_minus_2 = F_0_minus2
input_func_minus_2 = G_0_minus2
f_dicr_minus_2, g_discr_minus_2, state_discr_minus_2 = compute_discrete_function_terms_single_step_euler_2(x_cont_minus_2, u_cont_minus_2, autonomous_func_minus_2, input_func_minus_2)

A_i_minus2 = - H @ g_discr_minus_2
b_i_minus2 = h_d - H @ state_discr_minus_2 + H @ f_dicr_minus_2

for index in range(starting_instant, ending_instant):
    
    # # # # ============================================================================================== # # #
    # # # #                                  UNFALSIFIED PARAMETER SET                                     # # #
    # # # #                                          𝚫_k                                                   # # #
    # # # # ============================================================================================== # # #
    
               
    print("Starting cycle")
    # Istant i minus 1
    F_0_minus1, G_0_minus1 = continuous_matrices_2(index - 1, steering_input, vx, vy, w, tau)
    delta_minus_1 = steer_angle(steering_input[index - 1])
    x_cont_minus_1 = np.array([[vx[index -1]], [vy[index -1]], [w[index -1]]]) 
    u_cont_minus_1 = np.array([[tau[index -1]], [delta_minus_2]])

    # Changing names
    
    autonomous_func_minus_1 = F_0_minus1
    input_func_minus_1 = G_0_minus1

    # Maps in discrete time
    f_dicr_minus_1, g_discr_minus_1, state_discr_minus_1 = compute_discrete_function_terms_single_step_euler_2(x_cont_minus_1, u_cont_minus_1, autonomous_func_minus_1, input_func_minus_1)

    ## The inequality to solve is: - H * G * mu <= h_d - H * x_discr + H * F
    ## Grouping the terms: A = - H * G and b = h_d - H * x_discr + H * F
    ## Finally:  A * mu <= blen(df)

    A_i_minus1 = - H @ g_discr_minus_1
    b_i_minus1 = h_d - H @ state_discr_minus_1 + H @ f_dicr_minus_1

    # print(A_i_minus1.shape)
    A = np.concatenate([A_i_minus1, A_i_minus2])
    b = np.concatenate([b_i_minus1, b_i_minus2])

    # print(f"A_shape = {A.shape}, b_shape = {b.shape}")

    vertex = compute_vertices(A, b)
    vertex_i = compute_vertices(A_i_minus1, b_i_minus1)
    vertex_i_minus1 = compute_vertices(A_i_minus2, b_i_minus2)
    # Cicles for checking vetex vector dimension 

    ## TOTAL 
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

    ## ONLY INDEX MINUS 1
    if len(vertex_i) == 0:
        # print(f"Warning: No vertex found at iteration {index}. Skipping this iteration.")
        continue  # Skip this iteration to avoid errors
    vertex_i = np.array(vertex_i)

    if vertex_i.shape[1] == 0:
        # print(f"⚠️  Iteration {index}: Vertex is empty (shape={vertex.shape}), skipping iteration.")
        continue

    if vertex_i.ndim > 2:
        vertex_i = vertex_i.squeeze(-1)
    # print(f"Debug: vertex.shape = {vertex.shape}")

    ## ONLY INDEX MINUS 2
    if len(vertex_i_minus1) == 0:
        # print(f"Warning: No vertex found at iteration {index}. Skipping this iteration.")
        continue  # Skip this iteration to avoid errors
    vertex_i_minus1 = np.array(vertex_i_minus1)

    if vertex_i_minus1.shape[1] == 0:
        # print(f"⚠️  Iteration {index}: Vertex is empty (shape={vertex.shape}), skipping iteration.")
        continue

    if vertex_i_minus1.ndim > 2:
        vertex_i_minus1 = vertex_i_minus1.squeeze(-1)
    # print(f"Debug: vertex.shape = {vertex.shape}")
    
    Hp, hp = underapproximate_convex_polytope(vertex, direction)

    print(f"Iteration {index}: Hp prima della funzione:\n{Hp}")
    print(f"Iteration {index}: hp prima della funzione:\n{hp}")
    Hp_act, hp_act = underapproximate_convex_polytope(vertex_i, direction)

    Hp_act_minus1, hp_act_minus1 = underapproximate_convex_polytope(vertex_i_minus1, direction)

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

    mu_i_1_up = mu_i_1_low = mu_i_2_up = mu_i_2_low = None  

    if Hp_act.shape[0] > 0 and hp_act.shape[0] > 0:
        mu_i_1_up = hp_act[0] / Hp_act[0, 0] if Hp_act[0, 0] != 0 else None  # Evita inf

    if Hp_act.shape[0] > 1 and hp_act.shape[0] > 1:
        mu_i_1_low = hp_act[1] / Hp_act[1, 0] if Hp_act[1, 0] != 0 else None  # Evita inf

    if Hp_act.shape[0] > 2 and hp_act.shape[0] > 2:
        mu_i_2_up = hp_act[2] / Hp_act[2, 1] if Hp_act[2, 1] != 0 else None  # Evita inf

    if Hp_act.shape[0] > 3 and hp_act.shape[0] > 3:
        mu_i_2_low = hp_act[3] / Hp_act[3, 1] if Hp_act[3, 1] != 0 else None  # Evita None

    mu_i_minus1_1_up = mu_i_minus1_1_low = mu_i_minus1_2_up = mu_i_minus1_2_low = None  
    
    print("Printing results\n")
    print(f"Iteration {index} --> mu_1 = [{mu_1_low}, {mu_1_up}], mu_2 = [{mu_2_low}, {mu_2_up}]")
    print(f"mu_i_1 = [{mu_i_1_low}, {mu_i_1_up}], mu_i_2 = [{mu_i_2_low}, {mu_i_2_up}]")

    if index != starting_instant:
        if Hp_act_minus1.shape[0] > 0 and hp_act_minus1.shape[0] > 0:
            mu_i_minus1_1_up = hp_act_minus1[0] / Hp_act_minus1[0, 0] if Hp_act_minus1[0, 0] != 0 else None  # Evita inf

        if Hp_act_minus1.shape[0] > 1 and hp_act_minus1.shape[0] > 1:
            mu_i_minus1_1_low = hp_act_minus1[1] / Hp_act_minus1[1, 0] if Hp_act_minus1[1, 0] != 0 else None  # Evita inf

        if Hp_act_minus1.shape[0] > 2 and hp_act_minus1.shape[0] > 2:
            mu_i_minus1_2_up = hp_act_minus1[2] / Hp_act_minus1[2, 1] if Hp_act_minus1[2, 1] != 0 else None  # Evita inf

        if Hp_act_minus1.shape[0] > 3 and hp_act_minus1.shape[0] > 3:
            mu_i_minus1_2_low = hp_act_minus1[3] / Hp_act_minus1[3, 1] if Hp_act_minus1[3, 1] != 0 else None  # Evita None

        print(f"mu_i_minus1_1 = [{mu_i_minus1_1_low}, {mu_i_minus1_1_up}], mu_i_minus1_2 = [{mu_i_minus1_2_low}, {mu_i_minus1_2_up}]\n")

    else:
        continue

    

    if index == 453 or index == 454:
        print(f"\n🚀 Iteration {index}: Debugging Hp, hp, and vertices")

        print(f"Hp shape: {Hp.shape}, hp shape: {hp.shape}")
        print(f"Hp:\n{Hp}")
        print(f"hp:\n{hp}")

        print(f"vertex.shape: {vertex.shape}")
        print(f"vertex:\n{vertex}")

        print(f"vertex_i.shape: {vertex_i.shape}")
        print(f"vertex_i:\n{vertex_i}")

        print(f"mu_1 = [{mu_1_low}, {mu_1_up}], mu_2 = [{mu_2_low}, {mu_2_up}]")
        print(f"mu_i_minus1_1 = [{mu_i_minus1_1_low}, {mu_i_minus1_1_up}], mu_i_minus1_2 = [{mu_i_minus1_2_low}, {mu_i_minus1_2_up}]")


        print(f"A_i_minus1.shape: {A_i_minus1.shape}, b_i_minus1.shape: {b_i_minus1.shape}")
        print(f"A_i_minus1:\n{A_i_minus1}")
        print(f"b_i_minus1:\n{b_i_minus1}")

        print(f"A_i_minus2.shape: {A_i_minus2.shape}, b_i_minus.shape: {b_i_minus2.shape}")
        print(f"A_i_minus2:\n{A_i_minus2}")
        print(f"b_i_minus2:\n{b_i_minus2}")

        print(f"A.shape: {A.shape}, b.shape: {b.shape}")
        print(f"A:\n{A}")
        print(f"b:\n{b}")

    condition_number = np.linalg.cond(A)
    print(f"Iteration {index}: Condition number of A = {condition_number}")
    print(f"Iteration {index}: Directions used: {direction}")
    print(f"Iteration {index}: Vertices = {compute_vertices(Hp, hp)}")

    print(f"Iteration {index}: Verifica intersezione manuale")
    print(f"Expected intersection: X[{max(mu_i_1_low, mu_i_minus1_1_low)}, {min(mu_i_1_up, mu_i_minus1_1_up)}]")
    print(f"Computed theta_i: X[{mu_1_low}, {mu_1_up}]")
    print(f"Expected intersection: Y[{max(mu_i_2_low, mu_i_minus1_2_low)}, {min(mu_i_2_up, mu_i_minus1_2_up)}]")
    print(f"Computed theta_i: Y[{mu_2_low}, {mu_2_up}]")
    vertex_debug = compute_vertices(A, b)
    print(f"Iteration {index}: vertex_debug = {vertex_debug}")


    A_i_minus2 = Hp
    b_i_minus2 = hp
    b_i_minus2 = np.atleast_2d(b_i_minus2).T
    print(b_i_minus2.shape)



