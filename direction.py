import numpy as np
from itertools import combinations, product

def is_zero(vector, tol=1e-10):
    return np.all(np.abs(vector) < tol)

def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def add_combinations(directions, new_dir_set, current_sum, start, depth, max_depth):
    if depth == max_depth:
        if not is_zero(current_sum):
            new_dir_set.add(tuple(normalize(current_sum)))
        return
    
    for i in range(start, len(directions)):
        add_combinations(directions, new_dir_set, current_sum + directions[i], i + 1, depth + 1, max_depth)

def generate_predefined_directions(p, phi):
    directions = [np.eye(p)[i] for i in range(p)] + [-np.eye(p)[i] for i in range(p)]
    
    for _ in range(phi):
        new_dir_set = set()
        for k in range(1, p + 1):  # Combinations up to p vectors
            add_combinations(directions, new_dir_set, np.zeros(p), 0, 0, k)
        directions = [np.array(d) for d in new_dir_set]
    
    return directions

def predefined_normalized_directions(dimensions):
    directions = []
    total_combinations = 3 ** dimensions
    
    for num in range(total_combinations):
        direction = np.zeros(dimensions)
        temp_num = num
        is_zero_vector = True
        
        for dim in range(dimensions):
            value = temp_num % 3
            temp_num //= 3
            direction[dim] = value - 1  # 0 -> -1, 1 -> 0, 2 -> 1
            if value != 1:
                is_zero_vector = False
        
        if not is_zero_vector and np.linalg.norm(direction) > 0:
            directions.append(normalize(direction))
    
    return directions

def generate_orthogonal_directions(dimensions):
    directions = []
    for i in range(dimensions):
        direction = np.zeros(dimensions)
        direction[i] = 1
        directions.append(direction.copy())
        direction[i] = -1
        directions.append(direction.copy())
    
    return directions

# Test delle funzioni
# if __name__ == "__main__":
#     p, phi = 2, 1  # Esempio per test
#     # print("Predefined Directions:", generate_predefined_directions(p, phi))
#     print("Octagon-like Directions:", predefined_normalized_directions(p))
#     print("Orthogonal Directions:", generate_orthogonal_directions(p))
