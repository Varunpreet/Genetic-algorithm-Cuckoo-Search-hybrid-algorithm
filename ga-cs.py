import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

rounds = 600
generations = 20
size = 30
ch_perc = 0.04
nodes = 200
bs = (-80, -80)
crossover_prob = 0.6
mut_prob = 0.1
beta = 1.5
BS_COORDINATE = np.array([-80, -80])
dimension = 100
KE = 1000
init_energy = 0.05
E_ELEC = 50e-9
E_DA = 5e-9
P_FS = 10e-12
P_MP = 0.0013e-12
D0 = (P_FS / P_MP) ** 0.5

sensor_nodes = np.random.randint(0, 100, (nodes, 2))
sensor_nodes_energy = [{'position': np.random.randint(0, 100, (2,)), 'energy': init_energy} for _ in range(nodes)]

def cal_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    distance = np.linalg.norm(p1 - p2)
    return distance

def transmit_energy(d, ke=KE):
    if d < D0:
        return ke * (E_ELEC + E_DA + P_FS * d ** 2)
    else:
        return ke * (E_ELEC + E_DA + P_MP * d ** 4)

def adjust_transmit_energy(node_index, d, ke=KE):
    energy_consumed = transmit_energy(d, ke)
    sensor_nodes_energy[node_index]['energy'] -= energy_consumed
    if sensor_nodes_energy[node_index]['energy'] < 0:
        sensor_nodes_energy[node_index]['energy'] = 0

def receive_energy(ke=KE):
    return ke * E_ELEC

def adjust_receive_energy(node_index, ke=KE):
    energy_consumed = receive_energy(ke)
    sensor_nodes_energy[node_index]['energy'] -= energy_consumed
    if sensor_nodes_energy[node_index]['energy'] < 0:
        sensor_nodes_energy[node_index]['energy'] = 0

def count_alive_sensor_nodes():
    return sum(node['energy'] > 0 for node in sensor_nodes_energy)

def levy_flight(beta=beta, scale=1.0, size=1, current_position=None):
    sigma_u = (sp.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (sp.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    sigma_v = 1
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, sigma_v, size)
    step = u / (np.abs(v) ** (1 / beta))
    direction = np.random.uniform(-1, 1, current_position.shape)
    norm_direction = direction / np.linalg.norm(direction)
    new_position = current_position + scale * step * norm_direction
    return np.clip(new_position, 0, dimension - 1)

def closest_node(position, nodes, exclude=[]):
    closest_node = None
    min_distance = np.inf
    for i, node_position in enumerate(nodes):
        if i not in exclude:
            distance = cal_distance(position, node_position)
            if distance < min_distance:
                min_distance = distance
                closest_node = i
    return closest_node, min_distance

def optimize_path(chs, sensor_nodes, clusters):
    optimized_paths = {}
    for ch_index in chs:
        if isinstance(ch_index, int) and ch_index < len(sensor_nodes):
            current_position = sensor_nodes[ch_index]['position']
            path = [ch_index]
            while True:
                new_position = levy_flight(scale=20, size=2, current_position=current_position)
                next_hop, _ = closest_node(new_position, [sensor_nodes[node]['position'] for node in clusters[ch_index]] + [BS_COORDINATE])
                if next_hop is None or next_hop in path:
                    break
                path.append(next_hop)
                current_position = sensor_nodes[next_hop]['position']
                if np.array_equal(current_position, BS_COORDINATE):
                    break
            optimized_paths[ch_index] = path
    return optimized_paths

def fitness_function(ch_selection, alive_sensor_nodes):
    A, B, C = 0, 0, 0
    valid_ch_selection = [ch for ch in ch_selection if ch < len(alive_sensor_nodes)]
    for ch_idx in valid_ch_selection:
        ch = alive_sensor_nodes[ch_idx]
        dist_to_bs = cal_distance(ch, BS_COORDINATE)
        B += transmit_energy(dist_to_bs)
        for node_idx, node in enumerate(alive_sensor_nodes):
            if node_idx not in valid_ch_selection:
                dist_to_ch = cal_distance(node, ch)
                A += dist_to_ch
                B += transmit_energy(dist_to_ch) + receive_energy()
                if sensor_nodes_energy[node_idx]['energy'] - transmit_energy(dist_to_ch) > 0:
                    C += 1
    f = (0.33 * (1 / A if A > 0 else 0) + 0.33 * (1 / B if B > 0 else 0) + 0.34 * C)
    return f

def initialize_population(alive_sensor_nodes, sensor_nodes_energy):
    MIN_CHS = max(1, int(len(alive_sensor_nodes) * ch_perc))
    energy_weight = 0.5
    distance_weight = 0.5
    scores = []
    for i, node in enumerate(alive_sensor_nodes):
        distance_to_bs = cal_distance(node, bs)
        energy_score = sensor_nodes_energy[i]['energy'] / init_energy
        distance_score = (dimension - distance_to_bs) / dimension
        score = energy_weight * energy_score + distance_weight * distance_score
        scores.append(score)
    sorted_indices = np.argsort(scores)
    ch_indices = sorted_indices[-MIN_CHS:]
    return [np.array(ch_indices)]

def selection(population, sensor_nodes):
    fitness_values = np.array([fitness_function(individual, sensor_nodes) for individual in population])
    total_fitness = fitness_values.sum()
    if total_fitness == 0:
        prob = np.ones(len(population)) / len(population)
    else:
        prob = fitness_values / total_fitness
    selected_indices = np.random.choice(len(population), size=len(population), replace=True, p=prob)
    return [population[i] for i in selected_indices]

def crossover(parents, num_alive_nodes):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents) and np.random.rand() < point:
            point = np.random.randint(1, num_alive_nodes - 1)
            parent1, parent2 = parents[i], parents[i+1]
            offspring1 = np.concatenate([parent1[:point], parent2[point:]])
            offspring2 = np.concatenate([parent2[:point], parent1[point:]])
            offspring.extend([offspring1, offspring2])
        else:
            offspring.extend([parents[i]])
    return offspring

def mutation(offspring, num_alive_nodes):
    for individual in offspring:
        if len(individual) < 2:
            continue
        if np.random.rand() < mut_prob:
            high_value = max(2, len(individual))
            size = np.random.randint(1, high_value // 2 + 1)
            points = np.random.choice(range(len(individual)), size=size, replace=False)
            for mp in points:
                new_value_options = [idx for idx in range(num_alive_nodes) if idx not in individual]
                if new_value_options:
                    individual[mp] = np.random.choice(new_value_options)
    return offspring

def replacement(population, offspring, alive_sensor_nodes):
    alive_indices = [i for i, node in enumerate(sensor_nodes_energy) if node['energy'] > 0]
    alive_sensor_nodes = [sensor_nodes[i] for i in alive_indices]
    combined = population + offspring
    scores = [fitness_function(individual, alive_sensor_nodes) for individual in combined]
    sorted_by_fitness = [x for _, x in sorted(zip(scores, combined), key=lambda pair: pair[0], reverse=True)]
    return sorted_by_fitness[:len(population)]

def form_clusters(ch_selection, alive_sensor_nodes):
    clusters = {ch: [] for ch in ch_selection}
    for node_idx, node in enumerate(alive_sensor_nodes):
        if node_idx not in ch_selection:
            closest_ch = min(ch_selection, key=lambda ch: cal_distance(alive_sensor_nodes[ch], node))
            clusters[closest_ch].append(node_idx)
    return clusters

def collect_data(clusters, alive_sensor_nodes):
    for ch, members in clusters.items():
        for member in members:
            dist_to_ch = cal_distance(alive_sensor_nodes[member], alive_sensor_nodes[ch])
            adjust_transmit_energy(member, dist_to_ch, KE)
            adjust_receive_energy(ch, KE)

def data_transmission(ch, optimized_path, sensor_nodes_energy, KE):
    for member_idx in optimized_path:
        if member_idx != ch:
            dist_to_ch = cal_distance(sensor_nodes[member_idx], sensor_nodes[ch])
            adjust_transmit_energy(member_idx, dist_to_ch, KE)
            adjust_receive_energy(ch, KE)
    for i in range(len(optimized_path)-1):
        current_node = optimized_path[i]
        next_node = optimized_path[i+1]
        dist_to_next = cal_distance(sensor_nodes[current_node], sensor_nodes[next_node])
        adjust_transmit_energy(current_node, dist_to_next, KE)
        adjust_receive_energy(next_node, KE)
    last_node = optimized_path[-1]
    dist_to_bs = cal_distance(sensor_nodes[last_node], BS_COORDINATE)
    adjust_transmit_energy(last_node, dist_to_bs, KE)

def remove_dead_nodes():
    global sensor_nodes, sensor_nodes_energy
    alive_indices = [i for i, node in enumerate(sensor_nodes_energy) if node['energy'] > 0]
    sensor_nodes = [sensor_nodes[i] for i in alive_indices]
    sensor_nodes_energy = [sensor_nodes_energy[i] for i in alive_indices]

def plot_alive_nodes(alive_nodes_per_round):
    plt.figure(figsize=(10, 6))
    rounds = range(1, len(alive_nodes_per_round) + 1)
    plt.plot(rounds, alive_nodes_per_round, color='red')
    plt.title('Network Lifetime')
    plt.xlabel('Number of Rounds')
    plt.ylabel('Number of Alive Sensor Nodes')
    plt.grid(True)
    plt.show()

def gacs_algorithm():
    alive_nodes_per_round = [] 
    global sensor_nodes, sensor_nodes_energy
    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")
        alive_indices = [i for i, node in enumerate(sensor_nodes_energy) if node['energy'] > 0]
        if len(alive_indices) == 0:
            print("All sensor nodes are dead.")
            break
        alive_sensor_nodes = [sensor_nodes[i] for i in alive_indices]
        population = initialize_population(alive_sensor_nodes, sensor_nodes_energy)
        best_fitness = -np.inf
        best_selection = None

        for generation in range(generations):
            selected_parents = selection(population, alive_sensor_nodes)
            offspring_crossover = crossover(selected_parents, len(alive_sensor_nodes))
            offspring_mutation = mutation(offspring_crossover, len(alive_sensor_nodes))
            population = replacement(population, offspring_mutation, alive_sensor_nodes)
            for individual in population:
                fitness = fitness_function(individual, alive_sensor_nodes)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_selection = individual
        if best_selection is None:
            print("No viable cluster heads left to form clusters.")
            continue
        clusters = form_clusters(best_selection, alive_sensor_nodes)
        collect_data(clusters, alive_sensor_nodes)
        optimized_paths = optimize_path(best_selection, sensor_nodes, clusters)
        for ch, path in optimized_paths.items():
            data_transmission(ch, path, sensor_nodes_energy, KE)
        alive_nodes = count_alive_sensor_nodes()
        alive_nodes_per_round.append(alive_nodes)
        print(f"End of Round {round + 1}: {alive_nodes} alive sensor nodes")
        if alive_nodes == 0:
            break
    print(f"Simulation ended after {round + 1} rounds with {alive_nodes} alive sensor nodes.")
    plot_alive_nodes(alive_nodes_per_round)

alive_nodes = count_alive_sensor_nodes()
print(f"{alive_nodes} alive sensor nodes in the beginning")
gacs_algorithm()