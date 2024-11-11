import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Parameters
num_nodes = 10
num_contents = 50
cache_capacity = 5
episodes = 10  # Reduced to 10 for quicker testing
alpha = 0.001
gamma = 0.9
epsilon = 0.2
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
memory_size = 10000

# Randomly generate content popularity and priority
content_popularity = np.random.rand(num_contents)
content_priority = np.random.rand(num_contents)
node_demand = np.random.rand(num_nodes, num_contents)

# Weighted content demand based on popularity and priority
content_demand = content_popularity * content_priority

# Cost matrix for delivery cost between each pair of nodes
delivery_cost_matrix = np.random.randint(1, 10, size=(num_nodes, num_nodes))

# DQN Neural Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x

# Initialize DQN
input_dim = num_contents
output_dim = num_contents
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
loss_fn = nn.MSELoss()
memory = deque(maxlen=memory_size)

# Lists to store metrics for plotting
episode_costs = []
epsilon_values = []

# Function to calculate weighted delivery cost for given cache placements
def calculate_delivery_cost(cache_placement):
    total_cost = 0
    for node in range(num_nodes):
        for content in range(num_contents):
            # Count the presence of content in each node's cache
            if content in cache_placement[node]:
                min_cost = float('inf')
                for user_node in range(num_nodes):
                    cost = delivery_cost_matrix[node][user_node] * node_demand[user_node, content]
                    min_cost = min(min_cost, cost)
                total_cost += min_cost * content_demand[content]
    return total_cost

# Replay Memory Sampling
def sample_memory():
    return random.sample(memory, min(len(memory), batch_size))

# Training loop
for episode in range(episodes):
    print(f"\nStarting Episode {episode + 1}")
    cache_placement = [[] for _ in range(num_nodes)]  # Ensuring each node has a list as its cache
    state = content_demand
    print("Initial state (content demand):", state)

    # Each node decides content to cache within its capacity constraints
    for node in range(num_nodes):
        print(f"\nNode {node + 1} deciding on content to cache")

        # Action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_contents - 1)
            print(f"Exploration: Randomly selected content {action}")
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).argmax().item()
                print(f"Exploitation: Selected best known content {action} from Q-network")

        # Add content to cache if capacity allows
        if len(cache_placement[node]) < cache_capacity:
            cache_placement[node].append(action)
            print(f"Content {action} cached at node {node + 1}")

        # Calculate reward as negative of delivery cost
        reward = -calculate_delivery_cost(cache_placement)
        print(f"Calculated reward: {reward}")

        # Next state: dynamic adjustment of content demand
        next_state = content_demand * (0.9 + 0.2 * np.random.rand(num_contents))
        print("Next state (updated content demand):", next_state)

        # Store experience in replay memory
        memory.append((state, action, reward, next_state))
        state = next_state

        # Experience Replay and DQN Update
        if len(memory) >= batch_size:
            print("Performing experience replay")
            batch = sample_memory()
            batch_state, batch_action, batch_reward, batch_next_state = zip(*batch)

            batch_state = torch.FloatTensor(batch_state)
            batch_action = torch.LongTensor(batch_action)
            batch_reward = torch.FloatTensor(batch_reward)
            batch_next_state = torch.FloatTensor(batch_next_state)

            current_q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
            max_next_q_values = target_net(batch_next_state).max(1)[0]
            target_q_values = batch_reward + gamma * max_next_q_values

            optimizer.zero_grad()
            loss = loss_fn(current_q_values, target_q_values)
            loss.backward()
            optimizer.step()
            print("Q-network updated")

    # Record episode metrics for plotting
    episode_costs.append(-reward)
    epsilon_values.append(epsilon)

    # Update target network periodically
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print("Target network updated")

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Final Evaluation
# Create an optimal cache placement as a list of lists, ensuring each node caches up to `cache_capacity` items
optimal_cache_placement = []
for node in range(num_nodes):
    # Select top `cache_capacity` contents based on Q-values
    node_cache = torch.topk(policy_net(torch.FloatTensor(content_demand)), cache_capacity).indices.numpy().tolist()
    optimal_cache_placement.append(node_cache)

# Calculate the optimized delivery cost for this final cache placement
optimized_cost = calculate_delivery_cost(optimal_cache_placement)
print("\nOptimal Cache Placement:", optimal_cache_placement)
print("Optimized Delivery Cost:", optimized_cost)
