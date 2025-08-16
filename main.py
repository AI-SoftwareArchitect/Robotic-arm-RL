import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

# -----------------------------
# 1️⃣ DQN Modeli
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

# -----------------------------
# 2️⃣ Çevre Kurulumu
# -----------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0,0,-9.8)
p.setTimeStep(1./240.)

plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("kuka_iiwa/model.urdf", [0,0,0])
cube = p.loadURDF("cube_small.urdf", [0.5, 0, 0.05])

n_joints = 7
deltas = [-0.1, 0.1]  # daha küçük action space
action_mapping = [(j, d) for j in range(n_joints) for d in deltas]
n_actions = len(action_mapping)

# -----------------------------
# 3️⃣ DQN Ayarları
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = n_joints*2 + 4  # pos+vel+ee_xy+cube_xy
policy_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

replay_buffer = []
max_buffer = 1000
batch_size = 32
gamma = 0.99

# Epsilon decay parametreleri
epsilon = 0.9
epsilon_min = 0.05
decay_rate = 0.995

# -----------------------------
# 4️⃣ Yardımcı Fonksiyonlar
# -----------------------------
def get_state():
    joint_states = p.getJointStates(robot, range(n_joints))
    pos = [s[0] for s in joint_states]
    vel = [s[1] for s in joint_states]
    ee_pos = p.getLinkState(robot, n_joints-1)[0]
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    return np.array(pos + vel + list(ee_pos[:2]) + list(cube_pos[:2]), dtype=np.float32)

def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return random.randint(0, n_actions-1)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return torch.argmax(q_values).item()

def step(action_idx):
    joint, delta = action_mapping[action_idx]
    # Velocity-based control
    p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL,
                            targetVelocity=delta*5,
                            force=200)
    for _ in range(10):
        p.stepSimulation()
        time.sleep(1./240.)

    state = get_state()
    cube_pos = np.array(p.getBasePositionAndOrientation(cube)[0])
    ee_pos = np.array(p.getLinkState(robot, n_joints-1)[0])
    dist = np.linalg.norm(cube_pos[:2] - ee_pos[:2])

    # Sofistike reward
    joint_vels = np.array([s[1] for s in p.getJointStates(robot, range(n_joints))])
    reward = -dist / 2.0
    reward += 0.01 * np.sum(np.abs(joint_vels))
    reward += 1.0 if dist < 0.05 else 0.0
    done = dist < 0.05
    return state, reward, done

# -----------------------------
# 5️⃣ Eğitim Döngüsü
# -----------------------------
episodes = 200
for ep in range(episodes):
    p.resetBasePositionAndOrientation(robot, [0,0,0],[0,0,0,1])
    rand_x = random.uniform(0.3, 0.7)
    rand_y = random.uniform(-0.3, 0.3)
    p.resetBasePositionAndOrientation(cube, [rand_x, rand_y, 0.05], [0,0,0,1])
    
    state = get_state()
    total_reward = 0
    for t in range(200):
        action = select_action(state)
        next_state, reward, done = step(action)
        total_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > max_buffer:
            replay_buffer.pop(0)

        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states_b, actions_b, rewards_b, next_b, dones_b = zip(*batch)
            states_b = torch.FloatTensor(states_b).to(device)
            actions_b = torch.LongTensor(actions_b).unsqueeze(1).to(device)
            rewards_b = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
            next_b = torch.FloatTensor(next_b).to(device)
            dones_b = torch.FloatTensor(dones_b).unsqueeze(1).to(device)

            q_values = policy_net(states_b).gather(1, actions_b)
            next_q = target_net(next_b).max(1)[0].detach().unsqueeze(1)
            target = rewards_b + gamma * next_q * (1 - dones_b)
            loss = criterion(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Target network sık güncelleme
        if t % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        state = next_state
        if done:
            break

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * decay_rate)

    print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
