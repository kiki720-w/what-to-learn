import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random


# 1. 模型定义 (全卷积架构，自动适应大尺寸地图)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ResUNet_RAILGUN(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch, base_ch = 7, 64
        self.inc = ResBlock(in_ch, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch * 2, base_ch * 4))
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec1 = ResBlock(base_ch * 4, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec2 = ResBlock(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, 5, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        up_x2 = self.up1(x3)
        concat_x2 = torch.cat([up_x2, x2], dim=1)
        d2 = self.dec1(concat_x2)
        up_x1 = self.up2(d2)
        concat_x1 = torch.cat([up_x1, x1], dim=1)
        d1 = self.dec2(concat_x1)
        return self.outc(d1)



# 2. 动态特征提取 (自动适配各种尺寸)
def get_bfs_distance_map(obs, gy, gx):
    H, W = obs.shape
    dist = np.full((H, W), 1e9, dtype=np.float32)
    dist[gy, gx] = 0.0
    q = deque([(gy, gx)])
    while q:
        y, x = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
                if dist[ny, nx] > dist[y, x] + 1:
                    dist[ny, nx] = dist[y, x] + 1
                    q.append((ny, nx))
    return dist


def build_7ch_features_dynamic(obs, agents_state):
    H, W = obs.shape
    f_map = torch.tensor(obs, dtype=torch.float32)
    f_cur = torch.zeros((H, W), dtype=torch.float32)
    f_goal = torch.zeros((H, W), dtype=torch.float32)
    f_cg = torch.zeros((H, W), dtype=torch.float32)
    f_grad_x = torch.zeros((H, W), dtype=torch.float32)
    f_grad_y = torch.zeros((H, W), dtype=torch.float32)
    f_capacity = torch.ones((H, W), dtype=torch.float32)
    f_capacity[obs >= 0.5] = 0.0

    for a in agents_state:
        cy, cx = a['cy'], a['cx']
        gy, gx = a['gy'], a['gx']
        f_cur[cy, cx] += 1.0
        f_goal[gy, gx] += 1.0
        dist_map = get_bfs_distance_map(obs, gy, gx)
        f_cg[cy, cx] = float(dist_map[cy, cx])
        if cx > 0 and dist_map[cy, cx - 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = -1.0
        elif cx < W - 1 and dist_map[cy, cx + 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = 1.0
        if cy > 0 and dist_map[cy - 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = -1.0
        elif cy < H - 1 and dist_map[cy + 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = 1.0
    return torch.stack([f_map, f_cur, f_goal, f_cg, f_grad_x, f_grad_y, f_capacity], dim=0)



# 3. 核心 PIBT 引擎
def run_neural_pibt(obs, agents_state, prob_map, use_neural=True):
    H, W = obs.shape
    N = len(agents_state)
    curr_pos = [(a['cy'], a['cx']) for a in agents_state]
    goals = [(a['gy'], a['gx']) for a in agents_state]
    action_map = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}

    ALPHA = 10.0 if use_neural else 0.0
    BETA = 1.0
    next_pos_table = {}
    undecided = [True] * N

    prio_indices = list(range(N))
    prio_indices.sort(key=lambda i: abs(curr_pos[i][0] - goals[i][0]) + abs(curr_pos[i][1] - goals[i][1]), reverse=True)

    def func_pibt(i, visited_agents):
        if not undecided[i]: return True
        if i in visited_agents: return False
        visited_agents.add(i)

        cy, cx = curr_pos[i]
        gy, gx = goals[i]
        cands = []
        for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
                dist = abs(ny - gy) + abs(nx - gx)
                act_idx = action_map[(dy, dx)]
                unet_prob = prob_map[act_idx, cy, cx].item() if prob_map is not None else 0.0
                score = ALPHA * unet_prob - BETA * dist
                cands.append((score, ny, nx))

        cands.sort(key=lambda x: x[0], reverse=True)

        for _, ny, nx in cands:
            if (ny, nx) in next_pos_table: continue
            occ = -1
            for oid, pos in enumerate(curr_pos):
                if oid != i and pos == (ny, nx): occ = oid; break
            if occ == -1:
                next_pos_table[(ny, nx)] = i
                undecided[i] = False
                return True
            if occ != -1 and undecided[occ]:
                if func_pibt(occ, visited_agents):
                    next_pos_table[(ny, nx)] = i
                    undecided[i] = False
                    return True
        if (cy, cx) not in next_pos_table:
            next_pos_table[(cy, cx)] = i
            undecided[i] = False
            return True
        return False

    for i in prio_indices:
        if undecided[i]: func_pibt(i, set())
    for i in range(N):
        if not undecided[i]:
            for pos, aid in next_pos_table.items():
                if aid == i: curr_pos[i] = pos; break
    return curr_pos


# 4. 在线主循环
def simulate_episode(model, obs, initial_agents, use_neural=True, max_steps=200):
    device = next(model.parameters()).device if model else "cpu"
    if model: model.eval()
    N = len(initial_agents)
    agents_state = [{'cy': a['sy'], 'cx': a['sx'], 'gy': a['gy'], 'gx': a['gx']} for a in initial_agents]

    for t in range(max_steps):
        all_reached = all((a['cy'] == a['gy'] and a['cx'] == a['gx']) for a in agents_state)
        if all_reached:
            print(f"  [Success] All {N} agents reached goals in {t} steps.")
            return t

        prob_map = None
        if use_neural:
            input_tensor = build_7ch_features_dynamic(obs, agents_state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                prob_map = F.softmax(logits, dim=1).squeeze(0).cpu()

        next_positions = run_neural_pibt(obs, agents_state, prob_map, use_neural)
        for i in range(N):
            agents_state[i]['cy'] = next_positions[i][0]
            agents_state[i]['cx'] = next_positions[i][1]

    print(f"  [Timeout] Failed to finish within {max_steps} steps. (Deadlock occurred)")
    return max_steps


# 5. 生成大地图
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Booting ZERO-SHOT Large Scale Evaluator on {device}...")

    model = ResUNet_RAILGUN().to(device)
    model.load_state_dict(torch.load("railgun_policy_final.pth"))
    print("Brain loaded. Model is ready to generalize!")

    # 创建 64x64 的大地图
    H, W = 64, 64
    obs = np.zeros((H, W), dtype=np.float32)
    # 中间建一堵长城，留下两个容量非常小的门
    obs[:, 31] = 1.0
    obs[20:23, 31] = 0.0  # 北门 (容量 3)
    obs[40:43, 31] = 0.0  # 南门 (容量 3)

    # 布置 40 个智能体大军
    agents = []
    # 20 个人在左边，要去右边
    for _ in range(20):
        sy, sx = random.randint(5, 58), random.randint(5, 15)
        gy, gx = random.randint(5, 58), random.randint(48, 58)
        agents.append({'sy': sy, 'sx': sx, 'gy': gy, 'gx': gx})

    # 20 个人在右边，要去左边
    for _ in range(20):
        sy, sx = random.randint(5, 58), random.randint(48, 58)
        gy, gx = random.randint(5, 58), random.randint(5, 15)
        agents.append({'sy': sy, 'sx': sx, 'gy': gy, 'gx': gx})

    print(f" Map Size: {H}x{W} | Agents: {len(agents)}")
    print("Match 1: Pure PIBT (Blind local search)...")
    steps_pure = simulate_episode(None, obs, agents, use_neural=False, max_steps=300)

    print("Match 2: Neural-PIBT (Guided by macroscopic flow)...")
    steps_neural = simulate_episode(model, obs, agents, use_neural=True, max_steps=300)


if __name__ == "__main__":
    main()