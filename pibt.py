import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np


# ==========================================
# 1. 配置与模型定义 (确保能完美加载你的权重)
# ==========================================
class Config:
    MODEL_PATH = "railgun_policy_final.pth"
    MAP_IN_CHANNELS = 7  # 障碍, 当前, 目标, 距离, 梯度x, 梯度y, 容量
    EMBED_DIM = 64
    H = 32
    W = 32


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
    def __init__(self, cfg):
        super().__init__()
        in_ch = cfg.MAP_IN_CHANNELS
        base_ch = cfg.EMBED_DIM
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


# ==========================================
# 2. 特征构建工具 (在线推理每一秒都要用)
# ==========================================
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


def build_7ch_features(obs, agents_state, cfg):
    """根据当前状态，实时构建 7 通道张量"""
    f_map = torch.tensor(obs, dtype=torch.float32)
    f_cur = torch.zeros((cfg.H, cfg.W), dtype=torch.float32)
    f_goal = torch.zeros((cfg.H, cfg.W), dtype=torch.float32)
    f_cg = torch.zeros((cfg.H, cfg.W), dtype=torch.float32)
    f_grad_x = torch.zeros((cfg.H, cfg.W), dtype=torch.float32)
    f_grad_y = torch.zeros((cfg.H, cfg.W), dtype=torch.float32)

    f_capacity = torch.ones((cfg.H, cfg.W), dtype=torch.float32)
    f_capacity[obs >= 0.5] = 0.0  # 障碍物容量为 0

    for a in agents_state:
        cy, cx = a['cy'], a['cx']
        gy, gx = a['gy'], a['gx']
        f_cur[cy, cx] += 1.0
        f_goal[gy, gx] += 1.0

        dist_map = get_bfs_distance_map(obs, gy, gx)
        f_cg[cy, cx] = float(dist_map[cy, cx])

        if cx > 0 and dist_map[cy, cx - 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = -1.0
        elif cx < cfg.W - 1 and dist_map[cy, cx + 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = 1.0

        if cy > 0 and dist_map[cy - 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = -1.0
        elif cy < cfg.H - 1 and dist_map[cy + 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = 1.0

    return torch.stack([f_map, f_cur, f_goal, f_cg, f_grad_x, f_grad_y, f_capacity], dim=0)


# ==========================================
# 3. 核心融合引擎：Neural-PIBT
# ==========================================
def run_neural_pibt(obs, agents_state, prob_map, use_neural=True):
    H, W = obs.shape
    N = len(agents_state)
    curr_pos = [(a['cy'], a['cx']) for a in agents_state]
    goals = [(a['gy'], a['gx']) for a in agents_state]

    action_map = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}

    # 调参中枢：你可以关闭神经网络，看看纯规则有多笨
    ALPHA = 10.0 if use_neural else 0.0
    BETA = 1.0

    next_pos_table = {}
    undecided = [True] * N

    # PIBT 优先级：离目标远的先走
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

                # 读取概率图
                unet_prob = prob_map[act_idx, cy, cx].item() if prob_map is not None else 0.0

                # 终极融合公式！
                score = ALPHA * unet_prob - BETA * dist
                cands.append((score, ny, nx))

        cands.sort(key=lambda x: x[0], reverse=True)  # 分数高的动作优先排查

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


# ==========================================
# 4. 在线循环调度器
# ==========================================
def simulate_episode(model, obs, initial_agents, cfg, use_neural=True, max_steps=100):
    device = next(model.parameters()).device
    model.eval()

    N = len(initial_agents)
    agents_state = [{'cy': a['sy'], 'cx': a['sx'], 'gy': a['gy'], 'gx': a['gx']} for a in initial_agents]
    makespan = 0

    for t in range(max_steps):
        all_reached = all((a['cy'] == a['gy'] and a['cx'] == a['gx']) for a in agents_state)
        if all_reached:
            print(f"  [Success] All agents reached goals in {t} steps.")
            return t

        prob_map = None
        if use_neural:
            input_tensor = build_7ch_features(obs, agents_state, cfg).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                prob_map = F.softmax(logits, dim=1).squeeze(0).cpu()  # [5, 32, 32]

        # 将决策交给底层引擎
        next_positions = run_neural_pibt(obs, agents_state, prob_map, use_neural)

        # 步进更新
        for i in range(N):
            agents_state[i]['cy'] = next_positions[i][0]
            agents_state[i]['cx'] = next_positions[i][1]

        makespan += 1

    print(f"  [Timeout] Failed to finish within {max_steps} steps.")
    return max_steps


# ==========================================
# 5. 主函数：跑个沙盒测试压压惊
# ==========================================
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载你的离线大脑
    model = ResUNet_RAILGUN(cfg).to(device)
    try:
        model.load_state_dict(torch.load(cfg.MODEL_PATH))
        print("Brain loaded successfully.")
    except Exception as e:
        print(f" Could not load {cfg.MODEL_PATH}. Running with untrained random weights for testing.")

    # 2. 构造一个刁钻的沙盒地图 (中间一堵墙，只有一个小缺口)
    obs = np.zeros((32, 32), dtype=np.float32)
    obs[:, 15] = 1.0  # 在第15列建一堵墙
    obs[15:18, 15] = 0.0  # 挖一个容量为 3 的小缺口，极易引发死锁

    # 3. 布置智能体 (左边一堆人要去右边，右边一堆人要去左边)
    agents = []
    # 左 -> 右
    agents.append({'sy': 10, 'sx': 5, 'gy': 10, 'gx': 25})
    agents.append({'sy': 12, 'sx': 5, 'gy': 12, 'gx': 25})
    agents.append({'sy': 14, 'sx': 5, 'gy': 14, 'gx': 25})
    # 右 -> 左
    agents.append({'sy': 15, 'sx': 26, 'gy': 15, 'gx': 6})
    agents.append({'sy': 17, 'sx': 26, 'gy': 17, 'gx': 6})
    agents.append({'sy': 19, 'sx': 26, 'gy': 19, 'gx': 6})

    print(" Match 1: Pure Traditional PIBT (Blind to macroscopic flow)")
    steps_pure = simulate_episode(model, obs, agents, cfg, use_neural=False)

    print(" Match 2: Neural-PIBT (Guided by your U-Net capacity map)")
    steps_neural = simulate_episode(model, obs, agents, cfg, use_neural=True)

    print(" --- Final Report ---")
    print(f"Pure PIBT steps: {steps_pure}")
    print(f"Neural-PIBT steps: {steps_neural}")
    if steps_neural < steps_pure:
        print(" Neural Network Successfully Resolved Deadlocks!")


if __name__ == "__main__":
    main()