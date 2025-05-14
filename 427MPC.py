# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import matplotlib.colors as mcolors
# --- 引入 MPC 相关库 ---
try:
    import cvxpy as cp
    from scipy.linalg import expm
    import time
    MPC_AVAILABLE = True
except ImportError:
    print("Warning: CVXPY or SciPy not found. MPC cost calculation will use fallback.")
    MPC_AVAILABLE = False
    def solve_mpc(*args, **kwargs): return np.zeros(3), None

# --- Constants ---
CURRENT_SCENARIO = 'Fig5'
NUM_SPACECRAFTS = 3  # !! 进一步减少 !!
NUM_TASKS = 2      # !! 进一步减少 !!
NUM_TASK_TYPES = 3
AREA_SIZE = 100
# MAX_GAME_ROUNDS = 5 # 不再直接控制游戏回合数，而是通过仿真时间
MAX_SIM_TIME_S = 150 # !! 总仿真物理时间 (秒) !!
GAME_ROUND_DURATION_S = 30 # !! 每个决策回合代表的物理时间 (秒) !!
OBSERVATIONS_PER_ROUND = 20 # 在每个决策回合结束时进行一次观测聚合
COMM_RANGE = 150 # 增大范围
STABLE_ITER_THRESHOLD = 3
MAX_COALITION_ITER = 10
PSEUDOCOUNT = 1.0; G0 = 9.80665
TASK_TYPE_MAP = {0: 'Easy', 1: 'Moderate', 2: 'Hard'}; TASK_REVENUE = {0: 300, 1: 500, 2: 1000}
TASK_RISK_COST = {0: 10, 1: 30, 2: 80}; TRUE_TASK_TYPES_FIG = [2, 1] # 确保长度 >= NUM_TASKS
TASK_COLORS = list(mcolors.TABLEAU_COLORS.values()); FIG2_TASK_COLORS = {1: 'red', 2: 'black', 3: 'magenta'}

# --- MPC Constants (如果可用) ---
if MPC_AVAILABLE:
    MU = 3.986004418e14; R_EARTH = 6378137.0; ALTITUDE = 726 * 1000; SEMI_MAJOR_AXIS = R_EARTH + ALTITUDE
    try: N_MEAN_MOTION = math.sqrt(MU / (SEMI_MAJOR_AXIS**3))
    except ValueError: N_MEAN_MOTION = 0.0011
    TS = 5.0         # !! 物理仿真和控制的时间步长 (s) !!
    NP = 5           # !! 预测时域 - 减小 !!
    NC = 3           # !! 控制时域 - 减小 !!
    QX = 1.0; QY = 1.0; QZ = 1.0; QVX = 0.0; QVY = 0.0; QVZ = 0.0; Q = np.diag([QX, QY, QZ, QVX, QVY, QVZ]) * 0.1
    R = np.diag([1.0, 1.0, 1.0]) * 1.0; PX = QX * 10; PY = QY * 10; PZ = QZ * 10; PVX = 0.0; PVY = 0.0; PVZ = 0.0
    P = np.diag([PX, PY, PZ, PVX, PVY, PVZ]); UMAX_COMPONENT = 0.05
else: N_MEAN_MOTION = 0.0011; TS = 5.0 # 仍需 TS 用于循环结构

# --- CW 模型 和 MPC 求解器 (如果可用) ---
if MPC_AVAILABLE:
    def get_cw_matrices(n): # (不变)
        A = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[3*n**2,0,0,0,2*n,0],[0,0,0,-2*n,0,0],[0,0,-n**2,0,0,0]])
        B = np.array([[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]]); return A, B
    def discretize_cw(A, B, Ts): # (不变)
        n_states = A.shape[0]; n_inputs = B.shape[1]; M = np.zeros((n_states + n_inputs, n_states + n_inputs)); M[:n_states, :n_states] = A; M[:n_states, n_states:] = B
        try: M_exp = expm(M * Ts); Ad = M_exp[:n_states, :n_states]; Bd = M_exp[:n_states, n_states:]
        except Exception as e: print(f"Error during discretization: {e}"); raise e
        return Ad, Bd
    def solve_mpc(current_state, target_state_ref, Ad, Bd, Np, Nc, Q, R, P, umax_comp): # (不变)
        n_states=Ad.shape[0]; n_inputs=Bd.shape[1]; U=cp.Variable((n_inputs, Nc)); X=cp.Variable((n_states, Np + 1)); cost=0
        for k in range(Np): u_k=U[:, k] if k<Nc else U[:, Nc-1]; cost+=cp.quad_form(X[:, k+1]-target_state_ref[:, k+1], Q);
        if k<Nc: cost+=cp.quad_form(U[:, k], R)
        cost+=cp.quad_form(X[:, Np]-target_state_ref[:, Np], P); constraints=[X[:, 0]==current_state]
        for k in range(Np): u_k=U[:, k] if k<Nc else U[:, Nc-1]; constraints+=[X[:, k+1]==Ad@X[:, k]+Bd@u_k]
        for k in range(Nc): constraints+=[cp.abs(U[:, k])<=umax_comp]
        problem=cp.Problem(cp.Minimize(cost), constraints); optimal_u_sequence=None
        solver_options={'max_iter': 2000, 'eps_abs': 1e-4, 'eps_rel': 1e-4};
        try: problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, **solver_options)
        except Exception: print(f"Solver OSQP failed. Trying SCS...");
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            try: problem.solve(solver=cp.SCS, warm_start=True, verbose=False);
            except Exception: print(f"Solver SCS also failed"); return None, None
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if U.value is not None: optimal_u_sequence=U.value; optimal_u0=optimal_u_sequence[:, 0]; return optimal_u0, optimal_u_sequence
            else: print(f"Solver status optimal/inaccurate but U.value is None."); return np.zeros(n_inputs), None
        else: print(f"MPC Problem status: {problem.status}"); return np.zeros(n_inputs), None

# --- Helper Functions ---
# (calculate_distance, get_neighbors_fig2_topology 不变)
# --- Helper Functions ---
def calculate_distance(pos1, pos2):
    """计算欧氏距离，增加内部检查"""
    # 检查输入是否为 None
    if pos1 is None or pos2 is None:
        # print("Debug: calculate_distance received None position.")
        return float('inf')

    # 使用 try...except 捕获潜在错误
    try:
        # 确保下面所有代码相对于 try 有正确的缩进
        x1, y1 = pos1
        x2, y2 = pos2

        # 检查坐标是否是有效数值
        if not all(isinstance(coord, (int, float)) and not math.isnan(coord) and not math.isinf(coord) for coord in [x1, y1, x2, y2]):
            print(f"Debug: Invalid coordinate type/value in calculate_distance: pos1={pos1}, pos2={pos2}")
            return float('inf') # 返回 inf 表示距离无效

        # 计算距离平方
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2

        # 理论上平方和不应为负，但增加检查以防万一
        if dist_sq < 0:
            print(f"Debug: Negative distance squared encountered: {dist_sq}")
            return float('inf')

        # 返回计算结果
        return math.sqrt(dist_sq)

    # except 块与 try 块对齐
    except (TypeError, IndexError, ValueError) as e: # 捕获更多可能的错误类型
        # 捕获可能的错误，例如 pos 不是包含两个数值的元组或索引错误
        print(f"Error inside calculate_distance with inputs {pos1}, {pos2}: {e}")
        return float('inf') # 在发生错误时返回 inf
def get_neighbors_fig2_topology(robot_id):
    topology = {0: [1, 3], 1: [0, 2], 2: [1, 3, 5], 3: [0, 2, 4, 6], 4: [3, 5, 7, 9], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7, 9], 9: [4, 8]}; complete_topology = defaultdict(list)
    # ... (不变) ...
    for u, neighbors in topology.items():
        for v in neighbors:
            if v not in complete_topology[u]: complete_topology[u].append(v)
            if u not in complete_topology[v]: complete_topology[v].append(u)
    return complete_topology.get(robot_id, [])


# --- Task Class ---
# (与之前相同)
class Task:
    def __init__(self, id, position, true_type):
        self.id = id; self.position = position; self.true_type = true_type
        # 占位符：目标的相对状态 (假设固定在原点)
        self.relative_state = np.array([position[0], position[1], 0.0, 0.0, 0.0, 0.0]) if position else np.zeros(6)
    def get_revenue(self, task_type): return TASK_REVENUE.get(task_type, 0)
    def calculate_risk_cost(self, robot, task_type): return TASK_RISK_COST.get(task_type, 0)
    def get_actual_revenue(self): return self.get_revenue(self.true_type)
    def calculate_actual_risk_cost(self, robot): return self.calculate_risk_cost(robot, self.true_type)
# --- Spacecraft Class (包含所有修正和方法) ---
class Spacecraft:
    def __init__(self, id, position, num_tasks, num_types,
                 initial_mass_kg, dry_mass_kg, isp_s,
                 obs_prob, initial_belief_type='uniform'):
        self.id = id
        self.position = position # 抽象 2D 位置
        # 6D 相对物理状态 [dx, dy, dz, dvx, dvy, dvz] - 占位符
        # !! 需要正确的初始化 !!
        self.relative_state = np.array([position[0], position[1], 0.0, 0.0, 0.0, 0.0])

        self.num_tasks = num_tasks; self.num_types = num_types
        self.initial_mass = initial_mass_kg; self.dry_mass = dry_mass_kg; self.isp = isp_s
        self.positive_observation_prob = obs_prob

        # MPC 矩阵预计算
        self.Ad = None
        self.Bd = None
        if MPC_AVAILABLE:
            try:
                A_cont, B_cont = get_cw_matrices(N_MEAN_MOTION)
                # *** 使用 ZOH 离散化，假设 u 是加速度 ***
                self.Ad, self.Bd = discretize_cw(A_cont, B_cont, TS)
            except Exception as e:
                 print(f"ERROR: Failed to initialize MPC matrices for SC {self.id}: {e}")
        # else:
        #      print(f"Info: MPC libraries not available for spacecraft {self.id}")


        # 信念、观测、联盟状态
        self.local_belief = np.zeros((num_tasks + 1, num_types))
        if initial_belief_type == 'uniform': self.local_belief[1:, :] = 1.0 / num_types
        elif initial_belief_type == 'arbitrary':
            for j in range(1, num_tasks + 1): random_vector = np.random.rand(num_types) + 1e-6; self.local_belief[j, :] = random_vector / np.sum(random_vector)
        else: self.local_belief[1:, :] = 1.0 / num_types
        self.local_belief[0, :] = 0; self.observation_matrix = np.zeros((num_tasks + 1, num_types))
        self.current_coalition_id = 0; self.local_partition = {}; self.partition_version = 0; self.partition_timestamp = random.random()
        self.neighbors = []; self.message_queue = []; self.needs_update = True
        # 新增: 存储轨迹
        self.trajectory = [list(self.relative_state)] # 记录初始状态

    # 物理状态更新方法
    def update_physical_state(self, control_input_u, dt):
        """根据离散CW动力学更新航天器的相对状态，并记录轨迹点"""
        if self.Ad is None or self.Bd is None: return

        if control_input_u is None: control_input_u = np.zeros(self.Bd.shape[1])
        control_input_u = np.array(control_input_u).flatten()
        if control_input_u.shape[0] != self.Bd.shape[1]: control_input_u = np.zeros(self.Bd.shape[1])

        try:
            if abs(dt - TS) > 1e-6: print(f"Warning: update_physical_state dt {dt} != TS {TS}")
            next_state = self.Ad @ self.relative_state + self.Bd @ control_input_u
            self.relative_state = next_state
            self.trajectory.append(list(self.relative_state))
            self.position = (self.relative_state[0], self.relative_state[1])
        except Exception as e:
            print(f"Error during state update calculation for SC {self.id}: {e}")

    # Delta-V 估算 (调用 MPC)
    def estimate_delta_v_for_task(self, target_task):
        """通过调用 MPC 假设性地求解来估算 Delta-V 成本。"""
        if not MPC_AVAILABLE or self.Ad is None or self.Bd is None:
            if target_task is None: return 0.0; distance = calculate_distance(self.position, target_task.position); estimated_dv = distance * 0.5; return max(10.0, min(estimated_dv, 1000.0))
        if target_task is None: return 0.0

        target_rel_state_final = target_task.relative_state
        target_state_reference = np.tile(target_rel_state_final, (NP + 1, 1)).T
        current_physical_state = self.relative_state # 使用自身的当前状态

        optimal_u0, optimal_u_sequence = solve_mpc(current_physical_state, target_state_reference, self.Ad, self.Bd, NP, NC, Q, R, P, UMAX_COMPONENT)

        if optimal_u_sequence is None: return float('inf')

        total_delta_v = 0.0; num_steps_to_sum = min(NC, optimal_u_sequence.shape[1])
        for k in range(num_steps_to_sum):
             control_vector = optimal_u_sequence[:, k]
             if control_vector is None: continue
             try: total_delta_v += np.linalg.norm(control_vector) * TS
             except TypeError: print(f"Warning: TypeError calculating norm..."); return float('inf')
        return total_delta_v

    # 燃料成本计算
    def calculate_fuel_cost(self, delta_v):
        """根据给定的 Delta-V (m/s) 计算所需的燃料质量 (kg)。"""
        # 如果无需速度增量，则无需燃料
        if delta_v <= 0:
            return 0.0
        # 如果比冲无效，则无法计算（成本无限大）
        if self.isp <= 0:
            # print(f"Warning: Invalid Isp ({self.isp}) for SC {self.id}.")
            return float('inf')

        # *** 添加缺失的 ve 计算 ***
        # 计算有效排气速度 ve = Isp * g0
        ve = self.isp * G0
        # *** -------------------- ***

        # 检查有效排气速度是否有效
        if ve <= 0:
            # print(f"Warning: Non-positive exhaust velocity ({ve}) calculated for SC {self.id}.")
            return float('inf')

        # 使用齐奥尔科夫斯基火箭方程计算质量比
        # delta_v = ve * ln(m_initial / m_final) => m_initial / m_final = exp(delta_v / ve)
        try:
            # 防止 delta_v / ve 过大导致溢出
            exponent = delta_v / ve
            if exponent > 700:  # exp(700) 已经非常大，接近溢出
                # print(f"Warning: Exponent {exponent} too large in rocket equation for SC {self.id}. Assuming infinite cost.")
                return float('inf')
            mass_ratio = math.exp(exponent)
        except OverflowError:
            print(f"Error: Overflow calculating mass_ratio for SC {self.id}, dv={delta_v}, ve={ve}")
            return float('inf')

        # 计算最终质量 (假设从初始满燃料状态开始估算)
        m_initial = self.initial_mass
        # 防止 mass_ratio 接近 0 或为 0
        if mass_ratio < 1e-9:
            # print(f"Warning: Mass ratio ({mass_ratio}) too small for SC {self.id}. Assuming infinite cost.")
            return float('inf')
        m_final = m_initial / mass_ratio

        # 计算消耗的燃料
        fuel_consumed = m_initial - m_final

        # 检查消耗的燃料是否超过初始携带量
        max_fuel = self.initial_mass - self.dry_mass
        # 使用小的容差比较浮点数
        if fuel_consumed > max_fuel + 1e-9:
            # print(f"Warning: Estimated fuel {fuel_consumed:.2f} exceeds max fuel {max_fuel:.2f} for dv={delta_v:.2f} for SC {self.id}")
            return float('inf')  # 燃料不足，成本无限大

        # 返回消耗的燃料质量作为成本 (可以乘以一个成本系数)
        cost_scaling_factor = 1.0  # 例子: 成本 = 燃料质量(kg)
        return fuel_consumed * cost_scaling_factor
    # 预期效用 (使用 MPC 燃料成本 - 包含 hypothetical_size 修正)
    def calculate_expected_utility(self, task_id, current_partition_view, all_tasks):
        if task_id == 0: return 0 # Void task has 0 utility

        coalition_j = current_partition_view.get(task_id, []) # Get current members list
        is_member = self.id in coalition_j # Check if self is already a member

        # *** 计算 hypothetical_size ***
        hypothetical_size = len(coalition_j) if is_member else len(coalition_j) + 1
        # *** ----------------------- ***

        # 确保 size 至少为 1
        if hypothetical_size == 0: hypothetical_size = 1

        expected_revenue_term = 0
        expected_risk_cost_term = 0
        task = all_tasks.get(task_id) # Get the task object

        if not task: return -float('inf') # Task不存在则效用负无穷

        # 计算预期收益和风险
        for k in range(self.num_types):
            belief_ikj = self.local_belief[task_id, k]
            revenue_k = task.get_revenue(k)
            risk_cost_ik = task.calculate_risk_cost(self, k)
            expected_revenue_term += belief_ikj * revenue_k
            expected_risk_cost_term += belief_ikj * risk_cost_ik

        expected_shared_revenue = expected_revenue_term / hypothetical_size

        # 计算估算的燃料成本
        estimated_delta_v = self.estimate_delta_v_for_task(task) # Pass task object
        fuel_cost = self.calculate_fuel_cost(estimated_delta_v)

        if math.isinf(fuel_cost): return -float('inf') # 燃料不足或MPC失败

        # 最终效用
        utility = expected_shared_revenue - expected_risk_cost_term - fuel_cost
        return utility

    # 实际效用 (使用 MPC 燃料成本 - 简化)
    def calculate_actual_utility(self, task_id, final_partition, all_tasks):
        if task_id == 0: return 0  # Void task utility is 0

        # 获取最终稳定分区中该任务的联盟成员列表
        coalition_j = final_partition.get(task_id, [])

        # *** 添加缺失的 coalition_size 计算 ***
        coalition_size = len(coalition_j)
        # *** --------------------------- ***

        # 检查联盟是否为空或自身是否不在联盟中
        # （如果不在联盟中，对该任务的实际效用贡献为 0）
        if coalition_size == 0 or self.id not in coalition_j:
            return 0

        # 获取任务对象
        task = all_tasks.get(task_id)
        if not task:
            print(f"Warning: Task {task_id} not found in actual utility calculation.")
            return 0  # 任务不存在则效用为 0

        # 获取实际收益和实际风险成本
        actual_revenue = task.get_actual_revenue()
        actual_risk_cost = task.calculate_actual_risk_cost(self)

        # 使用与预期效用相同的 *估算* 燃料成本 (因为没有模拟真实消耗)
        estimated_delta_v = self.estimate_delta_v_for_task(task)
        fuel_cost = self.calculate_fuel_cost(estimated_delta_v)

        # 处理燃料不足或 MPC 失败的情况
        if math.isinf(fuel_cost):
            # 如果估算成本无限大，意味着无法完成，实际效用也应极差或为0
            # 返回负无穷可能导致求和问题，返回0表示无贡献更安全
            return 0  # 或者可以返回一个非常大的负数

        # 计算实际共享收益
        actual_shared_revenue = actual_revenue / coalition_size

        # 计算最终实际效用
        utility = actual_shared_revenue - actual_risk_cost - fuel_cost
        return utility
    # 预期任务收益 (包含 task=None 初始化修正)
    def calculate_expected_task_revenue(self, task_id, all_tasks):
        task = None; # Initialize
        if task_id == 0: return 0; expected_revenue = 0; task = all_tasks.get(task_id);
        if not task: print(f"Warning: Task {task_id} not found..."); return 0;
        for k in range(self.num_types): belief_ikj = self.local_belief[task_id, k]; revenue_k = task.get_revenue(k); expected_revenue += belief_ikj * revenue_k;
        return expected_revenue

    # 贪婪选择 (不变)
    def greedy_selection(self, all_tasks):
        current_utility = self.calculate_expected_utility(self.current_coalition_id, self.local_partition, all_tasks)
        best_utility = current_utility if not math.isinf(current_utility) else -float('inf');
        best_task_id = self.current_coalition_id; changed = False; potential_tasks = list(range(self.num_tasks + 1))
        for task_id in potential_tasks:
             utility = self.calculate_expected_utility(task_id, self.local_partition, all_tasks)
             if not math.isinf(utility) and utility > best_utility + 1e-9: best_utility = utility; best_task_id = task_id;
        if best_task_id != self.current_coalition_id:
            old_coalition_id = self.current_coalition_id;
            if self.id in self.local_partition.get(old_coalition_id, []): self.local_partition[old_coalition_id].remove(self.id)
            if best_task_id not in self.local_partition: self.local_partition[best_task_id] = []
            if self.id not in self.local_partition[best_task_id]: self.local_partition[best_task_id].append(self.id)
            self.current_coalition_id = best_task_id; self.partition_version += 1; self.partition_timestamp = random.random(); changed = True; self.needs_update = True;
        return changed

    # 发送消息 (不变)
    def send_message(self, all_spacecrafts):
        partition_copy = copy.deepcopy(self.local_partition); message = {'sender_id': self.id, 'partition': partition_copy, 'version': self.partition_version, 'timestamp': self.partition_timestamp};
        for neighbor_id in self.neighbors:
             if neighbor_id in all_spacecrafts: all_spacecrafts[neighbor_id].message_queue.append(message)

    # 处理消息 (不变)
    def process_messages(self):
        if not self.message_queue: return False;
        own_info = {'sender_id': self.id, 'partition': copy.deepcopy(self.local_partition), 'version': self.partition_version, 'timestamp': self.partition_timestamp};
        received_info = [own_info] + [copy.deepcopy(msg) for msg in self.message_queue]; self.message_queue = [];
        dominant_info = own_info; changed_by_message = False;
        for msg in received_info:
            is_dominant = False;
            if msg['version'] > dominant_info['version']: is_dominant = True;
            elif msg['version'] == dominant_info['version'] and msg['timestamp'] > dominant_info['timestamp']: is_dominant = True;
            if is_dominant: dominant_info = msg;
        if dominant_info['sender_id'] != self.id:
             self.local_partition = dominant_info['partition']; self.partition_version = dominant_info['version']; self.partition_timestamp = dominant_info['timestamp'];
             found_self = False;
             for task_id, members in self.local_partition.items():
                 if self.id in members: self.current_coalition_id = task_id; found_self = True; break;
             if not found_self: self.current_coalition_id = 0;
             if 0 not in self.local_partition: self.local_partition[0] = [];
             if self.id not in self.local_partition[0]: self.local_partition[0].append(self.id);
             changed_by_message = True; self.needs_update = True;
        return changed_by_message

    # 观测 (不变)
    def take_observation(self, task):
        """模拟观测任务类型"""
        # 检查传入的 task 是否有效
        if task is None:
            # print(f"Warning: take_observation called with None task for SC {self.id}")
            return None  # 无法观测 None 任务

        # *** 添加缺失的赋值语句 ***
        # 从 task 对象获取真实的类型
        true_type = task.true_type
        # *** ------------------ ***

        # 以 P(观测正确) = self.positive_observation_prob 的概率返回真实类型
        if random.random() < self.positive_observation_prob:
            return true_type
        else:
            # 以 1 - P(观测正确) 的概率观测到错误的类型
            # 从所有可能的类型中，排除真实类型，随机选择一个错误的类型
            possible_false_types = [k for k in range(self.num_types) if k != true_type]
            # 如果没有其他错误类型可选（例如只有一种类型），则只能返回真实类型
            if not possible_false_types:
                return true_type
            # 随机选择一个错误的类型返回
            return random.choice(possible_false_types)

    # 信念更新 (不变)
    def update_belief_from_observations(self, aggregated_observations, pseudocount=1.0):
        self.observation_matrix += aggregated_observations;
        for j in range(1, self.num_tasks + 1):
            current_cumulative_counts_j = self.observation_matrix[j, :];
            alpha_j = current_cumulative_counts_j + pseudocount; sum_alpha_j = np.sum(alpha_j);
            if sum_alpha_j > 1e-9: self.local_belief[j, :] = alpha_j / sum_alpha_j;
            else: self.local_belief[j, :] = 1.0 / self.num_types

# --- Simulation Setup (修正邻居计算中的 UnboundLocalError) ---
def setup_simulation(num_spacecrafts, num_tasks, num_types, scenario='Fig4'):
    spacecrafts = {}
    tasks = {}
    initial_positions = {'robots': {}, 'tasks': {}} # Keep key 'robots'
    print(f"Setting up Simulation for Scenario: {scenario}...")
    belief_cfg = 'arbitrary' if scenario == 'Fig5' else 'uniform'
    if belief_cfg == 'arbitrary': print("  - Using ARBITRARY initial beliefs.")
    # Task Setup (不变)
    print(f"  - Tasks: {num_tasks}")
    # ... (Task setup logic as before) ...
    if scenario == 'Fig2':
        task_positions_fig2 = {1: (75, 75), 2: (15, 30), 3: (70, 30), 4: (45, 20), 5: (60, 45), 6: (85, 15)} if num_tasks >=6 else {}
        for j in range(1, num_tasks + 1): pos = task_positions_fig2.get(j, (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE))); true_type = TRUE_TASK_TYPES_FIG[j-1] if j-1 < len(TRUE_TASK_TYPES_FIG) else random.randrange(num_types); tasks[j] = Task(j, pos, true_type); initial_positions['tasks'][j] = pos; tasks[j].relative_state = np.array([pos[0], pos[1], 0, 0, 0, 0])
    else:
        for j in range(1, num_tasks + 1): pos = (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)); true_type = TRUE_TASK_TYPES_FIG[j-1] if j-1 < len(TRUE_TASK_TYPES_FIG) else random.randrange(num_types); tasks[j] = Task(j, pos, true_type); initial_positions['tasks'][j] = pos; tasks[j].relative_state = np.array([pos[0], pos[1], 0, 0, 0, 0])

    # Spacecraft Setup (不变)
    print(f"  - Spacecraft: {num_spacecrafts}")
    initial_partition = {j: [] for j in range(num_tasks + 1)}; all_spacecraft_ids = list(range(num_spacecrafts)); initial_partition[0] = all_spacecraft_ids; DEFAULT_INITIAL_MASS = 1000; DEFAULT_DRY_MASS = 200; DEFAULT_ISP = 300;
    # ... (Spacecraft initialization loop as before) ...
    if scenario == 'Fig2':
        robot_positions_fig2 = {0: (10, 20), 1: (10, 60), 2: (30, 40), 3: (40, 70), 4: (55, 55), 5: (50, 90), 6: (65, 25), 7: (60, 80), 8: (80, 60), 9: (95, 40)} if num_spacecrafts >= 10 else {};
        for i in range(num_spacecrafts): pos = robot_positions_fig2.get(i, (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE))); initial_positions['robots'][i] = pos; obs_prob = random.uniform(0.9, 1.0); spacecrafts[i] = Spacecraft(i, pos, num_tasks, num_types, DEFAULT_INITIAL_MASS, DEFAULT_DRY_MASS, DEFAULT_ISP, obs_prob, belief_cfg); spacecrafts[i].local_partition = copy.deepcopy(initial_partition); spacecrafts[i].current_coalition_id = 0; spacecrafts[i].relative_state = np.array([pos[0], pos[1], 0, 0, 0, 0])
    else:
        for i in range(num_spacecrafts): pos = (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)); initial_positions['robots'][i] = pos; obs_prob = random.uniform(0.9, 1.0); spacecrafts[i] = Spacecraft(i, pos, num_tasks, num_types, DEFAULT_INITIAL_MASS, DEFAULT_DRY_MASS, DEFAULT_ISP, obs_prob, belief_cfg); spacecrafts[i].local_partition = copy.deepcopy(initial_partition); spacecrafts[i].current_coalition_id = 0; spacecrafts[i].relative_state = np.array([pos[0], pos[1], 0, 0, 0, 0])


    # Neighbor Setup
    print("  - Setting up Neighbors...")
    initial_neighbors = {}
    if scenario == 'Fig2':
         for i in spacecrafts: spacecrafts[i].neighbors = get_neighbors_fig2_topology(i); initial_neighbors[i] = list(spacecrafts[i].neighbors)
    else:
         # --- Fallback Neighbor Logic with Error Checking & Initialization ---
         for i in spacecrafts: spacecrafts[i].neighbors = []; # Start fresh
         for i in spacecrafts:
            robot_pos = spacecrafts[i].position
            if robot_pos is None:
                # print(f"Warning: Spacecraft {i} has None position. Skipping neighbor calculation.")
                continue
            # *** Iterate through potential neighbors ***
            for other_id, other_craft in spacecrafts.items():
                 # *** Initialize other_pos at the start of the inner loop ***
                 other_pos = None
                 # ----------------------------------------------------
                 if i == other_id: continue # Skip self

                 try: # Wrap potential error sources in try block
                     other_pos = other_craft.position # Assign other_pos
                     if other_pos is None: # Check if assignment resulted in None
                         # print(f"Warning: Neighbor candidate {other_id} has None position. Skipping.")
                         continue

                     dist = calculate_distance(robot_pos, other_pos)
                     # Check type and value before comparison
                     if isinstance(dist, (int, float)) and not math.isinf(dist) and not math.isnan(dist):
                         if dist <= COMM_RANGE:
                             spacecrafts[i].neighbors.append(other_id)
                     # elif math.isinf(dist): pass
                     # else: print(f"Warning: calculate_distance returned non-numeric/inf/nan: {dist}")
                 except Exception as e:
                     # Now it's safer to print other_pos here, it's either None or the value it had before error
                     print(f"Error calculating distance between {i} (pos={robot_pos}) and {other_id} (pos={other_pos}): {e}")
         # --- End Fallback Neighbor Logic ---

                 # ... (邻居计算和双向性确保的代码) ...

                 # 确保双向性和基本连通性 (与之前相同)
                 for i in spacecrafts:
                     # 确保其他节点知道这个连接 (双向性)
                     # (之前的双向性循环代码保持不变)
                     for neighbor_id in spacecrafts[i].neighbors:
                         if neighbor_id in spacecrafts and i not in spacecrafts[neighbor_id].neighbors:
                             spacecrafts[neighbor_id].neighbors.append(i)

                     # *** 处理孤立节点 ***
                     if not spacecrafts[i].neighbors and len(spacecrafts) > 1:  # 如果节点 i 没有邻居且不止一个航天器
                         # !!! 确保下面这行存在且缩进正确 !!!
                         fallback_id = 0 if i != 0 else 1
                         # ------------------------------------

                         # 检查计算出的 fallback_id 是否有效存在
                         if fallback_id in spacecrafts:
                             # 添加备用连接 (如果尚未存在)
                             if fallback_id not in spacecrafts[i].neighbors:
                                 spacecrafts[i].neighbors.append(fallback_id)
                             # 确保备用连接也是双向的
                             if i not in spacecrafts[fallback_id].neighbors:
                                 spacecrafts[fallback_id].neighbors.append(i)
                             # print(f"    Fallback connection applied: {i} <-> {fallback_id}") # 调试信息
                         else:
                             # 如果计算出的 fallback_id 不存在 (例如只有0号航天器时 i=0, fallback=1 不存在)
                             print(f"Warning: Fallback ID {fallback_id} not found for isolated node {i}.")
                     # ------------------------

                     # 存储初始邻居用于绘图
                     initial_neighbors[i] = list(spacecrafts[i].neighbors)

            print("Setup Complete.")
            return spacecrafts, tasks, initial_positions, initial_neighbors

# --- 其他代码部分 (Spacecraft 类, Task 类, 绘图函数, run_simulation 等) ---
# ... (保持与上一次修正 UnboundLocalError in calculate_expected_task_revenue 后的版本一致) ...
# ... (这里省略了大量重复的代码，请确保您使用的是包含之前所有修正的完整版本) ...

# --- 运行入口 (保持不变) ---
# if __name__ == "__main__":
#     print(f"--- Starting ... ---")
#     ...
#     run_simulation(scenario=CURRENT_SCENARIO)

# --- Plotting Functions ---
# (与之前相同)
def plot_fig2(initial_positions, initial_neighbors, final_partition, tasks):
    # ... (不变) ...
    print("Generating Figure 2 plots..."); fig, axes = plt.subplots(1, 2, figsize=(16, 7)); ax1, ax2 = axes; ax1.set_title('Fig 2a: Initial State & Communication Topology'); ax1.set_xlim(-5, AREA_SIZE + 5); ax1.set_ylim(-5, AREA_SIZE + 5); ax1.set_xlabel('x-axis (m)'); ax1.set_ylabel('y-axis (m)'); ax1.set_aspect('equal', adjustable='box');
    for task_id, task in tasks.items(): pos = task.position; color = FIG2_TASK_COLORS.get(task_id, 'gray'); ax1.plot(pos[0], pos[1], '*', markersize=12, color=color); ax1.text(pos[0] + 1, pos[1] + 1, f'$t_{task_id}$', fontsize=9)
    robot_pos_map = initial_positions['robots']
    for robot_id, pos in robot_pos_map.items(): ax1.plot(pos[0], pos[1], 'o', markersize=8, color='purple', alpha=0.7); ax1.text(pos[0] + 1, pos[1] + 1, f'$r_{robot_id+1}$', fontsize=9)
    plotted_links = set();
    for robot_id, neighbors in initial_neighbors.items():
        pos1 = robot_pos_map.get(robot_id);
        if pos1 is None: continue;
        for neighbor_id in neighbors:
            if neighbor_id == robot_id: continue; link = tuple(sorted((robot_id, neighbor_id)));
            if link in plotted_links: continue; pos2 = robot_pos_map.get(neighbor_id);
            if pos2 is None: continue; ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], '--', color='blue', linewidth=0.8, alpha=0.6); plotted_links.add(link)
    ax1.grid(True, linestyle=':', alpha=0.6); ax2.set_title('Fig 2b: Final Coalition Formation Result'); ax2.set_xlim(-5, AREA_SIZE + 5); ax2.set_ylim(-5, AREA_SIZE + 5); ax2.set_xlabel('x-axis (m)'); ax2.set_ylabel('y-axis (m)'); ax2.set_aspect('equal', adjustable='box');
    for task_id, task in tasks.items(): pos = task.position; color = FIG2_TASK_COLORS.get(task_id, 'gray'); ax2.plot(pos[0], pos[1], '*', markersize=12, color=color); ax2.text(pos[0] + 1, pos[1] + 1, f'$t_{task_id}$', fontsize=9)
    safe_final_partition = {int(k): list(map(int, v)) for k, v in final_partition.items()}
    for task_id, members in safe_final_partition.items():
         coalition_color = FIG2_TASK_COLORS.get(task_id, 'gray');
         if task_id == 0: coalition_color = 'lightgrey';
         for robot_id in members:
              pos = robot_pos_map.get(robot_id);
              if pos is None: continue; ax2.plot(pos[0], pos[1], 'o', markersize=8, color=coalition_color, markeredgecolor='black', linewidth=0.5); ax2.text(pos[0] + 1, pos[1] + 1, f'$r_{robot_id+1}$', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); plt.savefig('fig2_replication.png', dpi=150); print("Figure 2 plot saved as fig2_replication.png")

def plot_fig3(global_utility_history):
    # ... (不变) ...
    print("Generating Figure 3 plot..."); fig, ax = plt.subplots(figsize=(8, 5)); valid_indices = [i for i, util in enumerate(global_utility_history) if not np.isnan(util)]
    if not valid_indices: print("No valid utility data to plot for Fig 3."); return
    rounds_axis = [i for i in valid_indices]; valid_utility = [global_utility_history[i] for i in valid_indices]
    ax.plot(rounds_axis, valid_utility, marker='s', markersize=4, linestyle='-', color='blue')
    ax.set_title('Fig 3: Global Utility Convergence (MPC Fuel Cost)'); ax.set_xlabel('Index of game'); ax.set_ylabel('Global utility')
    ax.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); plt.savefig('fig3_replication.png', dpi=150); print("Figure 3 plot saved as fig3_replication.png")

def plot_revenue_evolution(expected_revenue_history, tasks, title_prefix="Fig 4"):
    # ... (不变) ...
    print(f"Generating {title_prefix} plots..."); num_rounds_plotted = 0
    if expected_revenue_history: max_len = 0;
    for j in expected_revenue_history: valid_indices = [i for i, rev in enumerate(expected_revenue_history[j]) if not np.isnan(rev)];
    if valid_indices: max_len = max(max_len, valid_indices[-1] + 1); num_rounds_plotted = max_len
    if num_rounds_plotted == 0: print(f"No history data to plot for {title_prefix}."); return
    rounds_axis = range(num_rounds_plotted); fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True); axes = axes.flatten()
    actual_revenues = {j: tasks[j].get_actual_revenue() for j in range(1, NUM_TASKS + 1) if j in tasks}
    for j in range(1, NUM_TASKS + 1):
        if j-1 >= len(axes): continue
        ax = axes[j-1]; history = expected_revenue_history.get(j, [float('nan')] * num_rounds_plotted)[:num_rounds_plotted]
        valid_indices = [i for i, rev in enumerate(history) if not np.isnan(rev)];
        if not valid_indices: continue
        valid_rounds = [rounds_axis[i] for i in valid_indices]; valid_history = [history[i] for i in valid_indices]
        ax.plot(valid_rounds, valid_history, marker='.', markersize=4, linestyle='-', label="Exp. Rev (SC 0)")
        if j in actual_revenues: ax.axhline(y=actual_revenues[j], color='r', linestyle='--', linewidth=1.5, label=f'Actual ({actual_revenues[j]:.0f})')
        ax.set_title(f'Task $t_{j}$');
        if j > 3: ax.set_xlabel('Index of game');
        if j == 1 or j == 4: ax.set_ylabel('Expected task revenue')
        ax.grid(True, linestyle=':', alpha=0.7); ax.legend(fontsize=8)
        min_rev = min(TASK_REVENUE.values()); max_rev = max(TASK_REVENUE.values()); ax.set_ylim(min_rev - 150, max_rev + 150)
    plt.suptitle(f'{title_prefix}: Evolution of Expected Task Revenue (Spacecraft 0 View)'); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(f'{title_prefix}_replication.png', dpi=150); print(f"{title_prefix} plot saved as {title_prefix}_replication.png")

# --- NEW: Plot Trajectories ---
def plot_trajectories(spacecrafts, tasks, scenario):
    """绘制航天器的相对轨迹 (x-y 平面投影)"""
    print("Generating Trajectory Plot...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'Spacecraft Relative Trajectories (XY Projection) - Scenario: {scenario}')
    ax.set_xlabel('Relative X (m)')
    ax.set_ylabel('Relative Y (m)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')

    # 绘制目标位置 (作为参考)
    for task_id, task in tasks.items():
        pos = task.position # 使用抽象位置
        ax.plot(pos[0], pos[1], 'k*', markersize=10, label=f'Target {task_id} (Abstract Pos)')

    # 绘制每个航天器的轨迹
    for sc_id, craft in spacecrafts.items():
        traj = np.array(craft.trajectory) # 获取存储的轨迹点
        if traj.shape[0] > 1:
            ax.plot(traj[:, 0], traj[:, 1], marker='.', markersize=2, linestyle='-', label=f'SC {sc_id}')
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=6) # 起点
            ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=6) # 终点
        elif traj.shape[0] == 1: # 只有一个点
             ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=6, label=f'SC {sc_id} (Start)')


    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(f'trajectory_plot_{scenario}.png', dpi=150)
    print(f"Trajectory plot saved as trajectory_plot_{scenario}.png")

# *** 新增: 信念演化绘图函数 ***
def plot_belief_evolution(spacecrafts, tasks, num_total_rounds, title_prefix="Belief Evolution"):
    """绘制指定航天器对任务信念的变化"""
    print(f"Generating {title_prefix} plots...")
    if not spacecrafts: return

    sc_id_to_plot = 0 # 选择航天器 0 进行绘制
    if sc_id_to_plot not in spacecrafts:
        print(f"Spacecraft {sc_id_to_plot} not found for belief plotting.")
        if spacecrafts: sc_id_to_plot = next(iter(spacecrafts))
        else: return

    craft_to_plot = spacecrafts[sc_id_to_plot]
    belief_history = craft_to_plot.belief_history

    num_tasks_to_plot = len(tasks)
    if num_tasks_to_plot == 0: print("No tasks to plot beliefs for."); return

    ncols = min(3, num_tasks_to_plot) # 每行最多3个子图
    nrows = math.ceil(num_tasks_to_plot / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True, squeeze=False)
    axes = axes.flatten()

    # 确定实际绘制的回合数 (信念历史记录包含初始状态，所以长度是回合数+1)
    actual_rounds_in_history = len(next(iter(belief_history.values()), [])) # 长度为 N+1
    plot_rounds_indices = range(actual_rounds_in_history) # X轴: 0, 1, ..., N

    for task_id in range(1, num_tasks_to_plot + 1):
        ax_idx = task_id - 1
        if ax_idx >= len(axes): break # 避免索引超出 axes 范围
        ax = axes[ax_idx]

        if task_id not in belief_history or not belief_history[task_id]:
            ax.set_title(f'Task $t_{task_id}$: No belief history')
            continue

        history_array = np.array(belief_history[task_id]) # (N+1, num_types)

        if history_array.shape[0] == 0: continue # 没有数据

        # 绘制每种类型的信念概率变化
        for type_k in range(NUM_TASK_TYPES):
             ax.plot(plot_rounds_indices, history_array[:, type_k], marker='.', markersize=3, linestyle='-', label=f'Type {type_k} ({TASK_TYPE_MAP.get(type_k, "Unknown")})')

        # 标记真实类型
        if task_id in tasks: # 确保任务存在
             true_type = tasks[task_id].true_type
             ax.set_title(f'SC {sc_id_to_plot} Belief: Task $t_{task_id}$ (True: {true_type})')
        else:
             ax.set_title(f'SC {sc_id_to_plot} Belief: Task $t_{task_id}$')

        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5)
        ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=0.5)
        ax.set_ylim(-0.1, 1.1)
        if ax_idx // ncols == nrows - 1 : ax.set_xlabel('Index of game round (0=initial)')
        if ax_idx % ncols == 0: ax.set_ylabel('Belief Probability')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8)
        # 设置 X 轴刻度
        ax.set_xticks(np.arange(0, actual_rounds_in_history, max(1, actual_rounds_in_history // 5)))


    # 隐藏多余的子图
    for i in range(num_tasks_to_plot, nrows * ncols):
        if i < len(axes): fig.delaxes(axes[i])

    plt.suptitle(f'{title_prefix}: Belief Evolution for Spacecraft {sc_id_to_plot}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{title_prefix}_SC{sc_id_to_plot}.png', dpi=150)
    print(f"{title_prefix} plot saved as {title_prefix}_SC{sc_id_to_plot}.png")

# *** 新增: 简化甘特图绘制函数 ***
def plot_gantt_chart(assignment_history, num_spacecrafts, num_total_rounds):
    """绘制一个简化的甘特图，显示每个航天器在每个回合的任务分配"""
    print("Generating Simplified Gantt Chart...")
    if not assignment_history: return

    # assignment_history[0] 是 round 0 (初始) 的分配
    # assignment_history[1] 是 round 1 结束后的分配
    # 我们绘制 round 1 到 num_total_rounds 的分配情况
    history_to_plot = assignment_history[1:]
    plot_num_rounds = len(history_to_plot)
    if plot_num_rounds == 0:
        print("No assignment history (after round 0) to plot for Gantt chart.")
        return

    fig, ax = plt.subplots(figsize=(max(8, plot_num_rounds * 0.6), max(4, num_spacecrafts * 0.5)))
    ax.set_title('Simplified Gantt Chart (Task Assignment per Round)')
    ax.set_xlabel('Game Round Index (1 to N)')
    ax.set_ylabel('Spacecraft ID')

    # y 轴刻度
    y_ticks = np.arange(num_spacecrafts)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'SC {i}' for i in range(num_spacecrafts)])
    ax.invert_yaxis() # 让 SC 0 在顶部

    # x 轴刻度 (从 1 开始)
    ax.set_xticks(np.arange(1, plot_num_rounds + 1, max(1, plot_num_rounds // 10)))
    ax.set_xlim(0.5, plot_num_rounds + 0.5)

    # 准备颜色
    unique_task_ids = set()
    for assignments in history_to_plot: unique_task_ids.update(tid for tid in assignments.values() if tid != 0)
    task_color_map_gantt = {tid: TASK_COLORS[i % len(TASK_COLORS)] for i, tid in enumerate(sorted(list(unique_task_ids)))}; task_color_map_gantt[0] = 'lightgrey';

    # 绘制条形
    for round_idx_zero_based, assignments in enumerate(history_to_plot):
        round_num = round_idx_zero_based + 1 # 回合数从 1 开始绘图
        for sc_id, task_id in assignments.items():
            if sc_id >= num_spacecrafts: continue
            color = task_color_map_gantt.get(task_id, 'white');
            ax.barh(y=sc_id, width=1, left=round_num - 0.5, height=0.6, color=color, edgecolor='black', linewidth=0.5);
            if task_id != 0:
                rgb_color = mcolors.to_rgb(color); text_color = 'white' if np.mean(rgb_color) < 0.5 else 'black';
                ax.text(round_num, sc_id, f'T{task_id}', ha='center', va='center', fontsize=8, color=text_color)
    ax.grid(axis='x', linestyle=':', alpha=0.7); plt.tight_layout(); plt.savefig('gantt_chart_simplified.png', dpi=150); print("Simplified Gantt chart saved as gantt_chart_simplified.png")


# --- Main Simulation Loop (集成内外循环) ---
def run_simulation(scenario):
    spacecrafts, tasks, initial_positions, initial_neighbors = setup_simulation(
        NUM_SPACECRAFTS, NUM_TASKS, NUM_TASK_TYPES, scenario=scenario
    )

    # History Tracking
    expected_revenue_history_sc0 = defaultdict(lambda: []) # 使用列表动态记录
    global_utility_history = [] # 使用列表动态记录
    # 记录初始状态
    current_time_s = 0.0
    if 0 in spacecrafts:
         for j in range(1, NUM_TASKS + 1): expected_revenue_history_sc0[j].append(spacecrafts[0].calculate_expected_task_revenue(j, tasks))
         initial_partition_sc0 = copy.deepcopy(spacecrafts[0].local_partition); initial_utility = 0
         for i, craft in spacecrafts.items(): initial_utility += craft.calculate_actual_utility(craft.current_coalition_id, initial_partition_sc0, tasks)
         global_utility_history.append(initial_utility)

    # Simulation
    current_global_partition = copy.deepcopy(spacecrafts[0].local_partition if 0 in spacecrafts else {})
    game_round = 0 # 外循环计数器
    last_coalition_update_time = -GAME_ROUND_DURATION_S # 确保第一次执行决策

    print("\nStarting Simulation with INNER/OUTER LOOP...")
    if MPC_AVAILABLE: print("!!! WARNING: MPC calls active, will be VERY SLOW !!!")
    else: print("--- INFO: MPC NOT AVAILABLE - Using simplified distance cost fallback ---")

    # --- 外循环: 按物理时间推进 ---
    while current_time_s < MAX_SIM_TIME_S:

        outer_loop_start_time = time.time()
        current_round_start_time = current_time_s
        print(f"\n--- Outer Loop (Decision/Coalition) - Current Time: {current_time_s:.2f}s ---")
        game_round += 1

        # --- 决策层: 联盟形成 (Algorithm 1) ---
        # (保持不变，但现在只在每个外循环开始时运行一次)
        print(f"  Running Coalition Formation (Game Round {game_round})...")
        coalition_start_time = time.time()
        stable_iterations = 0; coalition_formation_iter = 0
        while True:
            coalition_formation_iter += 1;
            if coalition_formation_iter > MAX_COALITION_ITER: print(f"Warning: Alg 1 max iterations ({MAX_COALITION_ITER})."); break
            num_crafts_changed = 0;
            for i in spacecrafts: spacecrafts[i].needs_update = False
            # --- Greedy Selection (调用包含 MPC 成本估算的效用函数) ---
            # print(f"    Alg 1 Iter {coalition_formation_iter}: Greedy Phase...")
            greedy_start_time = time.time()
            for i in spacecrafts:
                 if spacecrafts[i].greedy_selection(tasks): num_crafts_changed += 1
            # print(f"      Greedy Phase took {time.time() - greedy_start_time:.2f}s")
            # --- Consensus Phase ---
            for i in spacecrafts: spacecrafts[i].send_message(spacecrafts)
            consensus_changed_count = 0
            for i in spacecrafts:
                 if spacecrafts[i].process_messages(): consensus_changed_count += 1
            # --- Check Stability ---
            total_changed_this_iter = 0
            for i in spacecrafts:
                 if spacecrafts[i].needs_update : total_changed_this_iter += 1
            if total_changed_this_iter == 0:
                stable_iterations += 1
                if stable_iterations >= STABLE_ITER_THRESHOLD: print(f"    Alg 1 Converged after {coalition_formation_iter} iterations."); break
            else: stable_iterations = 0
        print(f"  Coalition Formation took {time.time() - coalition_start_time:.2f}s")
        # --- End Algorithm 1 ---

        # 获取稳定的联盟划分结果
        stable_partition = copy.deepcopy(spacecrafts[0].local_partition if 0 in spacecrafts else {})
        stable_partition_intkeys = {int(k): v for k, v in stable_partition.items()}
        # 更新每个航天器的当前联盟 ID (基于共识结果) - 这一步很重要!
        for craft_id, craft in spacecrafts.items():
            found = False
            for task_id, members in stable_partition_intkeys.items():
                 if craft_id in members:
                      craft.current_coalition_id = task_id
                      found = True
                      break
            if not found: # 如果没在任何联盟里 (可能吗?)，设为 0
                 craft.current_coalition_id = 0


        # --- 内循环: 物理仿真 (步长 TS) ---
        print(f"  Running Inner Physics Loop from {current_round_start_time:.2f}s to {current_round_start_time + GAME_ROUND_DURATION_S:.2f}s (step={TS}s)...")
        num_inner_steps = int(GAME_ROUND_DURATION_S / TS)
        inner_loop_start_time = time.time()

        for step in range(num_inner_steps):
            current_inner_time = current_round_start_time + step * TS
            if current_inner_time >= MAX_SIM_TIME_S: break # 避免超过总时间

            # --- MPC 控制计算 (仅对已分配任务的航天器) ---
            control_inputs = {} # 存储当前时间步的控制输入
            if MPC_AVAILABLE:
                for craft_id, craft in spacecrafts.items():
                    target_id = craft.current_coalition_id # 获取当前分配的目标
                    if target_id is not None and target_id != 0:
                         target_task = tasks.get(target_id)
                         if target_task:
                              target_rel_state_final = target_task.relative_state # 使用目标状态
                              ref_traj = np.tile(target_rel_state_final, (NP + 1, 1)).T
                              # 调用 MPC
                              u0, _ = solve_mpc(craft.relative_state, ref_traj, craft.Ad, craft.Bd, NP, NC, Q, R, P, UMAX_COMPONENT)
                              control_inputs[craft_id] = u0 if u0 is not None else np.zeros(craft.Bd.shape[1])
                         else:
                              control_inputs[craft_id] = np.zeros(craft.Bd.shape[1]) # 目标不存在
                    else:
                         control_inputs[craft_id] = np.zeros(craft.Bd.shape[1]) # 未分配任务
            else: # 如果 MPC 不可用，所有控制为 0
                 for craft_id in spacecrafts: control_inputs[craft_id] = np.zeros(3)

            # --- 物理状态更新 (所有航天器) ---
            for craft_id, craft in spacecrafts.items():
                 control_to_apply = control_inputs.get(craft_id, np.zeros(3)) # 获取控制输入
                 craft.update_physical_state(control_to_apply, TS) # 使用 TS 更新状态

            # --- (可选) 更新目标状态 ---
            # update_all_target_physical_states(TS)

            # --- (可选) 在内循环中进行观测和信念更新? ---
            # all_observations = simulate_observations(...)
            # update_target_state_beliefs(...)

            # 更新当前物理时间
            current_time_s += TS

        print(f"    Inner Loop ({num_inner_steps} steps) took {time.time() - inner_loop_start_time:.2f}s")

        # --- 决策层: 观测聚合与信念更新 (在每个外循环结束时执行) ---
        print(f"  Running Observation & Belief Update...")
        # --- Observation Phase ---
        aggregated_observations = np.zeros((NUM_TASKS + 1, NUM_TASK_TYPES))
        for i, craft in spacecrafts.items():
            task_id = craft.current_coalition_id;
            if task_id != 0:
                # 在这里获取 task 是为了 take_observation
                task_for_observation = tasks.get(task_id);  # 可以用新名字避免混淆
                if task_for_observation:
                    for _ in range(OBSERVATIONS_PER_ROUND):
                        obs_type = craft.take_observation(task_for_observation);
                        if obs_type is not None and 0 <= obs_type < NUM_TASK_TYPES:
                            aggregated_observations[task_id, obs_type] += 1

        # --- Belief Update Phase ---
        print(f"  Running Observation & Belief Update...")  # 移到这里打印更合适
        for i, craft in spacecrafts.items():
            # *** 不再需要 'if task:' 判断 ***
            # 直接调用更新函数，它会更新该 craft 对所有 task 的信念
            craft.update_belief_from_observations(aggregated_observations, pseudocount=PSEUDOCOUNT)
        # --- Belief Update Phase 结束 ---

        # --- 记录外循环结束时的状态用于绘图 ---
        if 0 in spacecrafts:
             for j in range(1, NUM_TASKS + 1): expected_revenue_history_sc0[j].append(spacecrafts[0].calculate_expected_task_revenue(j, tasks))
        # 计算全局效用
        current_global_utility = 0
        for i, craft in spacecrafts.items():
             assigned_task_id = craft.current_coalition_id
             current_global_utility += craft.calculate_actual_utility(assigned_task_id, stable_partition_intkeys, tasks)
        global_utility_history.append(current_global_utility)
        print(f"Global Utility (End of Outer Loop {game_round}): {current_global_utility:.2f}")


        print(f"--- Outer Loop {game_round} finished in {time.time() - outer_loop_start_time:.2f} seconds (Sim Time: {current_time_s:.2f}s) ---")

        # 检查是否应该提前结束 (例如所有任务完成)
        # ...

    # --- Post Simulation & Plotting ---
    final_round_count = game_round
    print(f"\nSimulation finished after {final_round_count} outer loops ({current_time_s:.2f}s simulated time).")

    # Generate Plots
    if scenario == 'Fig2': plot_fig2(initial_positions, initial_neighbors, final_partition_for_plotting, tasks)

    # 调整绘图 X 轴为 Game Round
    plot_fig3(global_utility_history) # X轴是游戏回合
    # 调整预期收益绘图 (需要修改绘图函数以处理列表)
    # plot_revenue_evolution(expected_revenue_history_sc0, tasks, title_prefix="Revenue Evolution")

    # *** 新增: 绘制轨迹 ***
    plot_trajectories(spacecrafts, tasks, scenario)

    plt.show()


# --- Run ---
if __name__ == "__main__":
    print(f"--- Starting Coalition Formation with INNER/OUTER LOOP ---")
    print(f"--- N={NUM_SPACECRAFTS}, M={NUM_TASKS}, SimTime={MAX_SIM_TIME_S}s ---")
    if MPC_AVAILABLE: print(f"--- MPC Params: Np={NP}, Nc={NC}, Ts={TS}s ---")
    else: print(f"--- MPC NOT AVAILABLE - Physical state update disabled ---")
    print(f"!!! SIMULATION WILL BE VERY SLOW IF MPC IS AVAILABLE !!!")
    run_simulation(scenario=CURRENT_SCENARIO)