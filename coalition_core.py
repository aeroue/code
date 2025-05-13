# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import matplotlib.colors as mcolors
import time # Ensure time is imported

# --- 引入 MPC 相关库 ---
try:
    import cvxpy as cp
    from scipy.linalg import expm
    MPC_AVAILABLE = True
except ImportError:
    print("Warning: CVXPY or SciPy not found. MPC cost calculation will use fallback, control plot will be empty.")
    MPC_AVAILABLE = False
    # Define a dummy solve_mpc if MPC is not available
    def solve_mpc(*args, **kwargs): return np.zeros(3), None

# --- Constants ---
CURRENT_SCENARIO = 'Fig5' # 可选 'Fig2', 'Fig4', 'Fig5' 或其他自定义场景
NUM_SPACECRAFTS = 3      # 航天器数量
NUM_TASKS = 2          # 任务数量
NUM_TASK_TYPES = 3     # 任务类型数量 (e.g., Easy, Moderate, Hard)
AREA_SIZE = 100        # 抽象 2D 区域大小 (用于初始化位置)
MAX_SIM_TIME_S = 150   # 最大物理仿真时间 (秒)
GAME_ROUND_DURATION_S = 30 # 每个决策回合/外循环代表的物理时间 (秒)
OBSERVATIONS_PER_ROUND = 20 # 每个航天器在外循环结束时进行的观测次数
COMM_RANGE = 150       # 通信范围 (用于基于距离的邻居发现)
STABLE_ITER_THRESHOLD = 3  # 联盟形成算法判定收敛所需的稳定迭代次数
MAX_COALITION_ITER = 10    # 联盟形成算法的最大迭代次数，防止死循环
PSEUDOCOUNT = 1.0      # 狄利克雷信念更新的伪计数 (避免概率为0)
G0 = 9.80665           # 标准重力加速度 (m/s^2), 用于 Isp 计算
ARRIVAL_DISTANCE_THRESHOLD = 5.0 # 到达目标的位置距离阈值 (米) - 可调整
TASK_TYPE_MAP = {0: 'Easy', 1: 'Moderate', 2: 'Hard'} # 任务类型名称映射
TASK_REVENUE = {0: 300, 1: 500, 2: 1000} # 不同类型任务的基础收益
TASK_RISK_COST = {0: 10, 1: 30, 2: 80}   # 不同类型任务的风险成本
# 确保 TRUE_TASK_TYPES_FIG 列表长度 >= NUM_TASKS
TRUE_TASK_TYPES_FIG = [2, 1] # 预设的真实任务类型 (例如用于 Fig5, Task 1=Hard, Task 2=Moderate)
TASK_COLORS = list(mcolors.TABLEAU_COLORS.values()) # 用于绘图的颜色列表
# Fig2 场景特定颜色 (如果任务数少于6，则可能只用到部分)
FIG2_TASK_COLORS = {1: 'red', 2: 'black', 3: 'magenta', 4: 'blue', 5: 'green', 6: 'orange'}

# --- MPC Constants (if available) ---
if MPC_AVAILABLE:
    MU = 3.986004418e14  # 地球引力常数 (m^3/s^2)
    R_EARTH = 6378137.0 # 地球半径 (m)
    ALTITUDE = 726 * 1000 # 轨道高度 (m)
    SEMI_MAJOR_AXIS = R_EARTH + ALTITUDE # 轨道半长轴 (m)
    try:
        N_MEAN_MOTION = math.sqrt(MU / (SEMI_MAJOR_AXIS**3)) # 平均角速度 (rad/s)
    except ValueError:
        print("Warning: Invalid semi-major axis for mean motion calculation. Using default.")
        N_MEAN_MOTION = 0.0011 # 默认备用值
    TS = 5.0         # 物理仿真和 MPC 控制的时间步长 (s) - 必须 > 0
    NP = 5           # MPC 预测时域 (步数) - 减小以提高速度
    NC = 3           # MPC 控制时域 (步数) - 减小以提高速度
    # MPC 代价函数权重 (可调整)
    QX = 1.0; QY = 1.0; QZ = 1.0; QVX = 0.0; QVY = 0.0; QVZ = 0.0 # 状态权重 (位置重要，速度次要)
    Q = np.diag([QX, QY, QZ, QVX, QVY, QVZ]) * 0.1 # 状态代价矩阵 Q
    R_MPC = np.diag([1.0, 1.0, 1.0]) * 1.0 # 控制代价矩阵 R (重命名以避免与集合 R 冲突)
    PX = QX * 10; PY = QY * 10; PZ = QZ * 10; PVX = 0.0; PVY = 0.0; PVZ = 0.0 # 终端状态权重 (通常比 Q 大)
    P = np.diag([PX, PY, PZ, PVX, PVY, PVZ]) # 终端状态代价矩阵 P
    UMAX_COMPONENT = 0.05 # 每个轴的最大控制输入 (m/s^2, 示例)
else:
    # 如果 MPC 不可用，仍需 TS 用于循环结构
    N_MEAN_MOTION = 0.0011 # 伪值
    TS = 5.0         # 仿真时间步长

# --- CW Model and MPC Solver (if available) ---
if MPC_AVAILABLE:
    def get_cw_matrices(n):
        """ 获取连续时间 Clohessy-Wiltshire 状态空间矩阵 A 和 B。 """
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * n**2, 0, 0, 0, 2 * n, 0],
            [0, 0, 0, -2 * n, 0, 0],
            [0, 0, -n**2, 0, 0, 0]
        ])
        B = np.array([
            [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ])
        return A, B

    def discretize_cw(A, B, Ts):
        """ 使用零阶保持器 (ZOH) 离散化 CW 矩阵。 """
        n_states = A.shape[0]
        n_inputs = B.shape[1]
        # 构造增广矩阵用于离散化
        M = np.zeros((n_states + n_inputs, n_states + n_inputs))
        M[:n_states, :n_states] = A
        M[:n_states, n_states:] = B
        try:
            M_exp = expm(M * Ts) # 计算矩阵指数
            Ad = M_exp[:n_states, :n_states] # 离散状态矩阵
            Bd = M_exp[:n_states, n_states:] # 离散输入矩阵
        except Exception as e:
            print(f"Error during discretization with Ts={Ts}: {e}")
            # 离散化失败时的备用方案
            Ad = np.identity(n_states)
            Bd = np.zeros((n_states, n_inputs))
            # raise # 可以选择重新抛出异常以停止执行
        return Ad, Bd

    def solve_mpc(current_state, target_state_ref, Ad, Bd, Np, Nc, Q, R_param, P, umax_comp):
        """ 求解一步 MPC 问题。 """
        n_states = Ad.shape[0]
        n_inputs = Bd.shape[1]

        # 定义 CVXPY 优化变量
        U = cp.Variable((n_inputs, Nc), name="U_control") # 控制序列 (控制时域)
        X = cp.Variable((n_states, Np + 1), name="X_state") # 状态序列 (预测时域)

        # 定义代价函数
        cost = 0
        for k in range(Np): # 遍历预测时域
            # 如果超出控制时域，则重复使用最后一个控制输入
            u_k = U[:, k] if k < Nc else U[:, Nc - 1]
            # 状态代价：惩罚状态与参考轨迹的偏差
            # 注意: target_state_ref 的维度应该是 (n_states, Np+1)
            if target_state_ref.shape[1] > k+1:
                 cost += cp.quad_form(X[:, k + 1] - target_state_ref[:, k + 1], Q)
            else: # 如果参考轨迹不够长 (理论上不应发生)，使用最后一个参考点
                 cost += cp.quad_form(X[:, k + 1] - target_state_ref[:, -1], Q)

            # 控制代价：惩罚控制输入的大小 (仅在控制时域内)
            if k < Nc:
                cost += cp.quad_form(u_k, R_param)
        # 终端代价：惩罚预测时域末端状态与参考轨迹末端状态的偏差
        if target_state_ref.shape[1] > Np:
             cost += cp.quad_form(X[:, Np] - target_state_ref[:, Np], P)
        else:
             cost += cp.quad_form(X[:, Np] - target_state_ref[:, -1], P)


        # 定义约束条件
        constraints = [X[:, 0] == current_state] # 初始状态约束
        # 系统动力学约束
        for k in range(Np):
            u_k = U[:, k] if k < Nc else U[:, Nc - 1]
            constraints += [X[:, k + 1] == Ad @ X[:, k] + Bd @ u_k]
        # 控制输入大小约束
        for k in range(Nc):
            constraints += [cp.abs(U[:, k]) <= umax_comp] # 限制每个轴的绝对值

        # 定义优化问题
        problem = cp.Problem(cp.Minimize(cost), constraints)
        optimal_u_sequence = None
        optimal_u0 = np.zeros(n_inputs) # 默认返回零控制

        # 设置求解器选项 (OSQP 通常对 MPC 效果较好)
        # 减少迭代次数和放宽容差可能提高速度，但可能牺牲精度
        solver_options = {'max_iter': 1000, 'eps_abs': 1e-3, 'eps_rel': 1e-3, 'verbose': False}

        try:
            # 使用 OSQP 求解
            problem.solve(solver=cp.OSQP, warm_start=True, **solver_options)
        except Exception as e_osqp:
            # print(f"Solver OSQP failed: {e_osqp}. Trying SCS...") # 打印信息可能过多
            # 如果 OSQP 失败，尝试使用 SCS
            try:
                problem.solve(solver=cp.SCS, warm_start=True, verbose=False)
            except Exception as e_scs:
                print(f"Warning: Solver SCS also failed: {e_scs}. Returning zero control.")
                return optimal_u0, None # 两个求解器都失败，返回零控制

        # 检查求解结果状态
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if U.value is not None:
                optimal_u_sequence = U.value
                optimal_u0 = optimal_u_sequence[:, 0] # 获取第一个控制输入
                # print(f"Debug MPC: Optimal u0 found: {optimal_u0}") # 可选调试信息
            else:
                # 状态为最优但 U.value 为 None 的情况
                # print(f"Warning: Solver status optimal/inaccurate but U.value is None.")
                pass # 保持 optimal_u0 为零
        elif problem.status == cp.INFEASIBLE or problem.status == cp.UNBOUNDED:
             # print(f"Warning: MPC Problem is {problem.status}. Returning zero control.")
             pass
        else: # 其他失败状态
            # print(f"Warning: MPC Problem status: {problem.status}. Returning zero control.")
            pass # 保持 optimal_u0 为零

        return optimal_u0, optimal_u_sequence

# --- Helper Functions ---
def calculate_distance(pos1, pos2):
    """ 计算两个 2D 点之间的欧氏距离。 """
    if pos1 is None or pos2 is None: return float('inf')
    try:
        x1, y1 = pos1; x2, y2 = pos2
        # 基础类型检查
        if not all(isinstance(c, (int, float)) and np.isfinite(c) for c in [x1, y1, x2, y2]):
             return float('inf') # 处理非数字或 inf/nan
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
        return math.sqrt(dist_sq)
    except (TypeError, ValueError, IndexError):
        return float('inf') # 处理解包或类型错误

def get_neighbors_fig2_topology(robot_id, num_robots=10):
    """ 为 Fig 2 场景定义固定的通信拓扑。 """
    # 基础拓扑 (如果 num_robots < 10 会自动过滤)
    topology = {
        0: [1, 3], 1: [0, 2], 2: [1, 3, 5], 3: [0, 2, 4, 6],
        4: [3, 5, 7, 9], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8],
        8: [5, 7, 9], 9: [4, 8]
    }
    # 根据实际航天器数量过滤拓扑
    valid_ids = set(range(num_robots))
    filtered_topology = defaultdict(list)
    for u, neighbors in topology.items():
        if u in valid_ids:
            # 只保留在有效 ID 集合中的邻居
            filtered_topology[u] = [v for v in neighbors if v in valid_ids]

    # 确保在有效航天器之间是双向连接
    complete_topology = defaultdict(list)
    for u, neighbors in filtered_topology.items():
        for v in neighbors:
            if v not in complete_topology[u]: complete_topology[u].append(v)
            if u not in complete_topology[v]: complete_topology[v].append(u)

    return complete_topology.get(robot_id, []) # 返回指定 ID 的邻居列表，若无则为空列表


# --- Task Class ---
class Task:
    """ 代表一个任务及其属性。 """
    def __init__(self, id, position, true_type):
        self.id = id
        self.position = position # 2D 抽象位置 (用于初始化和绘图)
        self.true_type = true_type # 任务的真实类型 (整数索引)
        # 目标的相对状态 (6D: [x, y, z, vx, vy, vz])
        # 假设目标在抽象位置，相对速度为零
        self.relative_state = np.array([position[0], position[1], 0.0, 0.0, 0.0, 0.0]) if position else np.zeros(6)

    def get_revenue(self, task_type):
        """ 获取指定类型的任务收益。 """
        return TASK_REVENUE.get(task_type, 0) # 若类型无效返回 0

    def calculate_risk_cost(self, robot, task_type):
        """ 计算执行指定类型任务的风险成本。 """
        # 在此实现中，风险成本仅与任务类型相关
        return TASK_RISK_COST.get(task_type, 0)

    def get_actual_revenue(self):
        """ 获取任务的实际收益（基于真实类型）。 """
        return self.get_revenue(self.true_type)

    def calculate_actual_risk_cost(self, robot):
        """ 计算执行任务的实际风险成本。 """
        return self.calculate_risk_cost(robot, self.true_type)

# --- Spacecraft Class ---
class Spacecraft:
    """ 代表一个航天器及其状态和行为。 """
    def __init__(self, id, position, num_tasks, num_types,
                 initial_mass_kg, dry_mass_kg, isp_s,
                 obs_prob, initial_belief_type='uniform'):
        self.id = id
        self.position = position # 2D 抽象位置
        # 6D 相对物理状态 [dx, dy, dz, dvx, dvy, dvz]
        self.relative_state = np.array([position[0], position[1], 0.0, 0.0, 0.0, 0.0])

        self.num_tasks = num_tasks
        self.num_types = num_types
        self.initial_mass = initial_mass_kg
        self.current_mass = initial_mass_kg # 当前质量，会因燃料消耗减少
        self.dry_mass = dry_mass_kg       # 干重
        self.isp = isp_s                  # 比冲 (s)
        self.positive_observation_prob = obs_prob # 正确观测任务类型的概率

        # MPC 离散化矩阵
        self.Ad = None
        self.Bd = None
        if MPC_AVAILABLE and TS > 0:
            try:
                A_cont, B_cont = get_cw_matrices(N_MEAN_MOTION)
                self.Ad, self.Bd = discretize_cw(A_cont, B_cont, TS)
                if self.Ad is None or self.Bd is None:
                     print(f"Warning: Discretization failed for SC {self.id}. MPC will be ineffective.")
                     self.Ad = np.identity(6); self.Bd = np.zeros((6, 3)) # 备用
            except Exception as e:
                 print(f"ERROR: Failed to initialize MPC matrices for SC {self.id}: {e}")
                 self.Ad = np.identity(6); self.Bd = np.zeros((6, 3)) # 备用

        # 信念、观测、联盟状态
        self.local_belief = np.zeros((num_tasks + 1, num_types)) # 任务 0 为虚拟任务
        self.observation_counts = np.zeros((num_tasks + 1, num_types)) # 累计观测计数
        self.initialize_beliefs(initial_belief_type) # 初始化信念和计数

        self.current_coalition_id = 0 # 初始加入虚拟联盟
        self.local_partition = {} # 对联盟划分的本地视图
        self.partition_version = 0 # 本地划分的版本号
        self.partition_timestamp = random.random() # 用于打破版本号冲突的时间戳
        self.neighbors = [] # 邻居列表
        self.message_queue = [] # 接收到的消息队列
        self.needs_update = True # 标记状态是否需要更新（用于联盟收敛判断）

        # 绘图用历史数据
        self.trajectory = [list(self.relative_state)] # 6D 状态轨迹
        self.control_history = [] # 控制输入历史 (time, control_vector)
        self.belief_history = defaultdict(list) # 信念历史 (task_id -> list of belief_vectors)
        # 记录初始信念状态
        self.record_belief_state()

    def initialize_beliefs(self, belief_type):
        """ 初始化本地信念。 """
        self.local_belief[0, :] = 0.0 # 虚拟任务信念为 0
        # 初始化观测计数（用于狄利克雷更新）
        # 每个类型的初始计数值 = 伪计数 / 类型数
        self.observation_counts[:, :] = PSEUDOCOUNT / self.num_types if self.num_types > 0 else 0

        if belief_type == 'uniform' or self.num_types == 0:
            if self.num_types > 0:
                 self.local_belief[1:, :] = 1.0 / self.num_types
            else: # 处理类型数为0的边缘情况
                 self.local_belief[1:, :] = 0.0
        elif belief_type == 'arbitrary':
            for j in range(1, self.num_tasks + 1):
                if self.num_types > 0:
                     random_vector = np.random.rand(self.num_types) + 1e-9 # 避免全零
                     self.local_belief[j, :] = random_vector / np.sum(random_vector)
                     # 稍微根据初始信念调整初始计数
                     if j < self.observation_counts.shape[0]:
                          self.observation_counts[j, :] = self.local_belief[j, :] * PSEUDOCOUNT
                else:
                     self.local_belief[j, :] = 0.0 # 类型数为0
        else: # 默认为 uniform
             if self.num_types > 0:
                  self.local_belief[1:, :] = 1.0 / self.num_types
             else:
                  self.local_belief[1:, :] = 0.0

    def update_physical_state(self, control_input_u, dt):
        """ 使用离散 CW 动力学更新航天器相对状态并记录轨迹。 """
        if not MPC_AVAILABLE or self.Ad is None or self.Bd is None or dt <= 0:
            self.trajectory.append(list(self.relative_state)) # 记录静态状态
            return

        # 确保控制输入有效
        if control_input_u is None: control_input_u = np.zeros(self.Bd.shape[1])
        control_input_u = np.array(control_input_u).flatten()
        if control_input_u.shape[0] != self.Bd.shape[1]:
            # print(f"Warning: Invalid control input shape for SC {self.id}. Using zeros.")
            control_input_u = np.zeros(self.Bd.shape[1])

        try:
            # x[k+1] = Ad*x[k] + Bd*u[k]
            next_state = self.Ad @ self.relative_state + self.Bd @ control_input_u
            self.relative_state = next_state
            # 基于 6D 状态更新 2D 抽象位置
            self.position = (self.relative_state[0], self.relative_state[1])
            self.trajectory.append(list(self.relative_state)) # 记录新状态
        except Exception as e:
            print(f"Error during state update calculation for SC {self.id}: {e}")
            self.trajectory.append(list(self.relative_state)) # 记录当前状态以保持长度

    def estimate_delta_v_for_task(self, target_task):
        """ 使用 MPC 仿真估算到达目标任务所需的 Delta-V 成本。 """
        if target_task is None: return 0.0 # 虚拟任务成本为 0

        if not MPC_AVAILABLE or self.Ad is None or self.Bd is None:
            # MPC 不可用或矩阵无效时的备用方案：基于距离估算
            if self.position and target_task.position:
                 distance = calculate_distance(self.position, target_task.position)
                 estimated_dv = distance * 0.1 # 简化比例因子，需校准
                 return max(0.1, min(estimated_dv, 50.0)) # 增加上下限
            else:
                 return float('inf') # 位置未知，无法估算

        # MPC 参考轨迹：保持在目标状态
        target_rel_state_final = target_task.relative_state
        # 确保目标状态是 6D 向量
        if target_rel_state_final is None or target_rel_state_final.shape != (6,):
             print(f"Warning: Invalid target state for Task {target_task.id}. Using zeros.")
             target_rel_state_final = np.zeros(6)
        # 创建参考轨迹 (Np+1 步都保持目标状态)
        target_state_reference = np.tile(target_rel_state_final, (NP + 1, 1)).T

        current_physical_state = self.relative_state # 当前 6D 状态

        # 假设性地求解 MPC 问题以获取控制序列
        optimal_u0, optimal_u_sequence = solve_mpc(
            current_physical_state, target_state_reference,
            self.Ad, self.Bd, NP, NC, Q, R_MPC, P, UMAX_COMPONENT
        )

        # 若 MPC 求解失败，成本视为无限大
        if optimal_u_sequence is None:
            return float('inf')

        # 从计划的控制序列计算总 Delta-V (通常累加控制时域 Nc 步)
        total_delta_v = 0.0
        num_steps_to_sum = min(NC, optimal_u_sequence.shape[1])
        for k in range(num_steps_to_sum):
             control_vector = optimal_u_sequence[:, k]
             if control_vector is not None:
                 try:
                     # 单步 Delta-V = ||加速度|| * dt
                     total_delta_v += np.linalg.norm(control_vector) * TS
                 except (TypeError, ValueError):
                     # print(f"Warning: TypeError/ValueError calculating norm for control vector {control_vector}")
                     return float('inf') # 控制向量无效
        return total_delta_v

    def calculate_fuel_cost(self, delta_v):
        """ 使用齐奥尔科夫斯基火箭方程计算给定 Delta-V 所需的燃料质量。 """
        if delta_v <= 1e-6: return 0.0 # 极小 dv 不耗燃料
        if self.isp <= 0 or self.current_mass <= self.dry_mass:
             return float('inf') # Isp 无效或无燃料

        # 计算有效排气速度 ve = Isp * g0
        ve = self.isp * G0
        if ve <= 0: return float('inf')

        # 计算质量比 m_initial / m_final = exp(delta_v / ve)
        try:
            exponent = delta_v / ve
            if exponent > 600: return float('inf') # 防止 exp 溢出
            mass_ratio = math.exp(exponent)
        except OverflowError:
            return float('inf')

        # 计算消耗的燃料 m_fuel = m_initial * (1 - 1 / mass_ratio)
        m_initial_for_calc = self.current_mass # 基于当前质量估算
        if mass_ratio < 1e-9: return float('inf') # 避免除以零

        fuel_consumed = m_initial_for_calc * (1.0 - 1.0 / mass_ratio)

        # 检查所需燃料是否超过可用燃料
        available_fuel = self.current_mass - self.dry_mass
        if fuel_consumed > available_fuel + 1e-9: # 考虑浮点误差
             return float('inf') # 燃料不足

        # 返回燃料质量作为成本（可乘以成本系数）
        cost_scaling_factor = 1.0 # 示例：成本 = 燃料质量 (kg)
        return fuel_consumed * cost_scaling_factor

    def consume_fuel(self, delta_v_applied):
        """ 根据实际施加的 Delta-V 消耗燃料并更新当前质量。 """
        if delta_v_applied <= 1e-6: return # 无需消耗

        # Note: calculate_fuel_cost uses current_mass, so it correctly estimates fuel for this step
        fuel_needed = self.calculate_fuel_cost(delta_v_applied) # 计算所需燃料
        if not math.isinf(fuel_needed):
            available_fuel = self.current_mass - self.dry_mass
            fuel_to_consume = min(fuel_needed, available_fuel) # 只能消耗可用燃料
            self.current_mass -= fuel_to_consume
            # 确保质量不低于干重
            if self.current_mass < self.dry_mass: self.current_mass = self.dry_mass

    def calculate_expected_utility(self, task_id, current_partition_view, all_tasks):
        """ 计算加入某个任务联盟的预期效用。 """
        if task_id == 0: return 0.0 # 虚拟任务效用为 0

        task = all_tasks.get(task_id)
        if not task: return -float('inf') # 任务不存在

        # 计算假设加入后的联盟大小
        coalition_j = current_partition_view.get(task_id, [])
        is_member = self.id in coalition_j
        hypothetical_size = len(coalition_j) if is_member else len(coalition_j) + 1
        if hypothetical_size == 0: hypothetical_size = 1 # 避免除零

        # 基于本地信念计算预期收益和风险
        expected_revenue_term = 0.0
        expected_risk_cost_term = 0.0
        if task_id < self.local_belief.shape[0] and self.num_types > 0: # 检查任务 ID 和类型数
            task_beliefs = self.local_belief[task_id, :]
            for k in range(self.num_types):
                belief_ikj = task_beliefs[k]
                revenue_k = task.get_revenue(k)
                risk_cost_ik = task.calculate_risk_cost(self, k)
                expected_revenue_term += belief_ikj * revenue_k
                expected_risk_cost_term += belief_ikj * risk_cost_ik
        elif self.num_types == 0: # No types means no revenue/risk
             pass
        else: # 如果任务 ID 超出范围 (不应发生)
             return -float('inf')

        # 预期共享收益
        expected_shared_revenue = expected_revenue_term / hypothetical_size

        # 估算燃料成本
        estimated_delta_v = self.estimate_delta_v_for_task(task)
        fuel_cost = self.calculate_fuel_cost(estimated_delta_v)

        # 若燃料成本无限大，则效用无限小
        if math.isinf(fuel_cost): return -float('inf')

        # 总预期效用
        utility = expected_shared_revenue - expected_risk_cost_term - fuel_cost
        return utility

    def calculate_actual_utility(self, task_id, final_partition, all_tasks):
        """ 计算基于最终联盟划分实现的实际效用。 """
        if task_id == 0: return 0.0

        # 获取该任务的最终联盟成员
        coalition_j = final_partition.get(task_id, [])
        coalition_size = len(coalition_j)

        # 若本航天器不在最终联盟中，则对此任务贡献的效用为 0
        if coalition_size == 0 or self.id not in coalition_j:
            return 0.0

        task = all_tasks.get(task_id)
        if not task: return 0.0 # 任务不存在

        # 基于真实类型计算实际收益和风险
        actual_revenue = task.get_actual_revenue()
        actual_risk_cost = task.calculate_actual_risk_cost(self)

        # 实际共享收益
        actual_shared_revenue = actual_revenue / coalition_size

        # 使用执行此任务的 *估算* 燃料成本
        estimated_delta_v = self.estimate_delta_v_for_task(task)
        # Need to calculate fuel cost based on the mass *before* the maneuver potentially started
        # This is tricky without storing historical mass. Using current mass is an approximation.
        fuel_cost = self.calculate_fuel_cost(estimated_delta_v)

        # 若估算成本无限大，则实际效用也极差或为 0
        if math.isinf(fuel_cost): return 0.0

        # 最终实际效用
        utility = actual_shared_revenue - actual_risk_cost - fuel_cost
        return utility

    def calculate_expected_task_revenue(self, task_id, all_tasks):
        """ 基于当前信念计算任务的预期收益。 """
        if task_id == 0: return 0.0
        task = all_tasks.get(task_id)
        if not task: return 0.0

        expected_revenue = 0.0
        if task_id < self.local_belief.shape[0] and self.num_types > 0: # 检查边界和类型数
             task_beliefs = self.local_belief[task_id, :]
             for k in range(self.num_types):
                 belief_ikj = task_beliefs[k]
                 revenue_k = task.get_revenue(k)
                 expected_revenue += belief_ikj * revenue_k
        return expected_revenue

    def greedy_selection(self, all_tasks):
        """ 执行 Algorithm 1 的贪婪选择阶段。 """
        # 当前选择的效用
        current_utility = self.calculate_expected_utility(self.current_coalition_id, self.local_partition, all_tasks)
        best_utility = current_utility if not math.isinf(current_utility) else -float('inf')
        best_task_id = self.current_coalition_id
        changed = False

        # 评估切换到其他任务（包括虚拟任务 0）的效用
        potential_tasks = list(range(self.num_tasks + 1))
        for task_id in potential_tasks:
            if task_id == self.current_coalition_id: continue # 跳过当前选择

            utility = self.calculate_expected_utility(task_id, self.local_partition, all_tasks)

            # 如果新任务提供严格更优的效用 (使用浮点误差容忍)
            if not math.isinf(utility) and utility > best_utility + 1e-9:
                best_utility = utility
                best_task_id = task_id

        # 如果找到了更优的任务，更新本地联盟视图
        if best_task_id != self.current_coalition_id:
            old_coalition_id = self.current_coalition_id
            # 从旧联盟移除
            if old_coalition_id in self.local_partition and self.id in self.local_partition[old_coalition_id]:
                try: self.local_partition[old_coalition_id].remove(self.id)
                except ValueError: pass # Ignore if already removed somehow
            # 加入新联盟
            if best_task_id not in self.local_partition: self.local_partition[best_task_id] = []
            if self.id not in self.local_partition[best_task_id]: self.local_partition[best_task_id].append(self.id)

            # 更新内部状态
            self.current_coalition_id = best_task_id
            self.partition_version += 1 # 增加版本号
            self.partition_timestamp = random.random() # 生成新时间戳
            changed = True
            self.needs_update = True # 标记状态已改变

        return changed

    def send_message(self, all_spacecrafts, communication_counts):
        """ 向邻居发送本地联盟视图，并更新通信计数。 """
        # Only send if partition exists and is not empty
        if not self.local_partition: return

        try: partition_copy = copy.deepcopy(self.local_partition)
        except Exception as e: print(f"Error deep copying partition for SC {self.id}: {e}"); return

        message = {
            'sender_id': self.id, 'partition': partition_copy,
            'version': self.partition_version, 'timestamp': self.partition_timestamp
        }
        my_neighbors = list(self.neighbors) # Copy neighbors list before iterating
        for neighbor_id in my_neighbors:
            if neighbor_id in all_spacecrafts:
                # Append message to neighbor's queue
                try: all_spacecrafts[neighbor_id].message_queue.append(message)
                except AttributeError: print(f"Warning: Neighbor {neighbor_id} has no message_queue?"); continue

                # 更新通信计数: self.id -> neighbor_id
                try: communication_counts[self.id][neighbor_id] += 1
                except Exception as e: print(f"Error updating comm counts: {e}")

    def process_messages(self):
        """ 处理接收到的消息，根据共识规则更新本地联盟视图。 """
        if not self.message_queue: return False # 没有消息

        # 将自己的信息加入比较集合
        try: own_partition_copy = copy.deepcopy(self.local_partition)
        except Exception as e: print(f"Error copying own partition SC {self.id}: {e}"); own_partition_copy = {}

        own_info = {
            'sender_id': self.id, 'partition': own_partition_copy,
            'version': self.partition_version, 'timestamp': self.partition_timestamp
        }
        # 合并收到的消息（使用深拷贝）
        received_info = [own_info]
        for msg in self.message_queue:
             try: received_info.append(copy.deepcopy(msg))
             except Exception as e: print(f"Error copying received msg: {e}")

        self.message_queue = [] # 清空队列

        # 寻找主导信息（版本号最高，若相同则时间戳最大）
        dominant_info = own_info
        for msg in received_info[1:]: # Compare against others
            is_dominant = False
            # Basic validity check
            if not isinstance(msg, dict) or 'version' not in msg or 'timestamp' not in msg: continue
            if not isinstance(dominant_info, dict) or 'version' not in dominant_info or 'timestamp' not in dominant_info: continue

            try:
                if msg['version'] > dominant_info['version']: is_dominant = True
                elif msg['version'] == dominant_info['version'] and msg['timestamp'] > dominant_info['timestamp']: is_dominant = True
            except TypeError: continue # Skip if version/timestamp comparison fails

            if is_dominant: dominant_info = msg

        changed_by_message = False
        # 如果主导信息不是自己的，则采纳它
        if dominant_info['sender_id'] != self.id:
            # Check if dominant partition is valid
            if isinstance(dominant_info.get('partition'), dict):
                 self.local_partition = dominant_info['partition'] # Adopt the partition view
            self.partition_version = dominant_info['version'] # Adopt version
            self.partition_timestamp = dominant_info['timestamp'] # Adopt timestamp

            # 更新自己的当前联盟 ID
            found_self = False
            if isinstance(self.local_partition, dict): # Check if it's a dict before iterating
                for task_id, members in self.local_partition.items():
                    # Ensure members is iterable
                    if isinstance(members, list) and self.id in members:
                        self.current_coalition_id = task_id
                        found_self = True; break
            if not found_self: # 未找到则归入虚拟联盟 0
                self.current_coalition_id = 0
                if 0 not in self.local_partition or not isinstance(self.local_partition.get(0), list):
                     self.local_partition[0] = []
                if self.id not in self.local_partition[0]:
                     self.local_partition[0].append(self.id)

            changed_by_message = True
            self.needs_update = True # 标记状态已改变

        return changed_by_message

    def take_observation(self, task):
        """ 模拟对任务类型进行一次观测。 """
        if task is None or self.num_types <= 0: return None # Cannot observe null task or if no types defined

        true_type = task.true_type
        # 以 P(正确) = positive_observation_prob 的概率观测到真实类型
        if random.random() < self.positive_observation_prob:
            return true_type
        else:
            # 以 P(错误) = 1 - P(正确) 的概率观测到错误类型
            possible_false_types = [k for k in range(self.num_types) if k != true_type]
            if not possible_false_types: return true_type # 如果只有一种类型，则无法观测错误
            else: return random.choice(possible_false_types)

    def update_belief_from_observations(self, aggregated_observation_counts):
        """ 使用聚合的观测计数更新本地信念（狄利克雷更新）。 """
        # 检查输入维度是否匹配
        if (not isinstance(aggregated_observation_counts, np.ndarray) or
            aggregated_observation_counts.shape != self.observation_counts.shape):
             print(f"Warning: Aggregated observation shape/type mismatch for SC {self.id}. Skipping belief update.")
             # Still record the previous belief state? Or skip recording too? Let's skip.
             # self.record_belief_state()
             return

        # 累加本轮观测计数
        self.observation_counts += aggregated_observation_counts

        # 更新每个真实任务的信念 (j=1 to num_tasks)
        for j in range(1, self.num_tasks + 1):
             if j >= self.observation_counts.shape[0]: continue # 检查边界

             # 狄利克雷分布的参数 alpha = 累计观测计数 (伪计数已在初始化时加入)
             alpha_j = self.observation_counts[j, :]
             sum_alpha_j = np.sum(alpha_j)

             # 计算新信念（狄利克雷分布的期望值）
             if sum_alpha_j > 1e-9: # 避免除零
                 if j < self.local_belief.shape[0]: # 再次检查边界
                      self.local_belief[j, :] = alpha_j / sum_alpha_j
             # else: # 如果 alpha 和为 0 (理论上不应发生)，则保持原信念或恢复均匀分布
             #     if self.num_types > 0 and j < self.local_belief.shape[0]:
             #          self.local_belief[j, :] = 1.0 / self.num_types

        # 记录更新后的信念状态
        self.record_belief_state()

    def record_control_input(self, time_s, control_vector):
        """ 记录给定时间的控制输入向量。 """
        if control_vector is not None:
             try: self.control_history.append((time_s, control_vector.copy())) # 存储副本
             except AttributeError: print(f"Warning: Failed to copy control vector for SC {self.id}")

    def record_belief_state(self):
         """ 记录当前所有任务的信念状态。 """
         if self.num_types <= 0: return # No types, nothing to record

         for task_id in range(1, self.num_tasks + 1):
              if task_id < self.local_belief.shape[0]: # 检查边界
                   try: self.belief_history[task_id].append(self.local_belief[task_id, :].copy()) # 存储副本
                   except IndexError: print(f"Warning: IndexError accessing local_belief for Task {task_id} in SC {self.id}")

# --- Simulation Setup ---
def setup_simulation(num_spacecrafts, num_tasks, num_types, scenario='Fig4'):
    """ 设置模拟的初始状态。 """
    spacecrafts = {}
    tasks = {}
    initial_positions = {'robots': {}, 'tasks': {}} # 使用 'robots' 键以兼容旧绘图代码
    initial_neighbors = {} # 存储初始邻居关系用于绘图

    print(f"Setting up Simulation for Scenario: {scenario}...")
    # 根据场景确定初始信念类型
    belief_cfg = 'arbitrary' if scenario == 'Fig5' else 'uniform'
    print(f"  - Initial Beliefs: {'Arbitrary' if belief_cfg == 'arbitrary' else 'Uniform'}")

    # --- 任务设置 ---
    print(f"  - Tasks: {num_tasks}")
    if num_tasks == 0: print("Warning: 0 tasks specified.")
    # Fig2 使用固定位置，其他场景随机
    if scenario == 'Fig2' and num_tasks >= 6 and num_spacecrafts >=10 : # 确保实体数量足够
        task_positions_fig2 = {1: (75, 75), 2: (15, 30), 3: (70, 30), 4: (45, 20), 5: (60, 45), 6: (85, 15)}
        true_task_types = {1: 2, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1} # Fig2 任务类型示例
        for j in range(1, num_tasks + 1):
             pos = task_positions_fig2.get(j, (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE))) # 若超出则随机
             # 处理 num_types 为 0 的情况
             true_type = true_task_types.get(j, random.randrange(max(1,num_types))) if num_types > 0 else 0
             tasks[j] = Task(j, pos, true_type)
             initial_positions['tasks'][j] = pos
    else: # 其他场景或实体不足
        # 确保 TRUE_TASK_TYPES_FIG 列表足够长，不足则用随机类型补充
        default_type = random.randrange(max(1, num_types)) if num_types > 0 else 0
        true_types_list = (TRUE_TASK_TYPES_FIG + [default_type] * num_tasks)[:num_tasks]
        for j in range(1, num_tasks + 1):
            pos = (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE))
            true_type = true_types_list[j-1] if num_types > 0 else 0
            tasks[j] = Task(j, pos, true_type)
            initial_positions['tasks'][j] = pos

    # --- 航天器设置 ---
    print(f"  - Spacecraft: {num_spacecrafts}")
    if num_spacecrafts == 0: print("Warning: 0 spacecraft specified.")
    # 默认参数
    DEFAULT_INITIAL_MASS = 1000 # kg
    DEFAULT_DRY_MASS = 200    # kg
    DEFAULT_ISP = 300         # seconds
    # 初始划分：所有航天器都在虚拟联盟 0 中
    initial_partition = {j: [] for j in range(num_tasks + 1)}
    initial_partition[0] = list(range(num_spacecrafts))

    if scenario == 'Fig2' and num_spacecrafts >= 10:
        robot_positions_fig2 = {0: (10, 20), 1: (10, 60), 2: (30, 40), 3: (40, 70), 4: (55, 55), 5: (50, 90), 6: (65, 25), 7: (60, 80), 8: (80, 60), 9: (95, 40)}
        for i in range(num_spacecrafts):
            pos = robot_positions_fig2.get(i, (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)))
            initial_positions['robots'][i] = pos
            obs_prob = random.uniform(0.9, 1.0) # 随机观测概率
            spacecrafts[i] = Spacecraft(i, pos, num_tasks, num_types, DEFAULT_INITIAL_MASS, DEFAULT_DRY_MASS, DEFAULT_ISP, obs_prob, belief_cfg)
            spacecrafts[i].local_partition = copy.deepcopy(initial_partition) # 设置初始本地视图
            spacecrafts[i].current_coalition_id = 0 # 明确初始联盟为 0
    else: # 其他场景随机位置
        for i in range(num_spacecrafts):
            pos = (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE))
            initial_positions['robots'][i] = pos
            obs_prob = random.uniform(0.9, 1.0)
            spacecrafts[i] = Spacecraft(i, pos, num_tasks, num_types, DEFAULT_INITIAL_MASS, DEFAULT_DRY_MASS, DEFAULT_ISP, obs_prob, belief_cfg)
            spacecrafts[i].local_partition = copy.deepcopy(initial_partition)
            spacecrafts[i].current_coalition_id = 0

    # --- 邻居设置 ---
    print("  - Setting up Neighbors...")
    if scenario == 'Fig2' and num_spacecrafts >= 10:
        # Fig2 使用固定拓扑
        for i in spacecrafts:
            spacecrafts[i].neighbors = get_neighbors_fig2_topology(i, num_spacecrafts)
            initial_neighbors[i] = list(spacecrafts[i].neighbors) # 存储用于绘图
    else:
        # 其他场景使用通信范围确定邻居
        for i in spacecrafts:
            spacecrafts[i].neighbors = [] # 重置邻居列表
            pos_i = spacecrafts[i].position
            if pos_i is None: continue # 跳过无位置的航天器
            for j in spacecrafts:
                if i == j: continue # 不与自身通信
                pos_j = spacecrafts[j].position
                if pos_j is None: continue
                dist = calculate_distance(pos_i, pos_j)
                if dist <= COMM_RANGE: # 若在通信范围内
                    spacecrafts[i].neighbors.append(j)
            initial_neighbors[i] = list(spacecrafts[i].neighbors) # 存储初始邻居

        # 确保双向连接性和基本连通性（对共识很重要）
        all_ids = list(spacecrafts.keys())
        for i in all_ids:
            # 确保双向性
            for neighbor_id in list(spacecrafts[i].neighbors): # 遍历副本以允许修改
                if neighbor_id in spacecrafts:
                    # 如果邻居存在但不知道这个连接，添加反向连接
                    if i not in spacecrafts[neighbor_id].neighbors:
                        spacecrafts[neighbor_id].neighbors.append(i)
                else: # 如果邻居 ID 无效（可能已被移除），则从邻居列表中移除
                     if neighbor_id in spacecrafts[i].neighbors:
                          try: spacecrafts[i].neighbors.remove(neighbor_id)
                          except ValueError: pass # Ignore if not found

            # 处理孤立节点（如果存在多个航天器）
            if not spacecrafts[i].neighbors and len(spacecrafts) > 1:
                # 连接到 ID 0（如果自身不是 0）或 ID 1（如果自身是 0 且 1 存在）
                fallback_id = -1
                if i != 0 and 0 in spacecrafts: fallback_id = 0
                elif i == 0 and 1 in spacecrafts: fallback_id = 1

                if fallback_id != -1:
                     if fallback_id not in spacecrafts[i].neighbors: spacecrafts[i].neighbors.append(fallback_id)
                     if i not in spacecrafts[fallback_id].neighbors: spacecrafts[fallback_id].neighbors.append(i)
                     # print(f"    Fallback connection applied: {i} <-> {fallback_id}")
                     # 更新初始邻居信息用于绘图
                     initial_neighbors[i] = list(spacecrafts[i].neighbors)
                     if fallback_id in initial_neighbors: initial_neighbors[fallback_id] = list(spacecrafts[fallback_id].neighbors)

    print("Setup Complete.")
    return spacecrafts, tasks, initial_positions, initial_neighbors

# --- Plotting Functions ---

def plot_fig2(initial_positions, initial_neighbors, final_partition, tasks):
    """ 绘制初始状态/拓扑和最终联盟结构 (Fig 2 风格)。 """
    print("Generating Figure 2 plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1, ax2 = axes
    padding = 5
    ax1.set_xlim(-padding, AREA_SIZE + padding); ax1.set_ylim(-padding, AREA_SIZE + padding)
    ax2.set_xlim(-padding, AREA_SIZE + padding); ax2.set_ylim(-padding, AREA_SIZE + padding)
    ax1.set_xlabel('x-axis (m)'); ax1.set_ylabel('y-axis (m)')
    ax2.set_xlabel('x-axis (m)'); ax2.set_ylabel('y-axis (m)')
    ax1.set_title('Fig 2a: Initial State & Communication Topology')
    ax2.set_title('Fig 2b: Final Coalition Formation Result')
    ax1.set_aspect('equal', adjustable='box'); ax2.set_aspect('equal', adjustable='box')

    # 绘制任务 (面板 a & b)
    plotted_task_labels = set()
    for task_id, task in tasks.items():
        pos = task.position; color = FIG2_TASK_COLORS.get(task_id, 'gray')
        label_to_add = None
        if task_id not in plotted_task_labels: label_to_add=f'Task {task_id}'; plotted_task_labels.add(task_id)
        ax1.plot(pos[0], pos[1], '*', markersize=12, color=color, label=label_to_add)
        ax1.text(pos[0] + 1, pos[1] + 1, f'$t_{{{task_id}}}$', fontsize=9)
        ax2.plot(pos[0], pos[1], '*', markersize=12, color=color, label=label_to_add)
        ax2.text(pos[0] + 1, pos[1] + 1, f'$t_{{{task_id}}}$', fontsize=9)

    # 绘制初始航天器和通信链路 (面板 a)
    robot_pos_map = initial_positions.get('robots', {})
    plotted_sc_label = False
    for robot_id, pos in robot_pos_map.items():
        label_to_add = None
        if not plotted_sc_label: label_to_add='SC (Initial)'; plotted_sc_label = True
        ax1.plot(pos[0], pos[1], 'o', markersize=8, color='purple', alpha=0.7, label=label_to_add)
        ax1.text(pos[0] + 1, pos[1] + 1, f'$r_{{{robot_id}}}$', fontsize=9)

    plotted_links = set(); plotted_comm_label = False
    for robot_id, neighbors in initial_neighbors.items():
        pos1 = robot_pos_map.get(robot_id)
        if pos1 is None: continue
        for neighbor_id in neighbors:
            if neighbor_id == robot_id: continue
            link = tuple(sorted((robot_id, neighbor_id)))
            if link in plotted_links: continue
            pos2 = robot_pos_map.get(neighbor_id)
            if pos2 is None: continue
            label_to_add = None
            if not plotted_comm_label: label_to_add='Comm Link'; plotted_comm_label=True
            ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], '--', color='blue', linewidth=0.8, alpha=0.6, label=label_to_add)
            plotted_links.add(link)

    # 绘制最终联盟分配 (面板 b)
    safe_final_partition = {}
    if final_partition:
         try: safe_final_partition = {int(k): [int(m) for m in v] for k, v in final_partition.items()}
         except (ValueError, TypeError) as e: print(f"Warning: Could not convert final_partition for plotting: {e}")
    coalition_colors = {tid: FIG2_TASK_COLORS.get(tid, TASK_COLORS[tid % len(TASK_COLORS)]) for tid in tasks}
    coalition_colors[0] = 'lightgrey'
    plotted_final_robots = set()
    plotted_coalition_labels = set() # Track labels plotted for legend
    for task_id, members in safe_final_partition.items():
        coalition_color = coalition_colors.get(task_id, 'cyan')
        label_to_add = None
        if task_id not in plotted_coalition_labels:
             label_to_add = f'Coalition {task_id}'
             plotted_coalition_labels.add(task_id)

        for robot_id in members:
            pos = robot_pos_map.get(robot_id)
            if pos is None: continue
            # Only add label for the first robot plotted in this coalition
            current_label = label_to_add if robot_id not in plotted_final_robots else ""
            ax2.plot(pos[0], pos[1], 'o', markersize=8, color=coalition_color, markeredgecolor='black', linewidth=0.5, label=current_label)
            ax2.text(pos[0] + 1, pos[1] + 1, f'$r_{{{robot_id}}}$', fontsize=9)
            plotted_final_robots.add(robot_id)


    ax1.grid(True, linestyle=':', alpha=0.6); ax2.grid(True, linestyle=':', alpha=0.6)
    # ax1.legend(fontsize=8, loc='best') # Optional legends
    # ax2.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig('fig2_replication.png', dpi=150)
    print("Figure 2 plot saved as fig2_replication.png")

def plot_fig3(global_utility_history):
    """ 绘制全局效用随游戏回合的收敛情况。 """
    print("Generating Figure 3 plot (Global Utility)...")
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_indices = [i for i, util in enumerate(global_utility_history) if util is not None and np.isfinite(util)]
    if not valid_indices:
        print("No valid global utility data to plot for Fig 3.")
        plt.close(fig); return

    rounds_axis = [i for i in valid_indices] # 回合索引 (0, 1, 2...)
    valid_utility = [global_utility_history[i] for i in valid_indices]

    ax.plot(rounds_axis, valid_utility, marker='s', markersize=4, linestyle='-', color='blue')
    ax.set_title('Fig 3 Style: Global Utility Convergence')
    ax.set_xlabel('Index of game round (0=initial)')
    ax.set_ylabel('Global Utility')
    ax.grid(True, linestyle=':', alpha=0.6)
    if valid_utility: # 添加最终值注释
        final_round = rounds_axis[-1]; final_util = valid_utility[-1]
        ax.annotate(f'Final: {final_util:.2f}', xy=(final_round, final_util),
                    xytext=(final_round - max(1, final_round*0.1), final_util + max(1, abs(final_util*0.05))), fontsize=9)
    plt.tight_layout()
    plt.savefig('fig3_replication_global_utility.png', dpi=150)
    print("Figure 3 (Global Utility) plot saved as fig3_replication_global_utility.png")

def plot_expected_revenue_evolution_sc0(spacecrafts, tasks, title_prefix="Fig4/5"):
    """ 绘制航天器 0 视角下预期任务收益的演化。 """
    print(f"Generating {title_prefix} plots (Expected Revenue SC 0)...")
    if 0 not in spacecrafts: print("Spacecraft 0 not found for revenue evolution plotting."); return
    sc0 = spacecrafts[0]; belief_history_sc0 = sc0.belief_history
    num_tasks_to_plot = len(tasks)
    if num_tasks_to_plot == 0: print("No tasks to plot revenue for."); return

    num_rounds_plotted = 0
    if belief_history_sc0: num_rounds_plotted = len(next(iter(belief_history_sc0.values()), []))
    if num_rounds_plotted <= 1: print(f"Not enough history data ({num_rounds_plotted} points) for revenue plot."); return

    rounds_axis = range(num_rounds_plotted) # X轴: 0, 1, ..., N_rounds
    ncols = min(3, num_tasks_to_plot); nrows = math.ceil(num_tasks_to_plot / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, squeeze=False)
    axes = axes.flatten()

    expected_revenue_history = defaultdict(list)
    for task_id in range(1, num_tasks_to_plot + 1):
        if task_id in belief_history_sc0:
            task_obj = tasks.get(task_id)
            if task_obj and task_id in belief_history_sc0: # Ensure task and history exist
                 for belief_vector in belief_history_sc0[task_id]:
                     expected_rev = 0.0
                     if belief_vector is not None and len(belief_vector) == NUM_TASK_TYPES: # Check belief validity
                          for k in range(NUM_TASK_TYPES): expected_rev += belief_vector[k] * task_obj.get_revenue(k)
                     else: expected_rev = np.nan # Mark as invalid if belief is bad
                     expected_revenue_history[task_id].append(expected_rev)
            else: expected_revenue_history[task_id] = [np.nan] * num_rounds_plotted # No task object or history
        else: expected_revenue_history[task_id] = [np.nan] * num_rounds_plotted

    actual_revenues = {j: tasks[j].get_actual_revenue() for j in range(1, num_tasks_to_plot + 1) if j in tasks}
    for task_id in range(1, num_tasks_to_plot + 1):
        ax_idx = task_id - 1
        if ax_idx >= len(axes): break
        ax = axes[ax_idx]; history = expected_revenue_history.get(task_id, [])
        valid_indices = [i for i, rev in enumerate(history) if rev is not None and np.isfinite(rev)]
        if not valid_indices: ax.set_title(f'Task $t_{{{task_id}}}$: No Data'); continue

        valid_rounds = [rounds_axis[i] for i in valid_indices]; valid_history = [history[i] for i in valid_indices]
        ax.plot(valid_rounds, valid_history, marker='.', markersize=4, linestyle='-', label="Exp. Rev (SC 0)")
        actual_rev = actual_revenues.get(task_id, "N/A")
        if actual_rev != "N/A": ax.axhline(y=actual_rev, color='r', linestyle='--', linewidth=1.5, label=f'Actual ({actual_rev:.0f})')
        ax.set_title(f'Task $t_{{{task_id}}}$ (True Rev: {actual_rev})')
        if ax_idx // ncols == nrows - 1: ax.set_xlabel('Index of game round (0=initial)')
        if ax_idx % ncols == 0: ax.set_ylabel('Expected Task Revenue')
        ax.grid(True, linestyle=':', alpha=0.7); ax.legend(fontsize=8)
        min_rev = min(TASK_REVENUE.values()) - 50; max_rev = max(TASK_REVENUE.values()) + 50
        ax.set_ylim(min_rev, max_rev)
        ax.set_xticks(np.arange(0, num_rounds_plotted, max(1, num_rounds_plotted // 5)))

    for i in range(num_tasks_to_plot, nrows * ncols):
         if i < len(axes): fig.delaxes(axes[i])
    plt.suptitle(f'{title_prefix}: Evolution of Expected Task Revenue (Spacecraft 0 View)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{title_prefix}_expected_revenue_sc0.png', dpi=150)
    print(f"{title_prefix} (Expected Revenue SC 0) plot saved as {title_prefix}_expected_revenue_sc0.png")

def plot_trajectories(spacecrafts, tasks, scenario):
    """ 绘制相对轨迹 (XY 平面投影)。 """
    print("Generating Trajectory Plot...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'Spacecraft Relative Trajectories (XY Projection) - Scenario: {scenario}')
    ax.set_xlabel('Relative X (m)'); ax.set_ylabel('Relative Y (m)')
    ax.grid(True, linestyle=':', alpha=0.6); ax.set_aspect('equal', adjustable='box')

    task_positions_x = []; task_positions_y = []
    plotted_task_labels=set()
    for task_id, task in tasks.items():
        if task.relative_state is not None and len(task.relative_state) >= 2:
            target_pos = task.relative_state[:2]
            task_positions_x.append(target_pos[0]); task_positions_y.append(target_pos[1])
            label_to_add=None
            if task_id not in plotted_task_labels: label_to_add=f'Target {task_id}'; plotted_task_labels.add(task_id)
            ax.plot(target_pos[0], target_pos[1], 'k*', markersize=10, label=label_to_add)
            ax.text(target_pos[0]+1, target_pos[1]+1, f'T{task_id}', fontsize=9)

    colors = plt.cm.viridis(np.linspace(0, 1, len(spacecrafts)))
    all_traj_x = []; all_traj_y = []
    for idx, (sc_id, craft) in enumerate(spacecrafts.items()):
        if craft.trajectory and isinstance(craft.trajectory, list) and all(isinstance(p, (list, np.ndarray)) and len(p) >= 2 for p in craft.trajectory):
             try:
                 traj = np.array(craft.trajectory)
                 if traj.ndim == 2 and traj.shape[1] >= 2:
                     traj_x = traj[:, 0]; traj_y = traj[:, 1]
                     all_traj_x.append(traj_x); all_traj_y.append(traj_y)
                     ax.plot(traj_x, traj_y, marker='.', markersize=1, linestyle='-', color=colors[idx], label=f'SC {sc_id}')
                     ax.plot(traj_x[0], traj_y[0], 'go', markersize=6) # Start
                     ax.plot(traj_x[-1], traj_y[-1], 'rs', markersize=6) # End
                 elif traj.ndim == 1 and len(traj) >= 2: # Single point case
                      ax.plot(traj[0], traj[1], 'go', markersize=6, color=colors[idx], label=f'SC {sc_id} (Start Only)')
                      all_traj_x.append(np.array([traj[0]])); all_traj_y.append(np.array([traj[1]]))
                 # else: print(f"Warning: Trajectory for SC {sc_id} has unexpected shape: {traj.shape}")
             except Exception as e: print(f"Error processing trajectory for SC {sc_id}: {e}")
        elif craft.trajectory and isinstance(craft.trajectory[0], (list, np.ndarray)) and len(craft.trajectory[0]) >= 2: # Only initial point
             start_pos = craft.trajectory[0]
             ax.plot(start_pos[0], start_pos[1], 'go', markersize=6, color=colors[idx], label=f'SC {sc_id} (Start Only)')
             all_traj_x.append(np.array([start_pos[0]])); all_traj_y.append(np.array([start_pos[1]]))

    ax.legend(fontsize=8, loc='best')
    try: # Dynamically adjust plot limits
        valid_traj_x = [t for t in all_traj_x if t.size > 0]
        valid_traj_y = [t for t in all_traj_y if t.size > 0]
        if not valid_traj_x and not task_positions_x: raise ValueError("No points")
        all_x_points = np.concatenate(valid_traj_x if valid_traj_x else [np.array([])])
        all_x = np.concatenate([all_x_points, np.array(task_positions_x)])
        if not valid_traj_y and not task_positions_y: raise ValueError("No points")
        all_y_points = np.concatenate(valid_traj_y if valid_traj_y else [np.array([])])
        all_y = np.concatenate([all_y_points, np.array(task_positions_y)])
        if all_x.size > 0 and all_y.size > 0:
            x_min, x_max = np.min(all_x), np.max(all_x); y_min, y_max = np.min(all_y), np.max(all_y)
            x_range = max(x_max - x_min, 10); y_range = max(y_max - y_min, 10)
            padding_x = x_range * 0.1; padding_y = y_range * 0.1
            ax.set_xlim(x_min - padding_x, x_max + padding_x)
            ax.set_ylim(y_min - padding_y, y_max + padding_y)
        else: print("Warning: No valid points for trajectory plot limits.")
    except ValueError as e:
         print(f"Error during plot limit calculation: {e}. Using default limits.")
         ax.set_xlim(-AREA_SIZE*0.1, AREA_SIZE*1.1); ax.set_ylim(-AREA_SIZE*0.1, AREA_SIZE*1.1)
    plt.tight_layout()
    plt.savefig(f'trajectory_plot_{scenario}.png', dpi=150)
    print(f"Trajectory plot saved as trajectory_plot_{scenario}.png")

def plot_belief_evolution(spacecrafts, tasks, title_prefix="Belief Evolution"):
    """ 绘制每个航天器对每个任务信念的演化。 """
    print(f"Generating {title_prefix} plots...")
    if not spacecrafts or not tasks: print("No spacecraft or tasks for belief evolution plot."); return

    num_sc = len(spacecrafts); num_tk = len(tasks); num_types = NUM_TASK_TYPES
    if num_types <= 0: print("No task types defined, cannot plot belief evolution."); return # Handle zero types

    first_sc_id = next(iter(spacecrafts)); first_sc_history = spacecrafts[first_sc_id].belief_history
    num_rounds_plotted = 0
    if first_sc_history: num_rounds_plotted = len(next(iter(first_sc_history.values()), []))
    if num_rounds_plotted <= 1: print("Not enough history data for belief evolution plot."); return
    plot_rounds_indices = range(num_rounds_plotted)

    for sc_id, craft in spacecrafts.items():
        belief_history = craft.belief_history
        if not belief_history: continue
        ncols = min(3, num_tk); nrows = math.ceil(num_tk / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True, squeeze=False)
        axes = axes.flatten()
        for task_idx, task_id in enumerate(tasks.keys()):
            ax = axes[task_idx]; task_obj = tasks[task_id]; true_type = task_obj.true_type
            if task_id not in belief_history or not belief_history[task_id]: ax.set_title(f'Task $t_{{{task_id}}}$: No Data'); continue

            history_array = np.array(belief_history[task_id])
            # Check shape after converting to array
            if history_array.ndim != 2 or history_array.shape[1] != num_types:
                 print(f"Warning: Incorrect belief history shape {history_array.shape} for SC {sc_id}, Task {task_id}. Expected (rounds, {num_types})")
                 ax.set_title(f'Task $t_{{{task_id}}}$: Invalid Data Shape')
                 continue

            for type_k in range(num_types):
                type_label = TASK_TYPE_MAP.get(type_k, f'Type {type_k}')
                ax.plot(plot_rounds_indices, history_array[:, type_k], marker='.', markersize=3, linestyle='-', label=f'{type_label}')

            ax.set_title(f'Task $t_{{{task_id}}}$ (True: {TASK_TYPE_MAP.get(true_type, true_type)})')
            ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5); ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=0.5)
            ax.set_ylim(-0.1, 1.1)
            if task_idx // ncols == nrows - 1 : ax.set_xlabel('Index of game round (0=initial)')
            if task_idx % ncols == 0: ax.set_ylabel('Belief Probability')
            ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize=8)
            ax.set_xticks(np.arange(0, num_rounds_plotted, max(1, num_rounds_plotted // 5)))
        for i in range(num_tk, nrows * ncols):
            if i < len(axes): fig.delaxes(axes[i])
        plt.suptitle(f'{title_prefix}: Spacecraft {sc_id}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{title_prefix}_SC{sc_id}.png', dpi=150)
        print(f"{title_prefix} plot saved for SC {sc_id}.")
        plt.close(fig) # 关闭图形，避免一次性显示过多

def plot_gantt_chart_time_based(assignment_history, round_start_times, spacecrafts, tasks, total_sim_time):
    """
    绘制基于物理时间的甘特图。
    条形表示航天器到达任务位置后的执行时间段。
    """
    print("Generating Time-Based Gantt Chart (Task Execution after Arrival)...")

    num_spacecrafts = len(spacecrafts)
    if not assignment_history or len(round_start_times) < 2 or num_spacecrafts == 0:
        print("Not enough data for time-based Gantt chart.")
        return

    fig, ax = plt.subplots(figsize=(14, max(4, num_spacecrafts * 0.5))) # 调整图形大小
    ax.set_title('Gantt Chart: Task Execution Time vs. Physical Time')
    ax.set_xlabel('Simulation Time (s)')
    ax.set_ylabel('Spacecraft ID')

    # --- Y 轴设置 ---
    y_ticks = np.arange(num_spacecrafts)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'SC {i}' for i in range(num_spacecrafts)])
    ax.invert_yaxis() # SC 0 在顶部
    ax.tick_params(axis='y', length=0) # 隐藏 Y 轴刻度线

    # --- X 轴设置 ---
    ax.set_xlim(0, total_sim_time)
    # 设置更精细的时间刻度
    major_ticks_interval = max(10.0, total_sim_time / 10.0) # 主刻度间隔
    minor_ticks_interval = max(5.0, major_ticks_interval / 2.0) # 次刻度间隔
    try: # Add try-except for locator issues with zero range
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_ticks_interval))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_ticks_interval))
    except ValueError: print("Warning: Could not set X-axis locators, time range might be zero.")
    ax.tick_params(axis='x', which='major', length=6)
    ax.tick_params(axis='x', which='minor', length=3)

    # --- 颜色和样式 ---
    unique_task_ids = set(tid for assignments in assignment_history for tid in assignments.values() if tid is not None and tid != 0)
    # 使用更柔和的颜色主题 (例如 Pastel1 或 Tableau)
    # colors = plt.cm.Pastel1(np.linspace(0, 1, max(1, len(unique_task_ids))))
    colors = plt.cm.get_cmap('tab10', max(10, len(unique_task_ids))) # Tableau10
    task_color_map_gantt = {tid: colors(i % colors.N) for i, tid in enumerate(sorted(list(unique_task_ids)))}
    task_color_map_gantt[0] = '#E0E0E0' # 浅灰色表示空闲/虚拟任务
    travel_color = '#B0BEC5' # 定义一个表示“前往中”的颜色 (可选)

    # --- 网格线 ---
    ax.grid(True, axis='x', linestyle=':', color='grey', alpha=0.6, which='major')
    ax.grid(True, axis='x', linestyle=':', color='lightgrey', alpha=0.4, which='minor')
    ax.grid(False, axis='y') # 不显示水平网格线

    # --- 绘制条形 ---
    num_intervals = len(assignment_history)
    num_times = len(round_start_times)
    if num_times < num_intervals + 1: # 确保时间列表足够长
        print(f"Warning: Mismatch history({num_intervals}) vs times({num_times}). Gantt incomplete.")
        while len(round_start_times) < num_intervals + 1:
            round_start_times.append(round_start_times[-1] + GAME_ROUND_DURATION_S)

    # 缓存到达时间计算结果
    arrival_cache = {} # {(round_idx, sc_id): arrival_time_or_None}

    for round_idx in range(num_intervals):
        assignments = assignment_history[round_idx]
        interval_start_time = round_start_times[round_idx]
        interval_end_time = round_start_times[round_idx + 1]

        for sc_id, task_id in assignments.items():
            if sc_id not in spacecrafts: continue # 跳过无效的 SC ID

            craft = spacecrafts[sc_id]
            exec_start_time = -1 # 初始化为无效
            exec_end_time = interval_end_time
            bar_color = task_color_map_gantt.get(task_id, 'white')
            label_text = f'T{task_id}' if task_id != 0 else "Idle"
            plot_bar = True # 默认绘制

            # --- 处理非零任务，计算到达时间 ---
            if task_id != 0:
                target_task = tasks.get(task_id)
                if not target_task or target_task.relative_state is None: continue # 任务或目标状态无效

                cache_key = (round_idx, sc_id, task_id) # Add task_id to key if assignment changes
                if cache_key in arrival_cache:
                    arrival_time = arrival_cache[cache_key]
                else:
                    arrival_time = None
                    target_pos = target_task.relative_state[:3] # Target position (3D)
                    # Find trajectory points within the interval [interval_start_time, interval_end_time)
                    traj_indices_in_interval = []
                    current_traj_time = 0.0
                    for idx, state in enumerate(craft.trajectory):
                        # Approximate time for this state point
                        state_time = idx * TS
                        if state_time >= interval_start_time and state_time < interval_end_time:
                             traj_indices_in_interval.append(idx)

                    # Search for arrival within this segment
                    for idx in traj_indices_in_interval:
                         state = craft.trajectory[idx]
                         if len(state) >= 3:
                              current_pos = np.array(state[:3])
                              dist = np.linalg.norm(current_pos - target_pos)
                              if dist < ARRIVAL_DISTANCE_THRESHOLD:
                                  arrival_time = idx * TS # Approximate arrival time
                                  break # Found first arrival in interval
                    arrival_cache[cache_key] = arrival_time # Cache the result (None if not found)

                if arrival_time is not None:
                    # Arrived within the interval (or maybe before)
                    exec_start_time = max(interval_start_time, arrival_time)
                    label_text = f'T{task_id}' # Label as executing task
                else:
                    # Did not arrive in this interval
                    # Option 1: Plot "Traveling" bar for the whole interval
                    bar_color = travel_color
                    label_text = f'To T{task_id}'
                    exec_start_time = interval_start_time # Bar starts at interval beginning
                    # Option 2: Don't plot anything for execution
                    # plot_bar = False
            else: # Task ID is 0 (Idle)
                exec_start_time = interval_start_time # Idle starts immediately
                label_text = "Idle"


            # --- 绘制条形 ---
            if plot_bar and exec_start_time >= 0: # Ensure valid start time
                duration = max(0, exec_end_time - exec_start_time)
                if duration > 1e-6: # Only draw non-zero duration bars
                    ax.barh(y=sc_id, width=duration, left=exec_start_time, height=0.5, # Smaller height
                            color=bar_color, edgecolor='grey', linewidth=0.5, alpha=0.9)

                    # Add text label inside the bar if wide enough
                    if duration > total_sim_time * 0.02:
                        try:
                            rgb_color = mcolors.to_rgb(bar_color)
                            text_color = 'white' if np.mean(rgb_color) < 0.5 else 'black'
                            ax.text(exec_start_time + duration / 2, sc_id, label_text,
                                    ha='center', va='center', fontsize=7, color=text_color,
                                    bbox=dict(boxstyle='round,pad=0.1', fc=bar_color, ec='none', alpha=0.7))
                        except ValueError: pass # Handle potential color conversion issues


    # --- 图例 ---
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, label=f'Task {tid} (Exec)')
                       for tid, color in task_color_map_gantt.items() if tid != 0]
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=task_color_map_gantt[0], label='Idle/Void'))
    if travel_color not in task_color_map_gantt.values(): # Add travel legend if used
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=travel_color, label='Traveling'))
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(legend_elements)), fontsize=8)

    plt.subplots_adjust(bottom=0.25) # Adjust bottom margin for legend
    plt.savefig('gantt_chart_time_based_arrival.png', dpi=150)
    print("Time-based Gantt chart (with arrival condition) saved as gantt_chart_time_based_arrival.png")
# --- Plotting Functions ---
# ... (其他绘图函数保持不变) ...

# ***** 新增/修改: 分轴绘制控制输入的函数 *****
def plot_control_input_components(spacecrafts, total_sim_time):
    """
    为每个航天器绘制其三轴控制输入 (Ux, Uy, Uz) 随时间变化的图。
    每个航天器一个子图。
    """
    print("Generating Control Input Components Plot...")
    if not MPC_AVAILABLE:
        print("MPC was not available, cannot plot control inputs.")
        return

    num_sc = len(spacecrafts)
    if num_sc == 0:
        print("No spacecraft data to plot control inputs.")
        return

    # 创建一个包含 N 个子图的 Figure，N = 航天器数量
    # 共用 X 轴 (时间)
    fig, axes = plt.subplots(num_sc, 1, figsize=(12, 4 * num_sc), sharex=True, squeeze=False)
    # squeeze=False 确保即使 num_sc=1，axes 也是一个 2D 数组 (1x1)

    fig.suptitle('Control Input Components vs. Time', fontsize=16)

    max_abs_control = 0 # 用于统一 Y 轴范围

    sc_ids = sorted(spacecrafts.keys()) # 确保按 ID 顺序绘图

    for idx, sc_id in enumerate(sc_ids):
        craft = spacecrafts[sc_id]
        ax = axes[idx, 0] # 获取当前航天器的子图坐标轴
        control_hist = craft.control_history

        if control_hist:
            # 过滤掉 None 值和非有限值
            valid_hist = [(t, u) for t, u in control_hist if u is not None and u.shape == (3,) and np.all(np.isfinite(u))]

            if valid_hist:
                times = [t for t, u in valid_hist]
                ux = [u[0] for t, u in valid_hist]
                uy = [u[1] for t, u in valid_hist]
                uz = [u[2] for t, u in valid_hist]

                # 绘制三轴控制量
                ax.plot(times, ux, marker='.', markersize=2, linestyle='-', label='Ux')
                ax.plot(times, uy, marker='.', markersize=2, linestyle='-', label='Uy')
                ax.plot(times, uz, marker='.', markersize=2, linestyle='-', label='Uz')

                # 更新最大绝对控制值，用于设定 Y 轴范围
                current_max = np.max(np.abs(np.array([ux, uy, uz]))) if times else 0
                max_abs_control = max(max_abs_control, current_max)

                ax.legend(fontsize=8, loc='upper right')
            else:
                 ax.text(0.5, 0.5, 'No Valid Control Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        else: # 没有控制历史记录
             ax.text(0.5, 0.5, 'No Control History', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        # 设置子图标题和标签
        ax.set_title(f'Spacecraft {sc_id}')
        ax.set_ylabel('Control Input (m/s^2)')
        ax.grid(True, linestyle=':', alpha=0.7)

    # 设置统一的 X 轴标签 (仅最下方子图)
    axes[-1, 0].set_xlabel('Simulation Time (s)')

    # 设置统一的 Y 轴范围，稍大于最大绝对值或 UMAX_COMPONENT
    y_limit = max(max_abs_control * 1.1, UMAX_COMPONENT * 1.1, 0.01) if MPC_AVAILABLE else 0.1
    for ax_row in axes:
        ax_row[0].set_ylim(-y_limit, y_limit)
        # 可以添加一条 y=0 的水平线
        ax_row[0].axhline(0, color='black', linewidth=0.5, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，防止标题重叠
    plt.savefig('control_input_components.png', dpi=150)
    print("Control input components plot saved as control_input_components.png")


def plot_communication_frequency(comm_counts, spacecrafts):
    """ 将航天器之间的通信频率绘制为热力图。 """
    print("Generating Communication Frequency Heatmap...")
    num_sc = len(spacecrafts)
    if num_sc == 0 or not comm_counts: print("No spacecraft or communication data to plot."); return

    comm_matrix = np.zeros((num_sc, num_sc))
    sc_ids = sorted(spacecrafts.keys()) # 排序的 ID 列表
    id_to_idx = {id: idx for idx, id in enumerate(sc_ids)} # ID 到矩阵索引的映射

    for sender_id, receivers in comm_counts.items():
        if sender_id not in id_to_idx: continue
        sender_idx = id_to_idx[sender_id]
        for receiver_id, count in receivers.items():
            if receiver_id not in id_to_idx: continue
            receiver_idx = id_to_idx[receiver_id]
            comm_matrix[sender_idx, receiver_idx] = count

    fig, ax = plt.subplots(figsize=(max(6, num_sc*0.8), max(5, num_sc*0.7))) # 调整大小
    im = ax.imshow(comm_matrix, cmap="viridis", interpolation='nearest')
    ax.set_xticks(np.arange(num_sc)); ax.set_yticks(np.arange(num_sc))
    ax.set_xticklabels([f'SC {id}' for id in sc_ids]); ax.set_yticklabels([f'SC {id}' for id in sc_ids])
    ax.set_xlabel("Receiver Spacecraft ID"); ax.set_ylabel("Sender Spacecraft ID")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    max_count = np.max(comm_matrix) if np.max(comm_matrix) > 0 else 1.0 # 避免除零
    for i in range(num_sc):
        for j in range(num_sc):
            val = comm_matrix[i, j]
            # 根据背景亮度选择文本颜色
            text_color = "white" if val / max_count < 0.5 else "black" # 调整阈值
            ax.text(j, i, f"{int(val)}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8) # 调整颜色条大小
    cbar.ax.set_ylabel("Number of Messages Sent", rotation=-90, va="bottom")
    ax.set_title("Communication Frequency Between Spacecraft (Messages Sent)")
    fig.tight_layout()
    plt.savefig('communication_frequency_heatmap.png', dpi=150)
    print("Communication frequency heatmap saved as communication_frequency_heatmap.png")


# --- Main Simulation Loop ---
def run_simulation(scenario):
    # Setup
    spacecrafts, tasks, initial_positions, initial_neighbors = setup_simulation(
        NUM_SPACECRAFTS, NUM_TASKS, NUM_TASK_TYPES, scenario=scenario
    )
    if not spacecrafts: print("Error: No spacecraft initialized. Exiting."); return

    # History Tracking Initialization
    global_utility_history = []
    assignment_history = [] # Stores {sc_id: task_id} dict decided *after* each round's coalition formation
    last_stable_partition = None
    communication_counts = defaultdict(lambda: defaultdict(int))
    round_start_times = [0.0] # Record start time of round 0 (initial state)

    # --- Record Initial State (Round 0) ---
    initial_assignment = {i: craft.current_coalition_id for i, craft in spacecrafts.items()}
    assignment_history.append(initial_assignment) # Assignment active from time 0
    initial_utility = 0
    initial_partition_view = spacecrafts[0].local_partition if 0 in spacecrafts and spacecrafts[0].local_partition else {} # Initial view
    if initial_partition_view: # Ensure partition view exists
        for i, craft in spacecrafts.items():
            # Use the initial assignment for utility calculation
            initial_utility += craft.calculate_actual_utility(initial_assignment.get(i, 0), initial_partition_view, tasks)
    global_utility_history.append(initial_utility)

    # Simulation Time Control
    current_time_s = 0.0
    game_round = 0

    print("\nStarting Simulation with INNER/OUTER LOOP...")
    if MPC_AVAILABLE: print(f"--- MPC ACTIVE: Np={NP}, Nc={NC}, Ts={TS}s ---")
    else: print("--- MPC NOT AVAILABLE - Using simplified distance cost & zero control ---")

    # --- Outer Loop: Decision + Belief Update ---
    while current_time_s < MAX_SIM_TIME_S:
        game_round += 1 # Increment round counter (Starts at 1)
        outer_loop_start_time_wall = time.time()
        current_round_start_time_sim = current_time_s # Time at the beginning of this round
        round_start_times.append(current_round_start_time_sim) # Record start time for round 'game_round'

        print(f"\n--- Outer Loop (Game Round {game_round}) Start - Sim Time: {current_round_start_time_sim:.2f}s ---")

        # --- 1. Coalition Formation (Algorithm 1) ---
        print(f"  Running Coalition Formation...")
        coalition_start_time_wall = time.time()
        stable_iterations = 0; coalition_formation_iter = 0
        while True:
            coalition_formation_iter += 1
            if coalition_formation_iter > MAX_COALITION_ITER: print(f"    Warning: Coalition max iterations ({MAX_COALITION_ITER})."); break
            num_crafts_changed_greedy = 0; num_crafts_changed_consensus = 0
            for i in spacecrafts: spacecrafts[i].needs_update = False
            # a) Greedy Selection
            for i in spacecrafts:
                if spacecrafts[i].greedy_selection(tasks): num_crafts_changed_greedy += 1
            # b) Consensus
            for i in spacecrafts: spacecrafts[i].send_message(spacecrafts, communication_counts) # Pass counter
            for i in spacecrafts:
                if spacecrafts[i].process_messages(): num_crafts_changed_consensus += 1
            # c) Check Stability
            total_changed_this_iter = sum(1 for i in spacecrafts if spacecrafts[i].needs_update)
            if total_changed_this_iter == 0:
                stable_iterations += 1
                if stable_iterations >= STABLE_ITER_THRESHOLD: print(f"    Coalition Converged after {coalition_formation_iter} iterations."); break
            else: stable_iterations = 0
        print(f"  Coalition Formation took {time.time() - coalition_start_time_wall:.3f}s (wall time)")

        # Get the stable partition view (e.g., from SC 0)
        partition_view_source_id = -1
        if 0 in spacecrafts: partition_view_source_id = 0
        elif spacecrafts: partition_view_source_id = next(iter(spacecrafts))

        if partition_view_source_id != -1:
             stable_partition_view = copy.deepcopy(spacecrafts[partition_view_source_id].local_partition)
             try: last_stable_partition = {int(k): [int(m) for m in v] for k, v in stable_partition_view.items()}
             except Exception as e: print(f"Error converting partition view: {e}"); last_stable_partition = {}
        else: last_stable_partition = {} # Handle no spacecraft case

        # Update current_coalition_id for all SC based on consensus
        # This assignment will be active during the *following* inner loop
        current_assignment = {}
        for craft_id, craft in spacecrafts.items():
            found = False
            if last_stable_partition:
                for task_id, members in last_stable_partition.items():
                     # Ensure members is list before checking 'in'
                    if isinstance(members, list) and craft_id in members:
                        craft.current_coalition_id = task_id; current_assignment[craft_id] = task_id; found = True; break
            if not found: craft.current_coalition_id = 0; current_assignment[craft_id] = 0
        assignment_history.append(current_assignment) # Record assignment for start of round 'game_round'

        # --- 2. Inner Loop: Physical Simulation ---
        print(f"  Running Inner Physics Loop from {current_round_start_time_sim:.2f}s to {min(current_round_start_time_sim + GAME_ROUND_DURATION_S, MAX_SIM_TIME_S):.2f}s (step={TS}s)...")
        inner_loop_start_time_wall = time.time()
        num_inner_steps = 0
        inner_loop_end_time_sim = current_round_start_time_sim + GAME_ROUND_DURATION_S

        # Get the assignment that is active *during* this inner loop
        # This is the assignment decided at the end of the *previous* outer loop
        active_assignment = assignment_history[-2] if len(assignment_history) >= 2 else assignment_history[0]


        while current_time_s < inner_loop_end_time_sim and current_time_s < MAX_SIM_TIME_S:
            if TS <= 0: break
            # a) MPC Control Calculation
            control_inputs_this_step = {}
            if MPC_AVAILABLE:
                for craft_id, craft in spacecrafts.items():
                    # Use the assignment active during this interval
                    target_task_id = active_assignment.get(craft_id, 0)
                    control_vector = np.zeros(3)
                    if target_task_id != 0:
                        target_task = tasks.get(target_task_id)
                        if target_task and craft.Ad is not None and craft.Bd is not None:
                            target_ref_state = target_task.relative_state
                            if target_ref_state is None or target_ref_state.shape != (6,): target_ref_state = np.zeros(6)
                            ref_trajectory = np.tile(target_ref_state, (NP + 1, 1)).T
                            u0, _ = solve_mpc(craft.relative_state, ref_trajectory, craft.Ad, craft.Bd, NP, NC, Q, R_MPC, P, UMAX_COMPONENT)
                            if u0 is not None: control_vector = u0
                    control_inputs_this_step[craft_id] = control_vector
            else: # MPC not available
                for craft_id in spacecrafts: control_inputs_this_step[craft_id] = np.zeros(3)

            # b) State Update, Fuel, Record Control
            for craft_id, craft in spacecrafts.items():
                control_to_apply = control_inputs_this_step.get(craft_id, np.zeros(3))
                craft.update_physical_state(control_to_apply, TS)
                craft.record_control_input(current_time_s, control_to_apply) # Record control at current time
                delta_v_this_step = np.linalg.norm(control_to_apply) * TS if control_to_apply is not None else 0.0
                craft.consume_fuel(delta_v_this_step)

            current_time_s += TS # Advance simulation time
            num_inner_steps += 1
        print(f"    Inner Loop ({num_inner_steps} steps) took {time.time() - inner_loop_start_time_wall:.3f}s (wall time)")

        # --- 3. Observation Aggregation & Belief Update ---
        print(f"  Running Observation & Belief Update...")
        aggregated_observation_counts = np.zeros((NUM_TASKS + 1, NUM_TASK_TYPES))
        for i, craft in spacecrafts.items():
            # Observations are based on the task assigned during the inner loop just completed
            assigned_task_id = active_assignment.get(i, 0) # Use the assignment active during the inner loop
            if assigned_task_id != 0:
                task_for_observation = tasks.get(assigned_task_id)
                if task_for_observation:
                    for _ in range(OBSERVATIONS_PER_ROUND):
                        observed_type = craft.take_observation(task_for_observation)
                        if observed_type is not None and 0 <= observed_type < NUM_TASK_TYPES:
                            if assigned_task_id < aggregated_observation_counts.shape[0] and observed_type < aggregated_observation_counts.shape[1]:
                                 aggregated_observation_counts[assigned_task_id, observed_type] += 1
        # Update beliefs for all SC
        for i, craft in spacecrafts.items(): craft.update_belief_from_observations(aggregated_observation_counts)

        # --- 4. Record Global Utility ---
        current_global_utility = 0
        # Utility is calculated based on the partition *active* during the inner loop simulation
        partition_for_utility_calc = {}
        if len(assignment_history) >=2 : # Get the partition corresponding to the active assignment
             assign_view = assignment_history[-2] # The assignment active during the loop
             temp_part = defaultdict(list)
             for sc, tk in assign_view.items(): temp_part[tk].append(sc)
             partition_for_utility_calc = dict(temp_part)
        else: # Use initial partition for first round utility
             assign_view = assignment_history[0]
             temp_part = defaultdict(list)
             for sc, tk in assign_view.items(): temp_part[tk].append(sc)
             partition_for_utility_calc = dict(temp_part)


        if partition_for_utility_calc:
             for i, craft in spacecrafts.items():
                 assigned_task_id = active_assignment.get(i, 0) # Assignment active during inner loop
                 # Calculate utility based on state *after* inner loop, but using partition *before* coalition formation
                 current_global_utility += craft.calculate_actual_utility(assigned_task_id, partition_for_utility_calc, tasks)
        global_utility_history.append(current_global_utility)
        print(f"  Global Utility (End of Round {game_round}): {current_global_utility:.2f}")

        # --- End of Outer Loop ---
        print(f"--- Outer Loop {game_round} finished in {time.time() - outer_loop_start_time_wall:.3f} seconds (wall time) ---")


    # --- Post Simulation & Plotting ---
    final_round_count = game_round
    final_sim_time = current_time_s
    print(f"\nSimulation finished after {final_round_count} outer loops ({final_sim_time:.2f}s simulated time).")
    # Add final time to mark the end of the last interval for Gantt chart
    round_start_times.append(final_sim_time)

    # Use the last partition formed for Fig2 plotting
    final_partition_for_plotting = last_stable_partition if last_stable_partition else {}

    # --- Generate Plots ---
    # Check conditions for Fig2 plot
    if scenario == 'Fig2' and NUM_SPACECRAFTS >= 10 and NUM_TASKS >= 6:
        plot_fig2(initial_positions, initial_neighbors, final_partition_for_plotting, tasks)

    plot_fig3(global_utility_history)
    plot_expected_revenue_evolution_sc0(spacecrafts, tasks, title_prefix=f"{scenario}_Revenue")
    plot_trajectories(spacecrafts, tasks, scenario)
    plot_belief_evolution(spacecrafts, tasks, title_prefix=f"{scenario}_Beliefs")
    # Pass necessary objects to the time-based Gantt chart function
    plot_gantt_chart_time_based(assignment_history, round_start_times, spacecrafts, tasks, final_sim_time)
    # ***** 修改: 调用新的控制输入绘图函数 *****
    plot_control_input_components(spacecrafts, final_sim_time)
    plot_communication_frequency(communication_counts, spacecrafts)

    print("\nDisplaying plots...")
    # Ensure matplotlib backend allows display
    try: plt.show()
    except Exception as e: print(f"Could not display plots automatically: {e}")

# --- Run ---
if __name__ == "__main__":
    # Set random seeds for reproducibility (optional)
    # random.seed(42)
    # np.random.seed(42)

    print(f"--- Starting Coalition Formation with INNER/OUTER LOOP ---")
    print(f"--- Scenario: {CURRENT_SCENARIO}, N={NUM_SPACECRAFTS}, M={NUM_TASKS}, SimTime={MAX_SIM_TIME_S}s ---")
    run_simulation(scenario=CURRENT_SCENARIO)
    print("--- Simulation Run Complete ---")