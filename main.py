
import numpy as np
from matplotlib.animation import FuncAnimation
import os, sys, importlib, json, itertools
from datetime import datetime
from IPython.display import HTML, display
import matplotlib.image as mpimg

# Import the aif module
pwd = os.path.abspath('') + "/"
sys.path.insert(1, pwd + '/aif_catkin_ws/aif_gazebo/scripts/simulation_methods')
import aif_functions_isobeliefs_convergent as aif # Interchangeable with aif_functions_isobeliefs and aif_functions_isobeliefs_mcts
# import aif_catkin_ws.aif_gazebo.scripts.aif_functions_isobeliefs_adaptable as aif # Interchangeable with aif_functions_isobeliefs and aif_functions_isobeliefs_mcts
importlib.reload(aif)

import os
print(os.cpu_count())

# Re-define the environment and simulation parameters here
interactive = True  # Set to True to display the animation in the notebook
fast_plot = True  # Set to True to use the fast plotter (no icons)
use_ep = True  # Set to True to use EP, False to use the standard algorithm
convergence_type = 'exclusive'  # Set to 'converge' to check for convergence, 'exclusive' to check for separate goals
args = {}
# draft_results = []

random_seed = 2 #39 is good too! 2 goals, 3 agents (actually 39 is perfect)
#2
#35 is good too! 2 goals, 3 agents --- i use this for the paper converging*
#5 is good too! 3 goals, 3 agents --- i use this for the paper exclusive*

np.random.seed(random_seed)  # Set random seed
# Set random goals
env_size = 30 # Environment size
iterations_per_episode = 150  # Number of iterations per episode
num_goals = 2 # Number of goals
num_agents = 2  # Number of agents
list_types = ['A','B']
goals = np.random.uniform(1,env_size-1,size=(num_goals, 2))  # Goal positions
# goals = np.array([[-2,5], [5,5]], dtype=float)
# goals = np.array([[0,0],[env_size-5,0]], dtype=float)  # Goal positions
# goals = np.array([[0.15, 0.1], [0.5,0.55]])
agent_positions = np.hstack((np.random.uniform(0,env_size,size=(num_agents, 2)),np.zeros((num_agents,1))))  # Initial agent positions
# agent_positions = np.array([[0,0,0],[5,0,0]], dtype=float)
# agent_positions = np.array([[4.45,4.45,-3*np.pi/4],[3.45,3.5,-3*np.pi/4]], dtype=float)
number_of_heading_options = 8; number_of_velocity_options = 4
args = dict({
    'goals': goals, # Goal positions
    'home_base': np.array([0,0]), # Home base position
    'agent_types': np.random.choice(list_types,num_agents),#['A','B','B'],#np.random.choice(list_types,num_agents), # Agent types
    'agent_positions': agent_positions, # Initial agent positions
    'velocity_options': np.linspace(0.0,1,number_of_velocity_options,endpoint=True), # Velocity options
    'num_heading_options': number_of_heading_options, # Number of heading options
    'heading_options': np.linspace(-np.pi/4,np.pi/4,number_of_heading_options,endpoint=True),
    'observation_error_std': 0.5, # Observation error standard deviation
    'num_actions': number_of_heading_options*number_of_velocity_options, # Number of actions
    'env_size': env_size, # Environment size
    'max_distance_measure': env_size + 1,
    'max_heading_measure': np.pi, # Maximum heading measure
    'prior': np.ones(goals.shape[0]) / goals.shape[0], # Prior belief
    'use_ep': use_ep, # Use epistemic planning (2nd order reasoning)
    'greedy': False, # Use greedy action selection
    'horizon': 1, # Horizon for free energy checking
    'mcts_iterations': 100, # Number of MCTS iterations
    'use_mcts': False,
    'use_rhc': False,
    'use_threading': False, #TODO: Implement threading
    'convergence_type': convergence_type, # Convergence type
    'dt': 1., # Time step
})

# Check convergence for different types of conergence criterion
if args['convergence_type'] == 'exclusive':
    tuple_elements = [i for i in range(agent_positions.shape[0])]
    configurations = list(itertools.permutations(tuple_elements))
    args['reward_configs'] = configurations # Reward configurations if different goals
else:
    args['reward_configs'] = [tuple(np.repeat(i, num_agents)) for i in range(num_goals)]

# args['reward_configs'] = [(0,1,1), (1,0,1), (1,1,0)]

# Run the simulation
results = aif.run_simulation(args, iterations_per_episode)
print("Final Prior: ", results['priors'])
print("Agent Types: ", args['agent_types'])


# Create animation
if not fast_plot:
    completed_task_img = mpimg.imread('figures/meeting_icon_selected.png')
    not_completed_task_img = mpimg.imread('figures/meeting_icon_not_selected.png')
    plt_sim = aif.PlotSim(num_agents, goals, args['agent_types'], completed_task_img, not_completed_task_img, env_size, padding=5, scale = 0.3)
else:
    plt_sim = aif.PlotSim_fast(num_agents, goals, args['agent_types'],env_size, padding=5, scale = 0.1)
ani = FuncAnimation(plt_sim.fig, plt_sim.update, frames=range(results['iteration']), init_func=plt_sim.init, fargs = (results['plot_args'],), blit=True, repeat=True)

# Save the animation as a video
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
config_file = ["numRobots", num_agents, "_numGoals", num_goals, "_seed", random_seed, "_EP", args['use_ep'], "_goalType", convergence_type, "_envSize", env_size]
config_file = ''.join([str(elem) for elem in config_file])
if results['converged'] == None and not interactive:
    filepath = pwd + "videos/NO_convergence_" + config_file + ".mp4"
    ani.save(filepath, writer='ffmpeg', fps=3, dpi=200)
    # Save location of the final image
    print("Image saved as: ", filepath)
elif not interactive:
    filepath = pwd + "videos/Converged_" + config_file + ".mp4"
    ani.save(filepath, writer='ffmpeg', fps=3, dpi=200)
    # Save location of the final image
    print("Image saved as: ", filepath)
else:
    display(HTML(ani.to_jshtml())) # Use an interactive backend for animation
