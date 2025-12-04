import numpy as np
import matplotlib.pyplot as plt
import random
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.controller.model_based_diffusion import ModelBasedDiffusionController
from ship_ice_planner.utils.sim_utils import load_real_obstacles, ICE_DENSITY, ICE_THICKNESS, SHIP_MASS
from ship_ice_planner.ship import FULL_SCALE_PSV_VERTICES, Ship
from ship_ice_planner.a_star_search import AStar
from ship_ice_planner.primitives import Primitives
from ship_ice_planner.swath import generate_swath

def demo_static():
    # 1. Setup Environment
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Map Configuration
    MAP_FILE = "data/MEDEA_fram_20100629.pkl"
    PADDING = 50
    
    # Channel dimensions (matched to sim2d_config.yaml)
    width = 600  # m
    height = 600  # m
    scale = 0.5  # pixels per meter (matched to config)

    # Ship
    ship = Ship(scale=scale, vertices=FULL_SCALE_PSV_VERTICES, mass=SHIP_MASS)
    # Start pose from config: [50.0, 0.0, 1.57...]
    start_pose = np.array([50.0, 0.0, np.deg2rad(90)]) # x, y, psi (facing up)
    start_nu = np.array([2.0, 0.0, 0.0]) # Initial velocity (surge, sway, yaw rate), target speed 2.0

    # Goal
    goal_y = 600.0
    
    # Load Real Obstacles
    print(f"Loading map from {MAP_FILE}...")
    obs_dicts, _ = load_real_obstacles(MAP_FILE)
    
    # Shift obstacles (apply padding)
    print(f"Applying padding of {PADDING}m...")
    for ob in obs_dicts:
        ob['vertices'][:, 1] += np.array(PADDING).astype(np.int32)
        # Re-calculate centre as it might be needed by CostMap or visualization
        ob['centre'] = np.mean(ob['vertices'], axis=0)
        
    obstacles = [ob['vertices'] for ob in obs_dicts]

    # Initialize CostMap
    costmap = CostMap(
        scale=scale,
        length=height, width=width,
        ship_mass=ship.mass,
        collision_cost_weight=0.00000048, # From config
        obs_cost_buffer_factor=0.15, # From config
        ice_thickness=ICE_THICKNESS,
        ice_density=ICE_DENSITY,
        ice_resistance_weight=1.0, # From config
        sic_kernel=(51, 51) # From config
    )
    
    # Update CostMap
    print("Updating CostMap...")
    costmap.update(obstacles, ship_pos_y=start_pose[1], ship_speed=start_nu[0], goal=goal_y)

    # 2. Initialize Lattice Planner for Better Reference Path
    print("Initializing lattice planner...")
    prim = Primitives(prim_name='PRIM_8H_4', scale=15, step_size=0.5)
    swath_dict, swath_shape = generate_swath(ship, prim)
    
    a_star = AStar(
        full_costmap=costmap,
        prim=prim,
        ship=ship,
        swath_dict=swath_dict,
        swath_shape=swath_shape,
        weight=1  # A* weight
    )
    
    # Run lattice planner to get initial path
    print("Running lattice planner for initial path...")
    start_pose_scaled = (start_pose[0] * scale, start_pose[1] * scale, start_pose[2])
    search_result = a_star.search(
        start=start_pose_scaled,
        goal_y=goal_y * scale  # Convert to costmap units
    )
    
    if search_result:
        lattice_path, _, _, _, _, _, _, _ = search_result
        # Convert from costmap units to meters
        local_path = np.c_[(lattice_path[:2] / scale).T, lattice_path[2]]
        print(f"Lattice planner found path with {len(local_path)} waypoints")
    else:
        print("Lattice planner failed, using straight line fallback")
        # Fallback to straight line if planning fails
        horizon = 500
        path_steps = int(horizon)
        target_y = np.linspace(start_pose[1], goal_y, path_steps) 
        target_x = np.ones_like(target_y) * start_pose[0]
        target_psi = np.ones_like(target_y) * np.deg2rad(90)
        local_path = np.column_stack([target_x, target_y, target_psi])

    # 3. Setup Controller & Run Multiple Trials (for Multi-Modality)
    num_trials = 5
    colors = ['r', 'm', 'orange', 'cyan', 'yellow']
    
    fig, ax = plt.subplots(figsize=(10, 10)) # Square aspect ratio for 600x600 map
    
    # Plot CostMap first
    print(f"CostMap Max Value: {costmap.cost_map.max()}")
    cmap_img = costmap.cost_map.copy()
    if cmap_img.sum() == 0: cmap_img[:] = 1 
    ax.imshow(cmap_img, origin='lower', cmap='viridis', extent=[0, width, 0, height])
    
    # Plot Obstacles
    from matplotlib import patches
    for obs in costmap.obstacles:
        # Use vertices for polygons instead of circles for better accuracy with real ice floes
        # CostMap stores vertices in pixels (scaled), so we need to unscale to meters
        poly_verts = obs['vertices'] / costmap.scale
        poly = patches.Polygon(poly_verts, fill=False, edgecolor='cyan', linewidth=0.5)
        ax.add_patch(poly)
        
    # Plot Ship
    from ship_ice_planner.geometry.utils import Rxy
    R = Rxy(start_pose[2])
    ship_poly = ship.vertices @ R.T + [start_pose[0], start_pose[1]]
    ax.add_patch(patches.Polygon(ship_poly, True, fill=True, color='red', label='Ship'))
    
    # Plot Reference Path (from lattice planner)
    ax.plot(local_path[:, 0], local_path[:, 1], 'g--', label='Lattice Reference Path', linewidth=2, alpha=0.7)

    print(f"Generating trajectories (running {num_trials} trials)...")
    
    for i in range(num_trials):
        # Instantiate new controller for each trial to get independent random seeds/initializations
        controller = ModelBasedDiffusionController(
            horizon=len(local_path),  # Match lattice path length
            num_samples=512,  # Number of samples per diffusion step
            num_diffusion_steps=50,  # Number of diffusion steps
            temperature=0.1,
            seed=seed + i  # Different seed for each trial
        )
        
        # Run Control
        controller.DPcontrol(
            pose=start_pose,
            setpoint=[start_pose[0], goal_y, np.deg2rad(90)],
            dt=0.1, 
            nu=start_nu,
            local_path=local_path,
            costmap=costmap
        )
        
        # Plot Mean Trajectory
        if hasattr(controller, 'predicted_trajectory'):
            traj = controller.predicted_trajectory
            ax.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=3, label=f'Mode {i+1} Mean')
            
        # Plot Sampled Trajectories (from the last trial only, to avoid clutter)
        if i == num_trials - 1 and hasattr(controller, 'sampled_trajectories_viz'):
            print(f"Plotting {len(controller.sampled_trajectories_viz)} sampled trajectories from Trial {i+1}.")
            for j, traj in enumerate(controller.sampled_trajectories_viz):
                label = 'Sampled Trajectories' if j == 0 else None
                ax.plot(traj[:, 0], traj[:, 1], 'w-', alpha=0.5, linewidth=1, label=label)

    ax.set_title(f"Diffusion Controller: Multi-Modal Trajectory Generation\n(Horizon={controller.horizon}, Samples={controller.num_samples}, Diffusion Steps={controller.diffusion_steps})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    plt.savefig('diffusion_static_demo.png', dpi=150)
    print("Saved plot to diffusion_static_demo.png")
    plt.show()

if __name__ == "__main__":
    demo_static()
