"""
Diffusion-based planner using model-based diffusion controller
"""
import logging
import os
import pickle
import time
import traceback

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from ship_ice_planner import PLANNER_PLOT_DIR, PATH_DIR, METRICS_FILE
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.controller.model_based_diffusion import ModelBasedDiffusionController
from ship_ice_planner.ship import Ship
from ship_ice_planner.utils.message_dispatcher import get_communication_interface
from ship_ice_planner.utils.storage import Storage
from ship_ice_planner.utils.utils import resample_path


def diffusion_planner(cfg, debug=False, **kwargs):
    """
    Diffusion-based planner using model-based diffusion controller.
    Similar interface to lattice_planner but uses diffusion for path generation.
    """
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('Starting up diffusion planner...')

    try:
        # setup message dispatcher
        md = get_communication_interface(**kwargs)
        md.start()

        # instantiate main objects
        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
        costmap = CostMap(length=cfg.map_shape[0],
                         width=cfg.map_shape[1],
                         ship_mass=ship.mass,
                         padding=0,  # No swath padding for diffusion planner
                         **cfg.costmap)
        
        # Initialize diffusion controller
        diffusion_cfg = cfg.get('diffusion', {})
        controller = ModelBasedDiffusionController(
            horizon=diffusion_cfg.get('horizon', 50),
            num_samples=diffusion_cfg.get('num_samples', 512),
            num_diffusion_steps=diffusion_cfg.get('num_diffusion_steps', 50),
            temperature=diffusion_cfg.get('temperature', 0.1),
            beta0=diffusion_cfg.get('beta0', 1e-4),
            betaT=diffusion_cfg.get('betaT', 1e-2),
            seed=cfg.get('seed', 0)
        )
        
        metrics = Storage(output_dir=cfg.output_dir, file=METRICS_FILE)

        # directory to store plots
        plot_dir = os.path.join(cfg.output_dir, PLANNER_PLOT_DIR) if cfg.output_dir else None
        if plot_dir:
            os.makedirs(plot_dir)
        # directory to store generated path at each iteration
        path_dir = os.path.join(cfg.output_dir, PATH_DIR) if cfg.output_dir else None
        if path_dir:
            os.makedirs(path_dir)
        if not cfg.plot.show and plot_dir:
            matplotlib.use('Agg')

        # keep track of the planning count
        replan_count = 0
        # keep track of planner rate
        compute_time = []
        prev_goal_y = np.inf
        # keep track of ship xy position in a list for plotting
        ship_actual_path = ([], [])
        horizon = cfg.get('horizon', 0) * costmap.scale
        plot = None

        max_replan = cfg.get('max_replan')
        max_replan = max_replan if max_replan is not None else np.infty

        # start main planner loop
        while replan_count < max_replan:
            logger.info('Re-planning count: {}'.format(replan_count))

            # get new state data
            md.receive_message()

            # start timer
            t0 = time.time()

            # check if the shutdown flag is set
            if md.shutdown:
                logger.info('\033[93mReceived shutdown signal!\033[0m')
                break

            ship_pos = md.ship_state
            goal = md.goal
            obs = md.obstacles
            obs_masses = md.masses

            # scale by the scaling factor
            ship_pos_scaled = ship_pos.copy()
            ship_pos_scaled[:2] *= costmap.scale

            # compute the current goal accounting for horizon
            if goal is not None:
                prev_goal_y = goal[1] * costmap.scale
            if horizon:
                sub_goal = ship_pos_scaled[1] + horizon
                goal_y = min(prev_goal_y, sub_goal)
            else:
                goal_y = prev_goal_y

            # stop planner when ship is within 1 ship length of the goal
            if ship_pos_scaled[1] >= goal_y - ship.length:
                logger.info('\033[92mAt final goal!\033[0m')
                break

            # check if there is new obstacle information
            if obs is not None:
                # update costmap
                t = time.time()
                costmap.update(obs_vertices=obs,
                              obs_masses=obs_masses,
                              ship_pos_y=ship_pos_scaled[1] - ship.length,
                              ship_speed=cfg.sim_dynamics.target_speed,
                              goal=goal_y)
                logger.info('Costmap update time: {}'.format(time.time() - t))

            if debug:
                logger.debug('Showing debug plot for costmap...')
                costmap.plot(ship_pos=ship_pos_scaled,
                            ship_vertices=ship.vertices,
                            goal=goal_y)

            # Generate path using diffusion controller
            t = time.time()
            
            # Convert ship_pos back to meters for controller (ship_pos is already in meters from message dispatcher)
            ship_pos_meters = ship_pos.copy()
            
            # Set goal (goal is in meters, goal_y is in costmap scale)
            goal_pose = [goal[0], goal_y / costmap.scale, np.pi / 2]  # Convert goal_y back to meters
            
            # Create a simple reference path (straight line to goal, will be improved by diffusion)
            horizon_steps = controller.horizon
            local_path = np.zeros((horizon_steps, 3))
            for i in range(horizon_steps):
                # Linear interpolation from current position to goal
                alpha = i / max(horizon_steps - 1, 1)
                local_path[i, 0] = ship_pos_meters[0] * (1 - alpha) + goal_pose[0] * alpha
                local_path[i, 1] = ship_pos_meters[1] * (1 - alpha) + goal_pose[1] * alpha
                local_path[i, 2] = np.pi / 2  # Heading towards goal
            
            # Run diffusion controller to get trajectory
            controller.horizon = horizon_steps  # Ensure horizon matches
            nu_start = [cfg.sim_dynamics.target_speed, 0.0, 0.0]  # Initial velocities
            
            try:
                controller.DPcontrol(
                    pose=ship_pos_meters,
                    setpoint=goal_pose,
                    dt=cfg.sim_dynamics.dt,
                    nu=nu_start,
                    local_path=local_path,
                    costmap=costmap
                )
                
                # Get predicted trajectory
                if hasattr(controller, 'predicted_trajectory'):
                    trajectory = controller.predicted_trajectory
                else:
                    logger.error('Controller did not generate trajectory')
                    replan_count += 1
                    continue
                
            except Exception as e:
                logger.error(f'Diffusion controller failed: {e}')
                traceback.print_exc()
                if path_dir:
                    with open(os.path.join(path_dir, str(replan_count) + '_failed.pkl'), 'wb') as handle:
                        pickle.dump({
                            'replan_count': replan_count,
                            'stamp': t0,
                            'goal': goal_y,
                            'ship_state': ship_pos_scaled,
                            'error': str(e),
                            'raw_message': md.raw_message,
                            'processed_message': md.processed_message
                        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
                replan_count += 1
                continue
            
            logger.info('Diffusion planning time: {}'.format(time.time() - t))

            # Convert trajectory to path format (x, y, psi) in costmap coordinates
            # trajectory is in meters, convert to costmap scale
            path = np.zeros((3, len(trajectory)))
            path[0] = trajectory[:, 0] * costmap.scale  # x
            path[1] = trajectory[:, 1] * costmap.scale  # y
            path[2] = trajectory[:, 2]  # psi (already in radians)

            # Resample path if needed
            if cfg.get('resample_path', False):
                path = resample_path(path, step_size=cfg.get('resample_step_size', 1.0))

            # Send path through pipe (convert to meters and transpose to (N, 3) format)
            path_meters = (path / costmap.scale).T  # Shape: (N, 3)
            md.send_message(path_meters)

            # compute metrics
            compute_time.append(time.time() - t0)
            logger.info('Total compute time: {}'.format(compute_time[-1]))
            logger.info('Average compute time: {}'.format(np.mean(compute_time)))
            
            metrics.store(dict(
                replan_count=replan_count,
                compute_time=compute_time[-1],
                path_length=len(path[0]),
                stamp=t0
            ))

            # save path
            if path_dir:
                with open(os.path.join(path_dir, str(replan_count) + '.pkl'), 'wb') as handle:
                    pickle.dump({
                        'path': path,
                        'trajectory': trajectory,
                        'replan_count': replan_count,
                        'stamp': t0,
                        'goal': goal_y,
                        'ship_state': ship_pos_scaled,
                        'costmap': costmap.cost_map,
                        'obstacles': costmap.obstacles
                    }, handle, protocol=pickle.HIGHEST_PROTOCOL)

            replan_count += 1

            # plotting
            if cfg.plot.show and replan_count <= cfg.plot.max_iter:
                ship_actual_path[0].append(ship_pos_scaled[0])
                ship_actual_path[1].append(ship_pos_scaled[1])

                if plot is None:
                    plot = plt.figure(figsize=(10, 10))
                    ax = plot.add_subplot(111)

                ax.cla()
                # Plot costmap
                cost_map = costmap.cost_map.copy()
                if cost_map.sum() == 0:
                    cost_map[:] = 1
                ax.imshow(cost_map, origin='lower', cmap='viridis',
                         extent=[0, cfg.map_shape[1] / costmap.scale,
                               0, cfg.map_shape[0] / costmap.scale])
                
                # Plot obstacles
                from matplotlib import patches
                for obs in costmap.obstacles:
                    poly_verts = obs['vertices'] / costmap.scale
                    poly = patches.Polygon(poly_verts, fill=False, edgecolor='cyan', linewidth=0.5)
                    ax.add_patch(poly)
                
                # Plot path
                ax.plot(path[0] / costmap.scale, path[1] / costmap.scale, 'g-', linewidth=2, label='Diffusion Path')
                ax.plot(ship_actual_path[0] / costmap.scale, ship_actual_path[1] / costmap.scale, 
                       'r--', linewidth=1, label='Actual Path')
                
                # Plot ship
                from ship_ice_planner.geometry.utils import Rxy
                R = Rxy(ship_pos_scaled[2])
                ship_poly = ship.vertices @ R.T + [ship_pos_scaled[0] / costmap.scale, 
                                                   ship_pos_scaled[1] / costmap.scale]
                ax.add_patch(patches.Polygon(ship_poly, True, fill=True, color='red'))
                
                ax.set_title(f'Diffusion Planner - Iteration {replan_count}')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.legend()
                ax.set_xlim(0, cfg.map_shape[1] / costmap.scale)
                ax.set_ylim(0, cfg.map_shape[0] / costmap.scale)
                
                plt.pause(0.01)

        logger.info('Done diffusion planner!')

    except KeyboardInterrupt:
        logger.info('\033[93mReceived keyboard interrupt!\033[0m')
    except Exception as e:
        logger.error(f'Planner error: {e}')
        traceback.print_exc()
        raise
    finally:
        if md is not None:
            md.close()


if __name__ == '__main__':
    # For testing
    pass

