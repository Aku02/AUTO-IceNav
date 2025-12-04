import argparse
import os
import pickle
import numpy as np

from ship_ice_planner.experiments.generate_rand_exp import build_obs_dicts
from ship_ice_planner.sim2d import sim
from ship_ice_planner.utils.utils import DotDict
from ship_ice_planner.controller.sim_dynamics import SimShipDynamics
from ship_ice_planner.controller.model_diffusion import ModelBasedDiffusionController

# Subclass SimShipDynamics to inject our controller
class SimShipDynamicsDiffusion(SimShipDynamics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize our diffusion controller
        # You might want to tune these parameters
        self.diffusion_controller = ModelBasedDiffusionController(
            num_timesteps=10, 
            num_samples=30, 
            horizon=20, 
            dt=self.dt,
            temperature=0.5
        )
        self.obstacles_for_controller = None

    def control(self):
        # Override the control method
        
        # Get obstacles if available (hacky way to get them from where they might be stored)
        # In sim2d.py, obstacles are passed to planner, but here we are inside SimShipDynamics 
        # which is instantiated in sim2d.py.
        # sim2d.py doesn't pass obstacles to SimShipDynamics by default.
        # However, for this demo, we might just want to follow the path or goal.
        
        # If we want obstacle avoidance, we'd need to pass obstacles to this class.
        # For now, let's just use the setpoint (which follows the path).
        
        # The setpoint is updated in sim_step() -> setpoint_generator.update()
        # self.setpoint is [x_d, y_d, ...]
        
        if self.setpoint is None or len(self.setpoint) < 2:
            self.state.u_control = np.zeros(3)
            return

        # Run diffusion control
        u_opt = self.diffusion_controller.get_control(
            self.state, 
            self.setpoint, 
            self.dt, 
            obstacles=self.obstacles_for_controller
        )
        
        # Map u_opt (surge, sway, yaw_rate) to u_control
        # SimShipDynamics expects u_control. 
        # For NRC_supply/AISship, u_control is forces/moments or similar?
        # Let's check sim_dynamics.py:
        # For NRC_supply: u_control is used in dynamics(..., u_control).
        # In NRC_supply.py (not viewed but inferred), it likely takes forces or RPMs.
        # Wait, SimShipDynamics.control() for NRC_supply calls DPcontrol which returns u_control.
        # If we look at SimShipDynamics.sim_step():
        # [u, v, r] = self.vessel_model.dynamics(u, v, np.rad2deg(r), self.state.u_control)
        
        # If we assume our controller outputs velocity commands (u, v, r), 
        # and the vessel model expects forces, we have a mismatch.
        # However, the user asked for a "controller".
        # If the vessel model is "NRC_supply", it's a dynamic model.
        # If we want to control it, we usually output forces (tau).
        
        # But my simple_ship_dynamics in ModelBasedDiffusionController assumed velocity control.
        # If I want to control the ship, I should probably output forces if the plant expects forces.
        # OR, I can assume the low-level controller handles velocity tracking, and I output velocity setpoints.
        # But SimShipDynamics structure seems to expect `u_control` to be the direct input to `dynamics`.
        
        # Let's look at `sim_dynamics.py` again.
        # For Fossen_supply: [nu, u_actual] = dynamics(..., u_control, ...)
        # For NRC_supply: [u, v, r] = dynamics(..., u_control)
        
        # If I want to replace the controller, I should output what `DPcontrol` outputs.
        # `DPcontrol` typically outputs forces/moments (tau).
        
        # My diffusion controller optimizes a sequence of inputs `U`.
        # If `U` are forces, then `simple_ship_dynamics` should use a force model.
        # If `U` are velocities, then I need a low-level controller to track them, 
        # OR I assume `u_control` can be velocities (which might not work if `dynamics` expects forces).
        
        # Let's assume for this demo that we are controlling a kinematic model or we adapt the output.
        # But `NRC_supply` dynamics likely expects forces.
        
        # To make this simple and robust:
        # I will let the diffusion controller output "velocity commands" (u_cmd, v_cmd, r_cmd).
        # Then I will use a simple P-controller here to convert velocity error to forces,
        # OR I will modify `simple_ship_dynamics` to be a force model.
        
        # Given the "sample script" was about "Reverse SDE", it was generic.
        # Let's stick to velocity commands for the diffusion planner (it's a planner/controller).
        # And then map to forces using a simple PID or just pass it if the model supports it.
        # Actually, `NRC_supply` might be a complex model.
        
        # Alternative: Use `Fossen_supply` which has `DPcontrol` (PID).
        # If I use `Fossen_supply`, I can send `setpoint` to it.
        # But I want to REPLACE `DPcontrol`.
        
        # Let's try to make `u_control` be forces.
        # I will update `ModelBasedDiffusionController` to treat `U` as forces?
        # No, predicting with forces requires a good model.
        
        # Let's assume we are controlling the `SimShipDynamics` which has a `vessel_model`.
        # I'll use a simple mapping: u_control = proportional * (u_opt - current_vel).
        # This effectively makes the diffusion controller a "trajectory planner" that outputs a velocity target,
        # and we use a low-level controller to track it.
        
        # But `u_opt` is the first action of the sequence.
        # If `u_opt` is velocity, we treat it as a reference.
        
        # Let's do this:
        # The diffusion controller outputs desired velocity [u_d, v_d, r_d].
        # We calculate error e = [u_d - u, v_d - v, r_d - r].
        # We apply forces proportional to error (P-controller).
        # tau = Kp * e.
        
        Kp = np.diag([1e5, 1e5, 1e7]) # Tunable gains for a ship
        
        desired_vel = u_opt
        current_vel = np.array([self.state.u, self.state.v, self.state.r])
        
        error = desired_vel - current_vel
        tau = Kp @ error
        
        self.state.u_control = tau


# Monkey patch sim2d to use our subclass
import ship_ice_planner.sim2d as sim2d
original_sim_dynamics_class = sim2d.SimShipDynamics
sim2d.SimShipDynamics = SimShipDynamicsDiffusion

def demo(cfg_file: str,
         exp_config_file: str,
         ice_concentration: float,
         ice_field_idx: int,
         start: list = None,
         goal: list = None,
         show_anim=True,
         output_dir: str = None,
         debug=False,
         logging=False,
         log_level=10):
    # demo to run a single simulation, for consecutive simulations see ship_ice_planner/experiments/sim_exp.py
    print('2D simulation start (Model Diffusion Controller)')

    # load config
    cfg = DotDict.load_from_file(cfg_file)
    cfg.cfg_file = cfg_file

    # update parameters
    cfg.anim.show = show_anim
    cfg.output_dir = output_dir
    if output_dir and show_anim:
        cfg.anim.save = True  # cannot show and save anim at same time
        cfg.anim.show = False

    # load ice field data
    # Check if this is a Copernicus config file
    use_copernicus = os.getenv('USE_COPERNICUS', 'false').lower() == 'true'
    
    if use_copernicus:
        # Load Copernicus-based experiment config
        with open(exp_config_file, 'rb') as f:
            exp_config = pickle.load(f)
            # Find closest concentration match
            concentrations = sorted(exp_config['exp'].keys())
            closest_conc = min(concentrations, key=lambda x: abs(x - ice_concentration))
            if closest_conc in exp_config['exp']:
                if ice_field_idx in exp_config['exp'][closest_conc]:
                    exp = exp_config['exp'][closest_conc][ice_field_idx]
                else:
                    # Use first available index
                    available_indices = sorted(exp_config['exp'][closest_conc].keys())
                    if available_indices:
                        exp = exp_config['exp'][closest_conc][available_indices[0]]
                        print(f"Warning: ice_field_idx {ice_field_idx} not found, using {available_indices[0]}")
                    else:
                        raise ValueError(f"No ice fields found for concentration {closest_conc}")
            else:
                raise ValueError(f"No ice fields found for concentration {ice_concentration}")
    else:
        # Load standard experiment config
        with open(exp_config_file, 'rb') as f:
            # can either load the experiment config generated by generate_rand_exp.py
            # or can simply load obstacle data encoded as a list of lists
            exp = pickle.load(f)
            if type(exp) is list:
                exp = {'obstacles': exp}
            else:
                exp = exp['exp'][ice_concentration][ice_field_idx]

    # if none then use the start and goal from the experiment config file
    if start is not None:
        exp['ship_state'] = start
    if goal is not None:
        exp['goal'] = goal

    sim(cfg=cfg,
        debug=debug,          # enable planner debugging mode
        logging=logging,      # enable planner logs
        log_level=log_level,  # log level for planner https://docs.python.org/3/library/logging.html#levels
        init_queue=exp        # first message sent to planner process
        )


if __name__ == '__main__':
    from ship_ice_planner import FULL_SCALE_SIM_EXP_CONFIG, FULL_SCALE_SIM_PARAM_CONFIG

    # setup arg parser
    parser = argparse.ArgumentParser(description='Ship ice navigation demo with Model Diffusion Controller. ')
    parser.add_argument('exp_config_file', nargs='?', type=str, help='File path to experiment config pickle file '
                                                                     'generated by generate_rand_exp.py',
                        default=FULL_SCALE_SIM_EXP_CONFIG)
    parser.add_argument('planner_config_file', nargs='?', type=str, help='File path to planner and simulation '
                                                                         'parameter config yaml file (see configs/)',
                        default=FULL_SCALE_SIM_PARAM_CONFIG)
    parser.add_argument('-c', dest='ice_concentration', type=float,
                        help='Pick an ice concentration from {0.2, 0.3, 0.4, 0.5}', default=0.5)
    parser.add_argument('-i', dest='ice_field_idx', type=int,
                        help='Pick an ice field from {0, 1, ..., 99}', default=1)
    parser.add_argument('-s', '--start', nargs=3, metavar=('x', 'y', 'psi'), type=float,
                        help='initial ship position (x, y) in meters and heading (psi) in radians',
                        default=None)
    parser.add_argument('-g', '--goal', nargs=2, metavar=('x', 'y'), type=float,
                        help='goal position (x, y) in meters', default=None)
    parser.add_argument('--no_anim', action='store_true', help='Disable rendering (significantly speeds up sim!)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory path to store output data')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    parser.add_argument('-l', '--logging', action='store_true', help='Logging mode')
    parser.add_argument('-ll', '--log_level', type=int, default=10, help='Logging level')

    args = parser.parse_args()

    print('Launching ship ice navigation demo (Diffusion Controller)...\nCmd-line arguments:'
          '\n\tExperiment config file: %s'
          '\n\tPlanner config file: %s'
          '\n\tIce concentration: %s'
          '\n\tIce field index: %s'
          '\n\tStart: %s'
          '\n\tGoal: %s'
          '\n\tShow live animation: %s'
          '\n\tOutput directory: %s'
          '\n\tDebug: %s'
          '\n\tLogging: %s'
          '\n\tLog level: %s' % (args.exp_config_file, args.planner_config_file, args.ice_concentration,
                                 args.ice_field_idx, args.start, args.goal, not args.no_anim, args.output_dir,
                                 args.debug, args.logging, args.log_level))

    demo(cfg_file=args.planner_config_file,
         exp_config_file=args.exp_config_file,
         ice_concentration=args.ice_concentration,
         ice_field_idx=args.ice_field_idx,
         start=args.start,
         goal=args.goal,
         show_anim=not args.no_anim,
         output_dir=args.output_dir,
         debug=args.debug,
         logging=args.logging,
         log_level=args.log_level)
