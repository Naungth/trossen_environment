#!/usr/bin/env python3
"""
Toolbox Trossen Task with OpenPI Policy Server

This script runs a toolbox trossen task using the OpenPI policy server.
The task involves manipulating tools in a toolbox using a robotic arm.
"""

import collections
import dataclasses
import logging
import pathlib
import numpy as np
import time
from typing import Dict, Any
import imageio

import os
os.environ["MUJOCO_GL"] = "glfw" 

import mujoco


@dataclasses.dataclass
class Args:
    """Command line arguments for toolbox trossen task."""
    
    # Policy server connection (for pi0)
    host: str = "0.0.0.0"
    port: int = 8000
    max_episode_steps: int = 10000
    replan_steps: int = 50  
    
    # Task parameters
    num_episodes: int = 10
    save_videos: bool = True
    video_out_path: str = "videos"  # Relative to this directory
    
    # Model path (relative to this script's directory)
    xml_model_path: str = "toolbox_trossen/scene.xml"
    
    # Random seed for reproducibility
    seed: int = 42


class ToolboxTrossenEnvironment:
    """
    MuJoCo environment for toolbox trossen task.
    
    This environment loads the scene.xml model and provides the interface
    for the policy to interact with the simulation.
    """
    
    def __init__(self, xml_model_path: str, args: Args):
        """Initialize the toolbox trossen environment."""
        
        self.args = args
        
        script_dir = pathlib.Path(__file__).parent
        self.xml_model_path = script_dir / xml_model_path
        
        logging.info(f"Loading MuJoCo model from: {self.xml_model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_model_path))
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        for _ in range(1):
            mujoco.mj_step(self.model, self.data)
        
        # Rendering setup
        try:
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280) 
            self.render_height, self.render_width = 720, 1280
            logging.info("MuJoCo renderer initialized successfully (1280x720)")
            self.rendering_available = True
        except Exception as e:
            logging.warning(f"Failed to create MuJoCo renderer: {e}")
            logging.warning("Falling back to dummy images")
            self.renderer = None
            self.rendering_available = False
            self.render_height, self.render_width = 720, 1280  
        

        self.wrist_camera_left  = "cam_left_wrist"
        self.wrist_camera_right = "cam_right_wrist"
        
        self.main_cam_params = {
            "distance": 1.1,
            "elevation": -90.0,
            "azimuth": 90.0,
            "lookat": np.array([0.0, 0.25, 0.0], dtype=float),
        }
        

        self.episode_step = 0
        self.total_steps = 0 
        self._last_images = None  
        self._rendering_enabled = True  
        
        
        logging.info("Robot initialization complete")
    
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to initial state."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        for _ in range(1):
            mujoco.mj_step(self.model, self.data)

        self.episode_step = 0
        
        return self.get_observation()
    
    def _randomize_object_positions(self):
        """Randomize positions of tools for each episode."""
        # TODO

        pass
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation from the simulation.
        
        Returns observation in the format expected by the pi0 policy.
        """
        # Get robot state (joint positions)
        robot_joint_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]
        robot_qpos = self.data.qpos[robot_joint_indices]  # 14 robot joint positions
        robot_state = robot_qpos  # 14-D
        
        # Render new images for video recording, reuse for policy inference
        if self.rendering_available and self.renderer is not None and self._rendering_enabled:
            # Always render for video quality, but cache for policy efficiency
            main_image = self._render_main_view()
            
            try:
                self.renderer.update_scene(self.data, camera=self.wrist_camera_left)
                left_wrist_img = self.renderer.render()
            except Exception:
                left_wrist_img = main_image.copy()
            
            try:
                self.renderer.update_scene(self.data, camera=self.wrist_camera_right)
                right_wrist_img = self.renderer.render()
            except Exception:
                right_wrist_img = left_wrist_img.copy()
            
            # Cache images for policy inference (only update every 50 steps for efficiency)
            if self.total_steps % 50 == 0 or self._last_images is None:
                self._last_images = (main_image, left_wrist_img, right_wrist_img)
        else:
            # Dummy images when rendering is disabled
            main_image = np.zeros((720, 1280, 3), dtype=np.uint8)  # 720p resolution
            left_wrist_img = np.zeros((720, 1280, 3), dtype=np.uint8)  # 720p resolution
            right_wrist_img = np.zeros((720, 1280, 3), dtype=np.uint8)  # 720p resolution


        main_image_processed = main_image.astype(np.uint8)
        left_wrist_processed = left_wrist_img.astype(np.uint8)
        right_wrist_processed = right_wrist_img.astype(np.uint8)

        observation = {
            # --- Flat image keys (Pi-0 compatible) ---
            "image.pov":    main_image_processed,   # Main camera
            "image.left":   left_wrist_processed,
            "image.right":  right_wrist_processed,
            "image.top":    main_image_processed,   

            "state": robot_state,  # 14-D: 14 joint positions
            "prompt": "pick up the tools and organize them in the toolbox",
        }
        
        return observation
    
    def step(self, action: np.ndarray) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Apply action to the simulation and return new observation.
        
        Args:
            action: Action from the policy (14D joint targets for dual arm)
            
        Returns:
            observation: New observation after applying action
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """

        if not np.isfinite(action).all():
            logging.warning("Invalid action received, using zero action")
            action = np.zeros_like(action)
        

        joint_commands = np.array(action, dtype=np.float64)
        
        if len(joint_commands) != self.model.nu:
            logging.warning(f"Action dimension mismatch: got {len(joint_commands)}, expected {self.model.nu}")

            if len(joint_commands) < self.model.nu:
                joint_commands = np.pad(joint_commands, (0, self.model.nu - len(joint_commands)))
            else:
                joint_commands = joint_commands[:self.model.nu]
        

        self.data.ctrl[:] = joint_commands
        
        expert_dt = 0.02 
        sim_dt = self.model.opt.timestep  
        steps_per_action = int(round(expert_dt / sim_dt))
        
        for _ in range(steps_per_action):
            self.data.ctrl[:] = joint_commands
            mujoco.mj_step(self.model, self.data)
        
        self.episode_step += 1
        self.total_steps += 1
        
        done = self.episode_step >= self.args.max_episode_steps or self._check_success()
        

        reward = self._compute_reward()
        
        observation = self.get_observation()
        info = {"episode_step": self.episode_step}
        
        return observation, reward, done, info


    def render_frame(self, camera: str = "main") -> np.ndarray:
        """Render a frame from the specified camera ('main', 'left', 'right')."""
        if not self.rendering_available or self.renderer is None:
            return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        
        if camera == "main":
            return self._render_main_view()
        elif camera == "left":
            self.renderer.update_scene(self.data, camera=self.wrist_camera_left)
            return self.renderer.render()
        elif camera == "right":
            self.renderer.update_scene(self.data, camera=self.wrist_camera_right)
            return self.renderer.render()
        else:
            raise ValueError(f"Unknown camera: {camera}")

    def _render_main_view(self) -> np.ndarray:
        """Render a free camera with top-down parameters as the main view."""
        if not self.rendering_available or self.renderer is None:
            return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)

        cam.distance = self.main_cam_params["distance"]
        cam.elevation = self.main_cam_params["elevation"]
        cam.azimuth  = self.main_cam_params["azimuth"]
        cam.lookat[:] = self.main_cam_params["lookat"]

        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()

    def set_rendering_enabled(self, enabled: bool):
        """Enable or disable rendering for performance."""
        self._rendering_enabled = enabled



def render_qpos_frame(qpos, camera='main', out_path='qpos_render.png'):
    """Render a frame from a given qpos vector and save it as an image."""
    
    args = Args()
    env = ToolboxTrossenEnvironment(args.xml_model_path, args)
    
    # Reset environment first to ensure proper initialization
    env.reset()
    
    # Convert qpos to numpy array and check dimensions
    qpos_array = np.array(qpos, dtype=np.float64)
    print(f"Input qpos shape: {qpos_array.shape}")
    print(f"Environment qpos shape: {env.data.qpos.shape}")
    
    # Ensure qpos length matches environment
    if len(qpos_array) != len(env.data.qpos):
        print(f"Warning: qpos length mismatch. Input: {len(qpos_array)}, Environment: {len(env.data.qpos)}")
        if len(qpos_array) > len(env.data.qpos):
            qpos_array = qpos_array[:len(env.data.qpos)]
            print(f"Truncated qpos to {len(qpos_array)} elements")
        else:
            # Pad with zeros if input is shorter
            padding = np.zeros(len(env.data.qpos) - len(qpos_array))
            qpos_array = np.concatenate([qpos_array, padding])
            print(f"Padded qpos to {len(qpos_array)} elements")
    
    env.data.qpos[:] = qpos_array
    env.data.qvel[:] = 0.0  
    

    mujoco.mj_forward(env.model, env.data)
    

    img = env.render_frame(camera=camera)
    

    imageio.imwrite(out_path, img)
    print(f"Rendered frame saved as {out_path}")
    print(f"Image shape: {img.shape}")
    
    return img


def test_toolbox_environment(args: Args) -> None:
    """Test the toolbox trossen environment with simple actions."""
    
    # Create output directory for videos
    script_dir = pathlib.Path(__file__).parent
    video_path = script_dir / args.video_out_path
    if args.save_videos:
        video_path.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    try:
        env = ToolboxTrossenEnvironment(args.xml_model_path, args)
        logging.info("Environment created successfully")
    except Exception as e:
        logging.error(f"Failed to initialize environment: {e}")
        return
    
    # Test environment with simple actions
    logging.info("Testing environment with simple actions...")
    
    # Reset environment
    obs = env.reset()
    logging.info(f"Initial observation keys: {obs.keys()}")
    
    # Storage for video recording
    episode_frames = []
    
    step_count = 0
    done = False
    episode_reward = 0.0
    
    # Test with simple sinusoidal actions
    while step_count < 100 and not done:  # Test for 100 steps
        try:
            # Generate simple sinusoidal action for testing
            action = np.sin(step_count * 0.1 + np.arange(14) * 0.5) * 0.1
            logging.debug(f"Step {step_count} executing action: {action[:3]}...")
            
            obs, reward, done, info = env.step(action)
            
            # Store frame for video
            if args.save_videos and "image.pov" in obs:
                video_frame = obs["image.pov"]
                episode_frames.append(video_frame.copy())
            
            step_count += 1
            episode_reward += reward
            
            # Log progress periodically
            if step_count % 10 == 0 or step_count <= 5:
                logging.info(f"Step {step_count}, Reward: {reward:.3f}, Total: {episode_reward:.3f}")
                logging.info(f"Action (first 3 joints): [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
            
            if done:
                logging.info(f"Episode completed in {step_count} steps!")
                break
                
        except Exception as e:
            logging.error(f"Error during episode: {e}")
            break
    
    # Save video if requested
    if args.save_videos and episode_frames:
        try:
            import imageio  # Import here to make it optional
            
            video_file = video_path / "test_episode.mp4"
            
            logging.info(f"Saving test video to {video_file} ({len(episode_frames)} frames)")
            if len(episode_frames) > 0:
                imageio.mimwrite(str(video_file), episode_frames, fps=30)
            else:
                logging.warning("No frames recorded for video")
        except ImportError:
            logging.warning("imageio not available, skipping video recording")
        except Exception as e:
            logging.error(f"Failed to save video: {e}")
    
    logging.info("Environment test completed!")


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Toolbox Trossen Environment Test")
    logging.info("=" * 50)
    
    # Create args manually since we removed tyro
    args = Args()
    test_toolbox_environment(args)


if __name__ == "__main__":
    main() 