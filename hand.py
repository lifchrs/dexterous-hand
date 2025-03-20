from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig, 
    PassiveControllerConfig,
    deepcopy_dict
)
import os

@register_agent()
class InspireHand(BaseAgent):
    uid = "InspireHand"
    urdf_path = os.path.join("inspire_hand", "inspire_hand_right.urdf")
    
    active_joints = [
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "index_proximal_joint",
        "middle_proximal_joint",
        "ring_proximal_joint",
        "pinky_proximal_joint"
    ]
    
    # These are modelled as passive joints
    mimic_joints = [
        "thumb_intermediate_joint",
        "thumb_distal_joint",
        "index_intermediate_joint",
        "middle_intermediate_joint",
        "ring_intermediate_joint",
        "pinky_intermediate_joint"
    ]

    disable_self_collisions = True
    load_multiple_collisions = True

    @property
    def _controller_configs(self):
        
        # Stiffness and damping are my guesses
        joint_stiffness = 0.2
        joint_damping = 0.1
        joint_force_limit = 1 # force_limit of 1 is listed in the urdf, fingers only don't fly away to nan with this set to around <=0.05
        
        passive_controller = PassiveControllerConfig(
            self.mimic_joints,
            joint_damping,
            joint_force_limit
        )
        # Using absolute position control
        joint_pos = PDJointPosControllerConfig(
            self.active_joints,
            None,  # use URDF limits
            None,  # use URDF limits
            joint_stiffness,
            joint_damping,
            joint_force_limit,
            normalize_action=True,
        )
        
        controller_configs = dict(
            pd_joint_pos=dict(
                hand=joint_pos,
                passive_finger_joints=passive_controller,
            ),
        )
        
        return deepcopy_dict(controller_configs)


if __name__ == "__main__":
    import gymnasium as gym
    from mani_skill.envs import EmptyEnv  

    env: EmptyEnv = gym.make(
        "Empty-v1",
        robot_uids="InspireHand",
        num_envs=1,
        control_mode="pd_joint_pos",  
        render_mode="rgb_array"
    )
    
    env.reset()

    frames = []
    done = False
    env.render()
    # Start with the window paused (for stepping)
    # env.viewer.paused = True
    for _ in range(20):  # Capture the first 20 frames
        frame = env.render()[0]
        frames.append(frame.cpu().numpy())  # Assuming get_image() captures the current frame
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    # Save frames as a video
    import cv2
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    env.close()