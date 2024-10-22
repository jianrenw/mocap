import joblib
import torch


class MotionLib:
    def __init__(self, num_envs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = num_envs
        self.motion_train_file = "data/h1_isaac.pt"
        self.load_data(self.motion_train_file)

    def load_data(self, motion_train_file):
        motion_dict = joblib.load(motion_train_file)
        self.motion_names = []
        self.body_poses = []
        self.root_poses = []
        self.dof_poses = []
        self.body_rots = []
        self.root_rots = []
        self.body_vels = []
        self.root_vels = []
        self.body_angular_vels = []
        self.root_angular_vels = []
        self.dof_vels = []

        self.motion_lens = []
        self.dts = []
        self.num_frames = []
        for key, value in motion_dict.items():
            self.motion_names.append(key)
            self.body_poses.append(value["body_pos"])
            self.root_poses.append(value["root_pos"] + 0.1)
            self.dof_poses.append(value["dof_pos"])
            self.body_rots.append(value["body_rot"])
            self.root_rots.append(value["root_rot"])
            self.body_vels.append(value["body_vel"])
            self.root_vels.append(value["root_vel"])
            self.body_angular_vels.append(value["body_angular_vel"])
            self.root_angular_vels.append(value["root_angular_vel"])
            self.dof_vels.append(value["dof_vel"])

            num_frames = value["root_pos"].shape[0]
            curr_dt = value["dt"]
            curr_len = curr_dt * (num_frames - 1)
            self.motion_lens.append(curr_len)
            self.num_frames.append(num_frames)
            self.dts.append(curr_dt)
        self.num_motions = len(self.motion_names)
        self.num_frames = torch.Tensor(self.num_frames).to(self.device)
        self.motion_lens = torch.Tensor(self.motion_lens).to(self.device)

    def load_motions(self):
        self.sampled_motion_idx = torch.randint(
            self.num_motions, size=(self.num_envs,)
        ).to(self.device)
        sampled_motion_names = []

        sampled_motion_lens = []
        sampled_body_poses = []
        sampled_root_poses = []
        sampled_dof_poses = []
        sampled_body_rots = []
        sampled_root_rots = []
        sampled_body_vels = []
        sampled_root_vels = []
        sampled_body_angular_vels = []
        sampled_root_angular_vels = []
        sampled_dof_vels = []

        sampled_frame_lens = []
        sampled_dts = []
        for i in self.sampled_motion_idx:
            sampled_motion_names.append(self.motion_names[i])
            sampled_body_poses.append(torch.from_numpy(self.body_poses[i]))
            sampled_root_poses.append(torch.from_numpy(self.root_poses[i]))
            # import ipdb; ipdb.set_trace()
            # print('....')
            if (self.root_poses[i][:, -1] < 0).any():
                print("sampled root z pos < 0")
                import ipdb

                ipdb.set_trace()
            sampled_root_rots.append(torch.from_numpy(self.root_rots[i]))
            sampled_body_rots.append(torch.from_numpy(self.body_rots[i]))
            sampled_body_vels.append(torch.from_numpy(self.body_vels[i]))
            sampled_root_vels.append(torch.from_numpy(self.root_vels[i]))
            sampled_body_angular_vels.append(
                torch.from_numpy(self.body_angular_vels[i])
            )
            sampled_root_angular_vels.append(
                torch.from_numpy(self.root_angular_vels[i])
            )
            sampled_dof_vels.append(torch.from_numpy(self.dof_vels[i]))
            sampled_dof_poses.append(torch.from_numpy(self.dof_poses[i]))

            sampled_motion_lens.append(self.motion_lens[i])
            sampled_frame_lens.append(self.num_frames[i])
            sampled_dts.append(self.dts[i])

        self.sampled_root_poses = torch.cat(sampled_root_poses, axis=0).to(self.device)
        self.sampled_body_poses = torch.cat(sampled_body_poses, axis=0).to(self.device)
        self.sampled_body_rots = torch.cat(sampled_body_rots, axis=0).to(self.device)
        self.sampled_root_vels = torch.cat(sampled_root_vels, axis=0).to(self.device)
        self.sampled_body_vels = torch.cat(sampled_body_vels, axis=0).to(self.device)
        self.sampled_root_angular_vels = torch.cat(
            sampled_root_angular_vels, axis=0
        ).to(self.device)
        self.sampled_body_angular_vels = torch.cat(
            sampled_body_angular_vels, axis=0
        ).to(self.device)
        self.sampled_dof_vels = torch.cat(sampled_dof_vels, axis=0).to(self.device)
        self.sampled_root_rots = torch.cat(sampled_root_rots, axis=0).to(self.device)
        self.sampled_dof_poses = torch.cat(sampled_dof_poses, axis=0).to(self.device)
        self.sampled_motion_lens = torch.Tensor(sampled_motion_lens).to(self.device)
        self.sampled_frame_lens = torch.Tensor(sampled_frame_lens).to(self.device)
        self.sampled_dts = torch.Tensor(sampled_dts).to(self.device)
        frame_indices = torch.cumsum(self.sampled_frame_lens, dim=0)
        frame_indices = torch.roll(frame_indices, 1)
        frame_indices[0] = 0
        self.frame_indices = frame_indices.to(self.device)

    # def sample_ref(self):
    #     sampled_frame = torch.floor(torch.rand(self.num_envs) * (self.sampled_frame_len - 1))
    #     return self.get_sampled_motion_state(motion_times, None)

    def sample_motions(self, n):
        motion_ids = torch.randint(self.num_motions, size=(n,)).to(self.device)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):

        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self.device)
        motion_len = self.motion_lens[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def sample_time_interval(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self.device)
        motion_len = self.motion_lens[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
        curr_fps = 1 / 30
        motion_time = ((phase * motion_len) / curr_fps).long() * curr_fps
        return motion_time

    def get_sampled_motion_state_by_env_id(self, env_ids, offset=None):
        motion_ids = self.sampled_motion_idx[env_ids]
        motion_times = self.sample_time_interval(motion_ids)
        motion_times = motion_times.clone()
        phase = (
            motion_times.to(self.sampled_frame_lens.device)
            / self.sampled_motion_lens[env_ids]
        )
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        # import pdb; pdb.set_trace()
        frames = (phase * (self.num_frames[motion_ids.to(self.device)] - 1)).long()
        sampled_frame_shift = (frames + self.frame_indices[env_ids]).long()

        root_pos = self.sampled_root_poses[sampled_frame_shift].to(self.device)

        root_rot = self.sampled_root_rots[sampled_frame_shift].to(self.device)
        dof_pos = self.sampled_dof_poses[sampled_frame_shift].to(self.device)
        body_pos = self.sampled_body_poses[sampled_frame_shift].to(self.device)
        body_rot = self.sampled_body_rots[sampled_frame_shift].to(self.device)
        root_vel = self.sampled_root_vels[sampled_frame_shift].to(self.device)
        body_vel = self.sampled_body_vels[sampled_frame_shift].to(self.device)
        root_angular_vel = self.sampled_root_angular_vels[sampled_frame_shift].to(
            self.device
        )
        body_angular_vel = self.sampled_body_angular_vels[sampled_frame_shift].to(
            self.device
        )
        dof_vel = self.sampled_dof_vels[sampled_frame_shift].to(self.device)

        if offset is not None:
            new_root_pos = root_pos + offset.to(self.device)
            root_pos = new_root_pos
        return {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "body_pos": body_pos,
            "body_rot": body_rot,
            "root_vel": root_vel,
            "body_vel": body_vel,
            "root_angular_vel": root_angular_vel,
            "body_angular_vel": body_angular_vel,
            "dof_vel": dof_vel,
        }

    def get_sampled_motion_state(self, motion_ids, motion_times, offset):
        motion_times = motion_times.clone()
        phase = (
            motion_times.to(self.sampled_frame_lens.device) / self.sampled_motion_lens
        )
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        # import pdb; pdb.set_trace()
        frames = (phase * (self.num_frames[motion_ids.to(self.device)] - 1)).long()
        sampled_frame_shift = (frames + self.frame_indices).long()

        root_pos = self.sampled_root_poses[sampled_frame_shift].to(self.device)

        root_rot = self.sampled_root_rots[sampled_frame_shift].to(self.device)
        dof_pos = self.sampled_dof_poses[sampled_frame_shift].to(self.device)
        body_pos = self.sampled_body_poses[sampled_frame_shift].to(self.device)
        body_rot = self.sampled_body_rots[sampled_frame_shift].to(self.device)
        root_vel = self.sampled_root_vels[sampled_frame_shift].to(self.device)
        body_vel = self.sampled_body_vels[sampled_frame_shift].to(self.device)
        root_angular_vel = self.sampled_root_angular_vels[sampled_frame_shift].to(
            self.device
        )
        body_angular_vel = self.sampled_body_angular_vels[sampled_frame_shift].to(
            self.device
        )
        dof_vel = self.sampled_dof_vels[sampled_frame_shift].to(self.device)

        if offset is not None:
            new_root_pos = root_pos + offset.to(self.device)
            root_pos = new_root_pos
        return {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "body_pos": body_pos,
            "body_rot": body_rot,
            "root_vel": root_vel,
            "body_vel": body_vel,
            "root_angular_vel": root_angular_vel,
            "body_angular_vel": body_angular_vel,
            "dof_vel": dof_vel,
        }
