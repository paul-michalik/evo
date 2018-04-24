import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
import io
import numpy as np
from evo.core.trajectory import PoseTrajectory3D,PosePath3D
from evo.tools import file_interface
from evo.tools.settings import SETTINGS

def parser():
    exp_parser = argparse.ArgumentParser(add_help=False)
    exp_parser.add_argument("--tum", 
                                help="tum trajectory files", nargs='+')
    exp_parser.add_argument("--kitti", 
                                help="kitti pose files", nargs='+')
    exp_parser.add_argument("--ref", 
                                help="kitti ground truth")                            
    exp_parser.add_argument("-p", "--plot", help="show plot window", action="store_true")
    exp_parser.add_argument("--plot_mode", help="the axes for  plot projection",
                             default=None, choices=["xy", "yx", "xz", "zx", "yz", "xyz"])
    exp_parser.add_argument("-s", "--scale", help="scale the trajectories", action="store_true")
    return exp_parser

def find_scale_factor(axtraj_1, axtraj_2):
    x_max_1 = axtraj_1[1][0][1]
    x_min_1 = axtraj_1[1][0][0]
    x_width_1 = x_max_1 - x_min_1

    y_max_1 = axtraj_1[1][1][1]
    y_min_1 = axtraj_1[1][1][0]
    y_width_1 = y_max_1 - y_min_1

    z_max_1 = axtraj_1[1][2][1]
    z_min_1 = axtraj_1[1][2][0]
    z_width_1 = z_max_1 - z_min_1

    x_max_2 = axtraj_2[1][0][1]
    x_min_2 = axtraj_2[1][0][0]
    x_width_2 = x_max_2 - x_min_2

    y_max_2 = axtraj_2[1][1][1]
    y_min_2 = axtraj_2[1][1][0]
    y_width_2 = y_max_2 - y_min_2

    z_max_2 = axtraj_2[1][2][1]
    z_min_2 = axtraj_2[1][2][0]
    z_width_2 = z_max_2 - z_min_2


    Sx = x_width_2 / x_width_1
    Sy = y_width_2 / y_width_1
    Sz = z_width_2 / z_width_1
    return (axtraj_1[0], (Sx, Sy, Sz))

def tum_scale_xyz(pose_traj, Sx, Sy, Sz):
    pos_xyz = pose_traj.positions_xyz
    pos_xyz[:,0] = pos_xyz[:,0] * Sx
    pos_xyz[:,1] = pos_xyz[:,1] * Sy
    pos_xyz[:,2] = pos_xyz[:,2] * Sz
    pos = PoseTrajectory3D(pos_xyz, pose_traj.orientations_quat_wxyz, pose_traj.timestamps)
    return pos

def kitti_scale_xyz(pose_path, Sx, Sy, Sz):
    pos_xyz = pose_path.positions_xyz
    pos_xyz[:,0] = pos_xyz[:,0] * Sx
    pos_xyz[:,1] = pos_xyz[:,1] * Sy
    pos_xyz[:,2] = pos_xyz[:,2] * Sz
    pos = PosePath3D(pos_xyz, pose_path.orientations_quat_wxyz)
    return pos

def scale(read_tum_traj,read_kitti_traj,traj_limits,ref_traj_limits=None):
    trajectories = []
    if ref_traj_limits:
        for limits in traj_limits:
            (traj_file, (Sx, Sy, Sz)) = find_scale_factor(limits, ref_traj_limits)
            if traj_file in read_tum_traj.keys():
                pos = tum_scale_xyz(read_tum_traj[traj_file], Sx, Sy, Sz)
                trajectories.append((traj_file,pos))
            elif traj_file in read_kitti_traj.keys():
                pos = kitti_scale_xyz(read_kitti_traj[traj_file], Sx, Sy, Sz)
                trajectories.append((traj_file,pos))
            else:
                pass 
    else:
        ref_traj = traj_limits[0]
        for limits in traj_limits:
            if (limits[1][0] >= ref_traj[1][0]) and (limits[1][1] >= ref_traj[1][1]) and (limits[1][2] >= ref_traj[1][2]):
                ref_traj = limits
        for limits in traj_limits:
            (traj_file, (Sx, Sy, Sz)) = find_scale_factor(limits, ref_traj)
            if traj_file in read_tum_traj.keys():
                pos = tum_scale_xyz(read_tum_traj[traj_file], Sx, Sy, Sz)
                trajectories.append((traj_file,pos))
            elif traj_file in read_kitti_traj.keys():
                pos = kitti_scale_xyz(read_kitti_traj[traj_file], Sx, Sy, Sz)
                trajectories.append((traj_file,pos))
            else:
                pass 
    return trajectories

def find_min_max(traj):
    import numpy as np
    x_lim = (np.max(traj.positions_xyz[:, 0]),np.min(traj.positions_xyz[:, 0]))
    y_lim = (np.max(traj.positions_xyz[:, 1]),np.min(traj.positions_xyz[:, 1]))
    z_lim = (np.max(traj.positions_xyz[:, 2]),np.min(traj.positions_xyz[:, 2]))
    return (x_lim,y_lim,z_lim)

def run(args):
    trajectories = []
    traj_limits = []
    read_tum_traj = {}
    read_kitti_traj = {}
    ref_traj_limits = None

    if args.tum and args.scale:
        for traj_file in args.tum:
            traj = file_interface.read_tum_trajectory_file(traj_file)
            (x_lim,y_lim,z_lim) = find_min_max(traj)
            traj_limits.append((traj_file,(x_lim,y_lim,z_lim)))
            read_tum_traj[traj_file] = traj
    elif args.tum and not args.scale:
        for traj_file in args.tum:
            trajectories.append((traj_file, file_interface.read_tum_trajectory_file(traj_file)))

    if args.kitti and args.scale:
        for pose_file in args.kitti:
            traj = file_interface.read_kitti_poses_file(pose_file)
            (x_lim,y_lim,z_lim) = find_min_max(traj)
            traj_limits.append((pose_file,(x_lim,y_lim,z_lim)))
            read_kitti_traj[pose_file] = traj
    elif args.kitti and not args.scale:
        for pose_file in args.kitti:
            trajectories.append((pose_file, file_interface.read_kitti_poses_file(pose_file)))    
    
    if args.ref and args.scale:
        ref_traj = file_interface.read_kitti_poses_file(args.ref)
        (x_lim,y_lim,z_lim) = find_min_max(ref_traj)
        ref_traj_limits = (args.ref,(x_lim,y_lim,z_lim)) 
    elif args.ref and not args.scale:
        ref_traj = file_interface.read_kitti_poses_file(args.ref)

    if args.scale:
        trajectories = scale(read_tum_traj,read_kitti_traj,traj_limits,ref_traj_limits)

    if args.plot:
        from evo.tools.plot import PlotMode
        plot_mode = PlotMode.xyz if not args.plot_mode else PlotMode[args.plot_mode]
        import numpy as np
        from evo.tools import plot
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import copy
        plot_collection = plot.PlotCollection("evo_traj - trajectory plot")
        fig_xyz, axarr_xyz = plt.subplots(3, sharex="col", figsize=tuple(SETTINGS.plot_figsize))
        fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=tuple(SETTINGS.plot_figsize))
        fig_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))
        ax_traj = plot.prepare_axis(fig_traj, plot_mode)

        cmap_colors = None
        for name, traj in trajectories:
            if cmap_colors is None:
                color = next(ax_traj._get_lines.prop_cycler)['color']
            else:
                color = next(cmap_colors)
            short_traj_name = os.path.splitext(os.path.basename(name))[0]
            if SETTINGS.plot_usetex:
                short_traj_name = short_traj_name.replace("_", "\\_")
            plot.traj(ax_traj, plot_mode, traj, '-', color, short_traj_name)
            start_time = None
            plot.traj_xyz(axarr_xyz, traj, '-', color, short_traj_name, start_timestamp=start_time)
            plot.traj_rpy(axarr_rpy, traj, '-', color, short_traj_name, start_timestamp=start_time)
        if args.ref:
            short_traj_name = os.path.splitext(os.path.basename(args.ref))[0]
            if SETTINGS.plot_usetex:
                short_traj_name = short_traj_name.replace("_", "\\_")
            plot.traj(ax_traj, plot_mode, ref_traj, '--', 'grey', short_traj_name)
            plot.traj_xyz(axarr_xyz, ref_traj, '--', 'grey', short_traj_name)
            plot.traj_rpy(axarr_rpy, ref_traj, '--', 'grey', short_traj_name,
                          alpha=0 if SETTINGS.plot_hideref else 1)

        plt.tight_layout()
        plot_collection.add_figure("trajectories", fig_traj)
        plot_collection.add_figure("xyz_view", fig_xyz)
        plot_collection.add_figure("rpy_view", fig_rpy)
        if args.plot:
            plot_collection.show()

if __name__ == '__main__':
    exp_parser = parser()
    args = exp_parser.parse_args()
    run(args)