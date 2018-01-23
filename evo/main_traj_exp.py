import argparse
import os
import io
import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.tools.settings import SETTINGS

def parser():
    exp_parser = argparse.ArgumentParser(add_help=False)
    exp_parser.add_argument("--tum", 
                                help="tum trajectory files", nargs='+')
    exp_parser.add_argument("--kitti", 
                                help="kitti pose files", nargs='+')
    exp_parser.add_argument("-p", "--plot", help="show plot window", action="store_true")
    exp_parser.add_argument("--plot_mode", help="the axes for  plot projection",
                             default=None, choices=["xy", "yx", "xz", "zx", "yz", "xyz"])
    return exp_parser

def run(args):
    trajectories = []
    #my code
    if args.tum:
        for traj_file in args.tum:
            trajectories.append((traj_file, file_interface.read_tum_trajectory_file(traj_file)))
    if args.kitti:
        for pose_file in args.kitti:
            trajectories.append((pose_file, file_interface.read_kitti_poses_file(pose_file)))     
    #my code ends
    if args.plot:
        from evo.tools.plot import PlotMode
        plot_mode = PlotMode.xyz if not args.plot_mode else PlotMode[args.plot_mode]
        import numpy as np
        from evo.tools import plot
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        plot_collection = plot.PlotCollection("evo_traj - trajectory plot")
        fig_xyz, axarr_xyz = plt.subplots(3, sharex="col", figsize=tuple(SETTINGS.plot_figsize))
        fig_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))
        ax_traj = plot.prepare_axis(fig_traj, plot_mode)
        cmap_colors = None
        if SETTINGS.plot_multi_cmap.lower() != "none":
            cmap = getattr(cm, SETTINGS.plot_multi_cmap)
            cmap_colors = iter(cmap(np.linspace(0, 1, len(trajectories))))
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
        plt.tight_layout()
        plot_collection.add_figure("trajectories", fig_traj)
        plot_collection.add_figure("xyz_view", fig_xyz)
        if args.plot:
            plot_collection.show()

if __name__ == '__main__':
    exp_parser = parser()
    args = exp_parser.parse_args()
    run(args)