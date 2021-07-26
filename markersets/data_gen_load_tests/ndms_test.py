from data_gen.ndms_datagenerator import NDMSDataGenerator
from markersets.trajsets.ndmstrajset import NDMSTrajSet
from utility.ndms_datagen_visu import animate_traj, plot_traj_slider
from data_gen.preproc import norm_emg
import os


def main():
    data_dir = r'C:\Users\hbkm9\Documents\School\PhD Year 1\AIRDG\traj\AIFDG_01\Dataset\Training'
    val_dir = r'C:\Users\hbkm9\Documents\School\PhD Year 1\AIRDG\traj\AIFDG_01\Dataset\Validation'

    def target_files(f):
        return 'Angular' in f or 'Freeform' in f

    gen_params = {
        'data_dir': data_dir,
        ###################
        'window_size': 1000,
        'delay': 1000,
        'gap_windows': None,
        ###################
        'stride': 20,
        'freq_factor': 20,
        'file_names': sorted([f for f in os.listdir(data_dir)
                              if target_files(f)]),
        'feature_names': ('t_h_x', 't_h_y'),
        'channel_mask': [f'Sensor {i}.EMG{i}' for i in range(1, 9)],
        'dims': range(19, 60, 20),
        # 'dims': list(range(1,6,2)),
        'time_step': 1,
        ###############################################################################
        'preproc': norm_emg,
        ###############################################################################
        'batch_size': 64,
        ###############################################################################
    }

    tcn_generator = NDMSDataGenerator(**gen_params)
    plot_traj_slider(tcn_generator, 3)


def animate_mocap():
    test_file = r'missing'

    ts = NDMSTrajSet(test_file,
                     horizon_vector=tuple(range(20, 61, 20)))
    ts.se_proc_traj()
    ts.show_global(save=False, traj_kwargs={'alpha': 0.3})


if __name__ == '__main__':
    main()
