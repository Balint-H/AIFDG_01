from data_gen.datagenerator import TCNDataGenerator
from data_gen.ndms_datagenerator import NDMSDataGenerator
from data_gen.preproc import norm_emg
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ndms_architectures import depthwise_model_ndms, depthwise_model_ndms_encode
from tcn_predict import plot_history, ndms_plot_pred
from utility.save_load_util import document_model, incr_dir



def main():
    data_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment3\Processed\NDMS01_Processed\Data'
    val_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment3\Processed\NDMS01_Processed\Validation'

    target_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\Models'
    child_dir = 'model_'
    new_dir, ends = incr_dir(target_dir, child_dir, make=True)

    def target_files(f):
        return 'Angular' in f or 'Freeform' in f or 'Circle' in f or 'Arc' in f or 'Slalom' in f

    def valid_target_file(f):
        return 'Freeform' in f

    gen_params = {
        'data_dir': data_dir,
        ###################
        'window_size': 1000,
        'delay': 0,
        'gap_windows': None,
        ###################
        'stride': 60,
        'freq_factor': 20,
        'file_names': sorted([f for f in os.listdir(data_dir)
                              if target_files(f)]),
        'feature_names': ('t_h_x', 't_h_y'),
        'channel_mask': [f'Sensor {i}.EMG{i}' for i in range(1, 9)],
        # 'dims': range(60),
        'dims': range(19, 60, 20),
        # 'dims': list(range(1,6,2)),
        'time_step': 1,
        'shuffle': True,
        ###############################################################################
        'preproc': norm_emg,
        ###############################################################################
        'batch_size': 64,
        ###############################################################################
    }

    tcn_generator = NDMSDataGenerator(**gen_params)
    # plot_traj_slider(tcn_generator, 3)
    # animate_traj(tcn_generator, 3)

    model_params = {
        'input_shape': tcn_generator.data_generation([0])[0][0].shape if gen_params['gap_windows'] is None else
        [temp[0].shape for temp in tcn_generator.data_generation([0])[0]],
        'output_shape': tcn_generator.data_generation([0])[1][0].shape,
        'acts': ('relu', 'selu', 'relu'),
        'krnl_in': ((1, 15), (1, 3)),  # (feature, time)
        'pad': 'same',
        'dil': ((1, 1), (1, 10)),
        'strides': ((1, 1), (1, 1)),
        'mpool': ((1, 6), (1, 5)),  # (feature, time)
        'depth_mul_in': (5, 4),
        'l_norm': False,

        'feature_conv': None,

        'drp': 0.5,
        'dense_drp': True,
        'dense': (1000, 200, 200, 100, 50),
        'b_norm': False,

        # 'krnl_out': (3, 5),
        # 'filters_out': (8, 8),
    }

    model = depthwise_model_ndms_encode(**model_params)
    model.summary()

    val_params = dict(gen_params)
    val_params['data_dir'] = val_dir
    val_params['file_names'] = sorted([f for f in os.listdir(val_dir) if f.endswith('.json') and valid_target_file(f)])
    val_generator = NDMSDataGenerator(**val_params)


    # region Train model on dataset
    callback_params = {
        'patience': 10
    }

    cb = callback_gen(dir_path=new_dir, end=max(ends) + 1, **callback_params)

    train_params = {
        'max_queue_size': 64,
        'workers': 6,
        'verbose': 1,
        'epochs': 200
    }

    history = model.fit(tcn_generator, validation_data=val_generator, callbacks=cb, **train_params)
    # endregion

    # ---------------------------------------------------------------------------------------------------------------- #

    emg_data, Y0 = val_generator.data_generation(range(val_generator.n_windows - 1))
    ndms_plot_pred(emg_data=emg_data, Y0=Y0, data_dir=new_dir, model=model)

    # region Save and document
    val_generator.save(new_dir + '\\valid_gen_' + str(max(ends) + 1) + '.pickle', unload=True)
    tcn_generator.save(new_dir + '\\train_gen_' + str(max(ends) + 1) + '.pickle', unload=True)
    document_model(new_dir, max(ends) + 1, model, history.history,
                   **{**gen_params, **model_params, **train_params, **callback_params},
                   datagenerator=type(tcn_generator))
    plot_history(history.history, new_dir)

    return


def callback_gen(dir_path, end, patience=8, verbose=(1, 1)):
    mc = ModelCheckpoint(dir_path + '\\best_model_' + str(end) +
                         '.h5', monitor='val_loss', mode='min', verbose=verbose[0], save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose[1], patience=patience)
    return [mc, es]

if __name__ == '__main__':
    main()
