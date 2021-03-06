import json
import numpy as np
import os
import re
import sys


def fix_stack(s):
    import os

    command = 'echo | set /p nul=' + s.replace('\n', ',').replace(' ', ',').replace(',,', ',').replace(',,', ',').\
        replace(',,', ',').replace('[,', '[').strip() + '| clip'
    os.system(command)
    return


def get_min_loss(s):
    import pickle
    import os
    import numpy as np
    with open(s, "rb") as input_file:
        h = pickle.load(input_file)
    text = str([np.min(d['val_loss']) for d in h])
    command = 'echo | set /p nul=' + text.strip() + '| clip'
    os.system(command)


# Sorry if you ever have to port these to linux! (just change to os.path.join())
def document_model(dir_path, end, model, history, **kwargs):
    os.makedirs(dir_path, exist_ok=True)
    model.save(os.path.join(dir_path, 'model_' + str(end) +'.h5'))
    import pickle
    with open(os.path.join(dir_path, 'history_' + str(end) + '.pickle'), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    str_sum = model_summary_to_string(model) + '\n'
    str_sum += kw_summary(**kwargs)
    print_to_file(str_sum, os.path.join(dir_path, 'summary_' + str(end) + '.txt'))


def model_summary_to_string(model):
    str_list = []
    model.summary(print_fn=lambda x: str_list.append(x))
    return "\n".join(str_list)


def print_to_file(str_in, filepath):
    with open(filepath, 'w+') as f:
        f.write(str_in)


def kw_summary(**kwargs):
    str_out = str()
    for key, item in kwargs.items():
        str_out += key + ': ' + str(item) + '\n'
    return str_out


def summary(k, scores, kernel, drop, model, data_path, epochs, batch, files):
    print(scores)
    m, st = np.mean(scores), np.std(scores)

    print('MSE: {0:.3f} (+/-{1:.3f})'.format(-m, st))
    print('K-fold: {0:.0f}'.format(k))

    # region Self-documentation

    file_name = incr_file(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries', r'model_summary', '.txt')

    ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
            for f in os.listdir(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries') if f.endswith('.txt')]
    if not ends:
        ends = [0]
    print(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries\model_summary' +
          str(max(ends) + 1) + '.txt')
    from datetime import datetime
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries\model_summary' +
              str(max(ends) + 1) + '.txt', 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\nDate: {}\n"
                "File: {}\n"
                "Scores: {}\n"
                "MSE: {:.3f} (+/-{:.3f})\n"
                "Dropout (if applicable): {:.3f}\n"
                "Kernel (if applicable): {:.0f}, {:.0f}\n"
                "Epochs: {}, Batch size: {}\n"
                "K: {:.0f}\n".format(dt_string, os.path.basename(sys.argv[0]),
                                     scores, m, st, drop, kernel[0], kernel[1], epochs, batch, k)
                + "\n\nUsing Files:\n")
        for file in files:
            f.write(file + "\n")


def incr_file(dir_path, file_name, ext):
    ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
            for f in os.listdir(dir_path) if f.endswith(ext)
            and file_name in f]
    if not ends:
        ends = [0]
    return dir_path + '\\', file_name + str(max(ends) + 1) + ext, ends


def incr_dir(dir_path, dir_name, make=True):
    ends = [int(re.search(r'(\d+)', d).group(0)) for d in next(os.walk(dir_path))[1] if dir_name in d]
    if not ends:
        ends = [0]
    new_dir = dir_path + '\\' + dir_name + str(max(ends) + 1)
    if make:
        os.makedirs(new_dir)
    return new_dir, ends


def get_file_names(dir_path, task=None):
    return [dir_path + '\\' + file for file in sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
            if task is None or task in file]


def load_dict(file_path):
    with open(file_path) as json_file:
        dict_data = json.load(json_file)
    return dict_data


def load_dict_stack(path, task='None'):
    dict_stack = list()

    def f_check(f):
        return np.any([n in f for n in task or task is 'None'])
    for file in sorted([f for f in os.listdir(path) if f.endswith('.json') and f_check(f)]):
        with open(path + '\\' + file) as json_file:
            dict_data = json.load(json_file)
            dict_stack.append(dict_data)
    return dict_stack


def save_dict(file_path, dict_in):
    with open(file_path, 'w') as fp:
        json.dump(dict_in, fp, indent=4)
    return


def load_emg_stack(path, task='None', n_channels=8):
    emg_stack = list()
    def f_check(f):
        return np.any([n in f for n in task or task is 'None'])

    for file in sorted([f for f in os.listdir(path) if f.endswith('.json') and f_check(f)]):
        with open(path + '\\' + file) as json_file:
            dict_data = json.load(json_file)
            emg_stack.append(np.array(dict_data["EMG"]))
    return emg_stack


def load_emg(path, task=None, n_channels=8):
    X = np.empty((n_channels, 0))
    if os.path.isdir(path):
        for file in sorted([f for f in os.listdir(path) if f.endswith('.json')]):
            if task not in file and task is not None:
                continue
            with open(path + '\\' + file) as json_file:
                dict_data = json.load(json_file)
                X = np.concatenate((X, dict_data["EMG"]), axis=1)
        return X
    with open(path) as json_file:
        dict_data = json.load(json_file)
    return np.array(dict_data["EMG"])
