from __future__ import print_function

import util.io as io
import sys
import os

def check_model(model_id, meas_key, reverse = 0):

    info = io.load_json(os.path.join('models', model_id, 'info.json'))

    # print train info
    print('Train')
    print(info['train_history'][-1])
    print('\n\n')

    # print last test
    print('Last Test')
    print(info['test_history'][-1])
    print('\n\n')

    # print best test
    print('Best Test')
    reverse = True if reverse == 1 else False
    info['test_history'].sort(key = lambda x:x[meas_key], reverse = reverse)
    print(info['test_history'][0])
    print('\n\n')


if __name__ == '__main__':

    model_id = sys.argv[1]
    if len(sys.argv) >=3:
        meas_key = sys.argv[2]
    else:
        meas_key = 'mae'

    if len(sys.argv) >= 4:
        reverse = int(sys.argv[3])
    else:
        reverse = 0

    check_model(model_id, meas_key, reverse)

    # print(sys.argv)
