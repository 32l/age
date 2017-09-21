import util.io as io
from util.pavi import PaviClient

import modules.opt_parser as opt_parser
import time



def upload_training_history():

    # load info
    model_info = io.load_json('models/joint_1.1/info.json')

    # create pavi client

    info = {
        'session_text': opt_parser.opts_to_string([('model_opts', model_info['opts']),
            ('train_opts', model_info['train_opts'])]),
    }

    pavi = PaviClient(username = 'ly015', password = '123456')
    pavi.connect(model_name = 'joint_1.1', info = info)


    # upload data

    curve_name_lst = ['loss_age', 'loss_pose', 'loss_attr']

    print('uploading training history...')
    for th in model_info['train_history'][0::10]:
        iter_num = th['iteration']
        outputs = {n:th[n] for n in curve_name_lst}
        pavi.log(phase = 'train', iter_num = iter_num, outputs = outputs)
        print(iter_num)

    print('uploading test history...')
    for th in model_info['test_history']:
        iter_num = th['iteration']
        outputs = {n:th[n] for n in curve_name_lst}
        pavi.log(phase = 'test', iter_num = iter_num, outputs = outputs)
        print(iter_num)

    time.sleep(1000)

if __name__ == '__main__':
    upload_training_history()

