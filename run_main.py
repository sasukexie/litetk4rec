import argparse

from recbole.quick_start import run
from datetime import datetime
import common.tool as tool
import platform
import multiprocessing as mp

RUNNING_FLAG = None


def main(model_name, dataset_name, parameter_dict, config_file=None):
    # 1.set param
    parser = argparse.ArgumentParser()
    # set model
    parser.add_argument('--model', '-m', type=str, default=model_name, help='name of models')
    # set datasets # ml-1m,ml-20m,amazon-books,lfm1b-tracks
    parser.add_argument('--dataset', '-d', type=str, default=dataset_name, help='name of datasets')
    # set config
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    # get param
    args, _ = parser.parse_known_args()
    # config list
    config_file_list = ['zone/common.yaml']

    if 'checkpoint_dir_t' not in parameter_dict:
        parameter_dict['checkpoint_dir_t'] = parameter_dict['checkpoint_dir']
    parameter_dict['checkpoint_dir'] = f'{parameter_dict['checkpoint_dir_t']}/{parameter_dict["train_batch_size"]}_{parameter_dict["eval_batch_size"]}_{parameter_dict["idx"]}'

    if config_file:
        config_file_list.append(f'zone/{config_file}.yaml')

    global RUNNING_FLAG
    RUNNING_FLAG = f'RF{datetime.now().strftime("%Y%m%d%H%M%S")}' if RUNNING_FLAG == None else RUNNING_FLAG
    parameter_dict['running_flag'] = RUNNING_FLAG
    system_name = platform.system()
    if system_name == 'Windows':
        parameter_dict['gpu_id'] = '0'
    elif system_name == 'Linux':
        pass

    # 设置多任务
    nproc = 1
    world_size = -1
    # nproc = torch.cuda.device_count()
    # gpu_id = ''
    # if nproc>1:
    #     world_size = nproc*2
    #     for i in range(nproc):
    #         gpu_id += f'{i},'
    # parameter_dict['gpu_id'] = gpu_id

    # 2.call recbole_trm: config,dataset,model,trainer,training,evaluation

    print("###独立线程训练开始###")
    # run(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict, nproc=nproc, world_size=world_size)
    # 创建新进程
    p = mp.Process(target=run, args=(args.model, args.dataset, config_file_list, parameter_dict, True, nproc, world_size))
    p.start()
    p.join()  # 等待完成
    # tool.clear_cache()
    print("###独立线程训练结束###")


def process(parameter_dict, dataset_name_arr):
    # param
    # set model
    model_name = parameter_dict['model_name']
    parameter_dict1 = {
        # 'n_heads': 2, # 2
        # 'n_layers': 2, # 2
        # 'loss_type': 'CE', # CE,BPR
        #
        # 'use_temporal_encoder': True,  # [True,False]
        # 'use_cosine_similarity': True,  # [True,False]
        #
        # 'use_cl': True, # [True,False]
        # 'cl_weight': 0.1,
        # 'cl_temperature': 0.2,
        # 'cl_mask_ratio': 0.2,         # 掩码比例
        #
        # # mamba
        # 'd_state': 16, # [8,16,32,48,64]
        # 'd_conv': 4, # [2,4,6,8]
        # 'expand': 2,  # [1,2,3,4]
    }

    tool.tranfer_dict(parameter_dict, parameter_dict1)

    system_name = platform.system()
    if system_name == 'Windows':
        print("This is a Windows System")
    elif system_name == 'Linux':
        print("This is a Linux System")

    # movielens: reg_weight:1e-05,cl_weight:0.5,dropout:0.1,n_heads:2,n_layers:4
    dropouts = [0.1] #[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    n_layerses = [2] #[1,2,4,6,8]
    n_headses = [2] #[1,2,4,6,8]
    max_lens = [50] #[50,60,100,200]
    for dataset_name in dataset_name_arr:
        for dropout in dropouts:
            for n_layers in n_layerses:
                for n_heads in n_headses:
                    for max_len in max_lens:
                        parameter_dict['hidden_dropout_prob'] = dropout
                        parameter_dict['attn_dropout_prob'] = dropout
                        parameter_dict['n_layers'] = n_layers
                        parameter_dict['n_heads'] = n_heads
                        parameter_dict['MAX_ITEM_LIST_LENGTH'] = max_len
                        main(model_name, dataset_name, parameter_dict)


# Motivation:
if __name__ == '__main__':
    parameter_dict = {
        'model_name': 'LiteTK4Rec',   # 本文模型
        'epochs': 50,
        'use_cl': False,   # 关闭对比式自监督（本文 §4.8 已报告其无效）
        'train_batch_size': 2048,
        'eval_batch_size': 512, # 256,512
        'stopping_step': 10, # 256,512
        # 'learning_rate': 0.001, # 0.001,0.0005
        'checkpoint_dir': 'saved/dataset', # saved/dataset
        'gpu_id': '0',  # (str) The id of GPU device(s).
        'idx': '0',  # 标记序号，不同数字则创建不同目录保存预处理数据集
    }
    # param
    # set model # MODEL,SimDCL,SASRec,BERT4Rec,BPR,GRU4RecF
    # set datasets # ['steam','lfm1b-tracks','ml-1m']
    # process_base()

    # model & dataset # movielens,RentTheRunway,netflix,lfm1b-tracks,ml-100k
    # ,'movielens','RentTheRunway','steam'
    dataset_name_arr = ['ml-3m']  # ['ml-100k','lfm1b-tracks','steam','netflix','RentTheRunway','movielens','mind','ml-3m','ml-10m']
    process(parameter_dict, dataset_name_arr)

