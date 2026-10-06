import common.tool as tool
import run_main

RUNNING_FLAG = None

def main(model_name, dataset_name, parameter_dict):
    if ['GRU4Rec', 'SASRec', 'DuoRec'].__contains__(model_name):
        if ['yelp', 'ml-1m'].__contains__(dataset_name):
            parameter_dict['attn_dropout_prob'] = 0.1
            parameter_dict['hidden_dropout_prob'] = 0.1
    run_main.main(model_name, dataset_name, parameter_dict)

def process_base(model_name_arr, dataset_name_arr, parameter_dict):
    # param
    # config_file = None  # None/base
    parameter_dicts = {
        'CL4SRec': {
            # 'train_batch_size': 1024,
            # 'eval_batch_size': 256,
            'train_neg_sample_args': None,
            'tau': 1.0,
            'sim': 'dot',
            'lmd': 0.01,
        },
        'DCRec': {
            # "train_batch_size": 1024,
            # "eval_batch_size": 256,
            'train_neg_sample_args': None,

            "hidden_dropout_prob": 0.3,
            "attn_dropout_prob": 0.3,
            # Graph Args:
            "graph_dropout_prob": 0.3,
            "graphcl_enable": 1,
            "graphcl_coefficient": 1e-4,
            "cl_ablation": 'full',
            "graph_view_fusion": 1,
            "cl_temp": 1,
            # "save_dataloaders": False,
        },
        'KGAT': {
            'load_col': {
                'inter': ['user_id', 'item_id', 'rating'],
                'kg': ['head_id', 'relation_id', 'tail_id'],
                'link': ['item_id', 'entity_id']
            }
        }
    }
    for dataset_name in dataset_name_arr:
        for model_name in model_name_arr:
            if parameter_dicts.__contains__(model_name):
                config = parameter_dicts[model_name]
            elif parameter_dicts.__contains__('Other'):
                config = parameter_dicts[model_name]
            else:
                config = {}
            tool.tranfer_dict(parameter_dict, config)
            custom_parameter(model_name, dataset_name, parameter_dict)
            try:
                main(model_name, dataset_name, parameter_dict)
            except Exception as e:
                # raise e
                print('e1:', e)


def custom_parameter(model_name, dataset_name, parameter_dict):
    if ['DCRec'].__contains__(model_name):
        # BEST SETTINGS
        if dataset_name == "reddit":
            parameter_dict["train_batch_size"] = 128
            parameter_dict["graphcl_coefficient"] = 1
            parameter_dict["weight_mean"] = 0.5
            parameter_dict["sim_group"] = 4
            parameter_dict["kl_weight"] = 1
        else:
            parameter_dict["graphcl_coefficient"] = 1e-1
            parameter_dict["graph_dropout_prob"] = 0.5
            parameter_dict["hidden_dropout_prob"] = 0.5
            parameter_dict["attn_dropout_prob"] = 0.5
            parameter_dict["kl_weight"] = 1e-2

            if ["beauty", "ml-1m1", "steam", "Amazon_Books", "lfm1b-tracks", "Amazon_Electronics"].__contains__(dataset_name):
                parameter_dict["schedule_step"] = 30
                parameter_dict["attn_dropout_prob"] = 0.1
                parameter_dict["sim_group"] = 4
                parameter_dict["weight_mean"] = 0.5
                parameter_dict["cl_temp"] = 1
            elif ["sports"].__contains__(dataset_name):
                parameter_dict["attn_dropout_prob"] = 0.3
                parameter_dict["sim_group"] = 4
                parameter_dict["weight_mean"] = 0.5
                parameter_dict["cl_temp"] = 1
            elif ["ml-20m", 'yelp', 'yelp1'].__contains__(dataset_name):
                parameter_dict["sim_group"] = 4
                parameter_dict["weight_mean"] = 0.4
                parameter_dict["cl_temp"] = 0.8
    else:
        if ['yelp'].__contains__(dataset_name):
            parameter_dict["eval_batch_size"] = 128


if __name__ == '__main__':
    parameter_dict = {
        'epochs': 1,
        'train_batch_size': 1024, # 1024,4096, PS: BERT4Rec,lfm1b-artists 128/32
        'eval_batch_size': 256, # 256,512
        'gpu_id': '0',  # (str) The id of GPU device(s).
        'idx': '0',  # 标记序号，不同数字则创建不同目录保存预处理数据集
        'checkpoint_dir': 'saved/dataset',  # saved/dataset
        # 'neg_sampling': None,
        # 'train_neg_sample_args': None,
    }

    ############## other #####################
    # model_name_arr = ['BPR']

    ############## base(源码运行) #####################
    ## CoSeRec,ICLRec

    ############## base #####################0
    # GNN:LightGCN, KG:KGAT, SR:SASRec, CL:SGL
    # model_name_arr = ['KGAT','LightGCN','Caser','GRU4Rec','SRGNN','GCSAN','SASRec','BERT4Rec','DuoRec','CL4SRec','DCRec','SGL']

    model_name_arr = ['SASRec']  # ['BPR','SASRec','BERT4Rec','GCSAN','SRGNN','Caser','CORE','CL4SRec','LightSANs','GRU4Rec']
    dataset_name_arr = ['gowalla']  # ['steam','netflix','RentTheRunway','movielens','lfm1b-artists','mind','lfm1b-tracks','ml-10m']
    process_base(model_name_arr, dataset_name_arr, parameter_dict)
