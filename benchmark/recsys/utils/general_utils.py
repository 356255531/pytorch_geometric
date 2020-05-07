import os.path as osp
import torch
import os
import pickle


from torch_geometric.datasets import *


def get_folder_path(model, dataset):
    data_folder = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'checkpoint', 'data', dataset)
    weights_folder = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'checkpoint', 'weights', dataset, model)
    logger_folder = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'checkpoint', 'logger', dataset, model)
    data_folder = osp.expanduser(osp.normpath(data_folder))
    weights_folder = osp.expanduser(osp.normpath(weights_folder))
    logger_folder = osp.expanduser(osp.normpath(logger_folder))

    return data_folder, weights_folder, logger_folder


def save_model(file_path, model, optim, epoch, rec_metrics, silent=False):
    model_states = {'model': model.state_dict()}
    optim_states = {'optim': optim.state_dict()}
    states = {
        'epoch': epoch,
        'model_states': model_states,
        'optim_states': optim_states,
        'rec_metrics': rec_metrics
    }

    with open(file_path, mode='wb+') as f:
        torch.save(states, f)
    if not silent:
        print("Saved checkpoint_backup '{}'".format(file_path))


def load_model(file_path, model, optim, device):
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path, map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_states']['model'])
        optim.load_state_dict(checkpoint['optim_states']['optim'])
        rec_metrics = checkpoint['rec_metrics']
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Loaded checkpoint_backup '{}'".format(file_path))
    else:
        print("No checkpoint_backup found at '{}'".format(file_path))
        epoch = 0
        rec_metrics = [], [], [], [], []

    return model, optim, epoch, rec_metrics


def save_global_logger(
        global_logger_filepath,
        HR_per_run, NDCG_per_run, AUC_per_run,
        train_loss_per_run, eval_loss_per_run
):
    with open(global_logger_filepath, 'wb') as f:
        pickle.dump(
            [HR_per_run, NDCG_per_run, AUC_per_run, train_loss_per_run, eval_loss_per_run],
            f
        )


def load_global_logger(global_logger_filepath):
    if os.path.isfile(global_logger_filepath):
        with open(global_logger_filepath, 'rb') as f:
            HR_per_run, NDCG_per_run, ROC_per_run, train_loss_per_run, eval_loss_per_run = pickle.load(f)
    else:
        print("No logger found at '{}'".format(global_logger_filepath))
        HR_per_run, NDCG_per_run, ROC_per_run, train_loss_per_run, eval_loss_per_run = [], [], [], [], []

    return HR_per_run, NDCG_per_run, ROC_per_run, train_loss_per_run, eval_loss_per_run, len(HR_per_run)


def load_dataset(dataset_args):
    if dataset_args['dataset'] == 'Movielens':
        return MovieLens(**dataset_args)
    else:
        raise NotImplemented('Dataset not implemented!')
