import torch
from torch.optim import Adam

from .train_eval import single_run_with_kg


def run_with_kg(model_class, loss_func, dataset_args, model_args, train_args):
    model.to(train_args['device']).reset_parameters()

    loss_func = torch.nn.MSELoss()
    opt = Adam(model.parameters(), lr=train_args['lr'], weight_decay=train_args['weight_decay'])

    best_kg_train_loss = float('inf')
    best_cf_train_loss = float('inf')
    best_kg_val_loss = float('inf')
    best_cf_val_loss = float('inf')
    for run in range(1, train_args['runs'] + 1):
        single_run_kg_train_loss, single_run_cf_train_loss, single_run_best_kg_val_loss, single_run_best_cf_val_loss = \
            single_run_with_kg(run, model, opt, loss_func, dataset_args, train_args)
        best_kg_train_loss = \
            single_run_kg_train_loss if single_run_kg_train_loss < best_kg_train_loss else best_kg_train_loss
        best_cf_train_loss = \
            single_run_cf_train_loss if single_run_cf_train_loss < best_cf_train_loss else best_cf_train_loss
        best_kg_val_loss = \
            single_run_best_kg_val_loss if single_run_best_kg_val_loss < best_kg_val_loss else best_kg_val_loss
        best_cf_val_loss = \
            single_run_best_cf_val_loss if single_run_best_cf_val_loss < best_cf_val_loss else best_cf_val_loss