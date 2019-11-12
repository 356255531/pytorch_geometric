import numpy as np

from .train_eval import single_run_with_kg


def run_with_kg(model_class, model_args, dataset_args, train_args):
    kg_train_losses = []
    cf_train_losses = []
    kg_val_losses = []
    cf_val_losses = []
    for run in range(1, train_args['runs'] + 1):
        single_run_kg_train_loss, single_run_cf_train_loss, single_run_best_kg_val_loss, single_run_best_cf_val_loss = \
            single_run_with_kg(run, model_class, model_args, dataset_args, train_args)
        kg_train_losses.append(single_run_kg_train_loss)
        cf_train_losses.append(single_run_cf_train_loss)
        kg_val_losses.append(single_run_best_kg_val_loss)
        cf_val_losses.append(single_run_best_cf_val_loss)
    mean_train_kg_loss, best_train_kg_loss = np.mean(kg_train_losses), np.min(kg_train_losses)
    mean_train_cf_loss, best_train_cf_loss = np.mean(cf_train_losses), np.min(cf_train_losses)
    mean_val_kg_loss, best_val_kg_loss = np.mean(kg_val_losses), np.min(kg_val_losses)
    mean_val_cf_loss, best_val_cf_loss = np.mean(cf_val_losses), np.min(cf_val_losses)
    print('mean_train_kg_loss: {}, best_train_kg_loss: {}'.format(mean_train_kg_loss, best_train_kg_loss))
    print('mean_train_cf_loss: {}, best_train_cf_loss: {}'.format(mean_train_cf_loss, best_train_cf_loss))
    print('mean_val_kg_loss: {}, best_val_kg_loss: {}'.format(mean_val_kg_loss, best_val_kg_loss))
    print('mean_val_cf_loss: {}, best_val_cf_loss: {}'.format(mean_val_cf_loss, best_val_cf_loss))
