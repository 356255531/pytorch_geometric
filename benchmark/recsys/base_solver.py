from abc import abstractmethod
import os
import random as rd
import numpy as np
import torch
import time
import pandas as pd
import tqdm

from utils import *


class BaseSolver(object):
    def __init__(self, model_class, dataset_args, model_args, train_args, rec_args):
        self.model_class = model_class
        self.dataset_args = dataset_args
        self.model_args = model_args
        self.train_args = train_args
        self.rec_args = rec_args

        self.init_eval = rec_args['init_eval']

    @abstractmethod
    def prepare_model_input(self, data):
        pass

    @abstractmethod
    def train_negative_sampling(self, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid):
        pass

    @abstractmethod
    def eval_sampling(self, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid):
        pass

    def metrics(
            self,
            run,
            epoch,
            propagated_node_emb,
            train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map,
            rec_args):
        HR, NDCG, AUC, eval_losses = [], [], [], []

        u_nids = list(test_pos_unid_inid_map.keys())
        test_bar = tqdm.tqdm(u_nids, total=len(u_nids))
        for u_idx, u_nid in enumerate(test_bar):
            pos_i_nids, neg_i_nids = self.eval_sampling(train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid)
            if len(pos_i_nids) == 0 or len(neg_i_nids) == 0:
                continue
            pos_i_nid_df = pd.DataFrame({'u_nid': [u_nid for _ in range(len(pos_i_nids))], 'pos_i_nid': pos_i_nids})
            neg_i_nid_df = pd.DataFrame({'u_nid': [u_nid for _ in range(len(neg_i_nids))], 'neg_i_nid': neg_i_nids})
            pos_neg_pair_np = pd.merge(pos_i_nid_df, neg_i_nid_df, how='inner', on='u_nid').to_numpy()

            u_node_emb = propagated_node_emb[pos_neg_pair_np[:, 0]]
            pos_i_node_emb = propagated_node_emb[pos_neg_pair_np[:, 1]]
            neg_i_node_emb = propagated_node_emb[pos_neg_pair_np[:, 2]]
            pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
            pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)

            loss = - (pred_pos - pred_neg).sigmoid().log().mean().item()

            u_node_emb = propagated_node_emb[u_nid]
            pos_i_node_emb = propagated_node_emb[pos_i_nids]
            neg_i_node_emb = propagated_node_emb[neg_i_nids]
            pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
            pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)

            _, indices = torch.topk(torch.cat([pred_pos, pred_neg]), rec_args['num_recs'])
            hit_vec = (indices < len(pos_i_nids)).cpu().detach().numpy()

            HR.append(hit(hit_vec))
            NDCG.append(ndcg(hit_vec))
            AUC.append(auc(hit_vec))
            eval_losses.append(loss)
            test_bar.set_description(
                'Run {}, epoch: {}, HR: {:.4f}, NDCG: {:.4f}, AUC: {:.4f}, eval loss: {:.4f}, '.format(run, epoch,
                                                                                   np.mean(HR), np.mean(NDCG), np.mean(AUC), np.mean(eval_losses)))

        return np.mean(HR), np.mean(NDCG), np.mean(AUC), np.mean(eval_losses)

    def run(self):
        global_logger_path = self.train_args['logger_folder']
        if not os.path.exists(global_logger_path):
            os.makedirs(global_logger_path, exist_ok=True)
        global_logger_file_path = os.path.join(global_logger_path, 'global_logger.pkl')
        HR_per_run, NDCG_per_run, AUC_per_run, train_loss_per_run, eval_loss_per_run, last_run = \
            load_global_logger(global_logger_file_path)

        start_run = last_run + 1
        if start_run <= self.train_args['runs']:
            for run in range(start_run, self.train_args['runs'] + 1):
                # Fix the random seed
                seed = 2019 + run
                rd.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                # Create the dataset
                self.dataset_args['seed'] = seed
                dataset = load_dataset(self.dataset_args)
                data = dataset.data.to(self.train_args['device'])
                train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = \
                    data.train_pos_unid_inid_map[0], data.test_pos_unid_inid_map[0], data.neg_unid_inid_map[0]
                model_input = self.prepare_model_input(data)

                # Create model and optimizer
                self.model_args['emb_dim'] = data.num_node_types
                self.model_args['num_nodes'] = data.x.shape[0]
                model = self.model_class(**self.model_args).to(self.train_args['device'])

                optimizer = torch.optim.Adam(
                    params=model.parameters(),
                    lr=self.train_args['lr'],
                    weight_decay=self.train_args['weight_decay']
                )

                # Load models
                weights_path = os.path.join(self.train_args['weights_folder'], 'run_{}'.format(str(run)))
                if not os.path.exists(weights_path):
                    os.makedirs(weights_path, exist_ok=True)
                weights_file = os.path.join(weights_path, 'latest.pkl')
                model, optimizer, last_epoch, rec_metrics = load_model(weights_file, model, optimizer,
                                                                       self.train_args['device'])
                HR_per_epoch, NDCG_per_epoch, AUC_per_epoch, train_loss_per_epoch, eval_loss_per_epoch = \
                    rec_metrics if rec_metrics is not None else ([], [], [], [], [])

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_epoch = last_epoch + 1
                if start_epoch == 1 and self.init_eval:
                    HR_before, NDCG_before, AUC_before, eval_loss_before = self.metrics(
                        run,
                        start_epoch,
                        model(**model_input),
                        train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map,
                        self.rec_args
                    )
                    print(
                        'Initial performance HR: {:.4f}, NDCG: {:.4f}, '
                        'AUC: {:.4f}, eval loss: {:.4f}'.format(
                            HR_before, NDCG_before, AUC_before, eval_loss_before
                        )
                    )

                t_start = time.perf_counter()
                if start_epoch <= self.train_args['epochs']:
                    # Start training model
                    for epoch in range(start_epoch, self.train_args['epochs'] + 1):
                        loss_per_batch = []

                        model.train()
                        u_nids = list(train_pos_unid_inid_map.keys())
                        rd.shuffle(u_nids)
                        train_bar = tqdm.tqdm(u_nids, total=len(u_nids))
                        for u_idx, u_nid in enumerate(train_bar):
                            pos_i_nids = train_pos_unid_inid_map[u_nid]
                            neg_i_nids = self.train_negative_sampling(train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid)

                            if len(pos_i_nids) == 0 or len(neg_i_nids) == 0:
                                continue

                            pos_i_nid_df = pd.DataFrame(
                                {'u_nid': [u_nid for _ in range(len(pos_i_nids))], 'pos_i_nid': pos_i_nids})
                            neg_i_nid_df = pd.DataFrame(
                                {'u_nid': [u_nid for _ in range(len(neg_i_nids))], 'neg_i_nid': neg_i_nids})
                            pos_neg_pair_np = pd.merge(pos_i_nid_df, neg_i_nid_df, how='inner', on='u_nid').to_numpy()

                            propagated_node_emb = model(**model_input)

                            u_node_emb = propagated_node_emb[pos_neg_pair_np[:, 0]]
                            pos_i_node_emb = propagated_node_emb[pos_neg_pair_np[:, 1]]
                            neg_i_node_emb = propagated_node_emb[pos_neg_pair_np[:, 2]]
                            pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
                            pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)
                            loss = - (pred_pos - pred_neg).sigmoid().log().mean()
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                            loss_per_batch.append(loss.cpu().item())
                            train_loss = np.mean(loss_per_batch)
                            train_bar.set_description(
                                'Run: {}, epoch: {}, user: {}, train loss: {:.4f}'.format(run, epoch, u_idx,
                                                                                          train_loss))

                        model.eval()
                        HR, NDCG, AUC, eval_loss = self.metrics(
                            run,
                            epoch,
                            model(**model_input),
                            train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map,
                            self.rec_args)
                        HR_per_epoch.append(HR)
                        NDCG_per_epoch.append(NDCG)
                        AUC_per_epoch.append(AUC)
                        train_loss_per_epoch.append(train_loss)
                        eval_loss_per_epoch.append(eval_loss)

                        if epoch in self.train_args['save_epochs']:
                            weightpath = os.path.join(weights_path, '{}.pkl'.format(epoch))
                            save_model(
                                weightpath,
                                model, optimizer, epoch,
                                rec_metrics=(
                                HR_per_epoch, NDCG_per_epoch, AUC_per_epoch, train_loss_per_epoch, eval_loss_per_epoch)
                            )
                        if epoch > self.train_args['save_every_epoch']:
                            weightpath = os.path.join(weights_path, 'latest.pkl')
                            save_model(
                                weightpath,
                                model, optimizer, epoch,
                                rec_metrics=(
                                HR_per_epoch, NDCG_per_epoch, AUC_per_epoch, train_loss_per_epoch, eval_loss_per_epoch)
                            )
                        print(
                            'Run: {}, epoch: {}, HR: {:.4f}, NDCG: {:.4f}, AUC: {:.4f}, '
                            'train loss: {:.4f}, eval loss: {:.4f}'.format(
                                run, epoch, HR, NDCG, AUC, train_loss, eval_loss
                            )
                        )

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                t_end = time.perf_counter()

                HR_per_run.append(HR_per_epoch[-1])
                NDCG_per_run.append(NDCG_per_epoch[-1])
                AUC_per_run.append(AUC_per_epoch[-1])
                train_loss_per_run.append(train_loss_per_epoch[-1])
                eval_loss_per_run.append(eval_loss_per_epoch[-1])

                save_global_logger(
                    global_logger_file_path,
                    HR_per_run, NDCG_per_run, AUC_per_run,
                    train_loss_per_run, eval_loss_per_run
                )

                print(
                    'Run: {}, Duration: {:.4f}, HR: {:.4f}, NDCG: {:.4f}, AUC: {:.4f}, '
                    'train_loss: {:.4f}, eval loss: {:.4f}'.format(
                        run, t_end - t_start, HR_per_epoch[-1], NDCG_per_epoch[-1],
                        AUC_per_epoch[-1], train_loss_per_epoch[-1], eval_loss_per_epoch[-1])
                )
        print(
            'Overall HR: {:.4f}, NDCG: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}'.format(
                np.mean(HR_per_run), np.mean(NDCG_per_run), np.mean(AUC_per_run), np.mean(train_loss_per_run),
                np.mean(eval_loss_per_run)
            )
        )

