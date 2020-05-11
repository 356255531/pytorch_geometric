from abc import abstractmethod
import os
import random as rd
import numpy as np
import torch
import time
import pandas as pd
import tqdm
import itertools

from utils import *


class BaseGraphSolver(object):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        self.model_class = model_class
        self.dataset_args = dataset_args
        self.model_args = model_args
        self.train_args = train_args

        self.init_eval = train_args['init_eval']

    @abstractmethod
    def prepare_model_input(self, data, if_use_features):
        """

        :param data:
        :param if_use_features:
        :return:
        """
        pass

    @abstractmethod
    def train_negative_sampling(self, u_nid, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, data):
        """
        Extract the negative samples from data
        :param u_nid:
        :param train_pos_unid_inid_map:
        :param test_pos_unid_inid_map:
        :param neg_unid_inid_map:
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def generate_candidates(self, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid):
        """
        Generate the candidates for evaluation
        :param train_pos_unid_inid_map:
        :param test_pos_unid_inid_map:
        :param neg_unid_inid_map:
        :param u_nid:
        :return:
        """
        pass

    def metrics(
            self,
            run,
            epoch,
            propagated_node_emb,
            train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map
    ):
        """
        Compute the usual metrics for recommendation system (Hit rate, NDCG and AUC)
        :param run:
        :param epoch:
        :param propagated_node_emb:
        :param train_pos_unid_inid_map:
        :param test_pos_unid_inid_map:
        :param neg_unid_inid_map:
        :return:
        """
        HRs, NDCGs, AUC, eval_losses = np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1))

        u_nids = list(test_pos_unid_inid_map.keys())
        test_bar = tqdm.tqdm(u_nids, total=len(u_nids))
        for u_idx, u_nid in enumerate(test_bar):
            pos_i_nids, neg_i_nids = self.generate_candidates(
                train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid
            )

            if len(pos_i_nids) == 0 or len(neg_i_nids) == 0:
                raise ValueError("No pos or neg samples found in evaluation!")
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

            _, indices = torch.sort(torch.cat([pred_pos, pred_neg]), descending=True)
            hit_vec = (indices < len(pos_i_nids)).cpu().detach().numpy()
            pred_pos = pred_pos.cpu().detach().numpy()
            pred_neg = pred_neg.cpu().detach().numpy()

            HRs = np.vstack([HRs, hit(hit_vec)])
            NDCGs = np.vstack([NDCGs, ndcg(hit_vec)])
            AUC = np.vstack([AUC, auc(pred_pos, pred_neg)])
            eval_losses = np.vstack([eval_losses, loss])
            test_bar.set_description(
                'Run {}, epoch: {}, HR@10: {:.4f}, NDCG@10: {:.4f}, '
                'AUC: {:.4f}, eval loss: {:.4f}, '.format(
                    run, epoch,
                    HRs.mean(axis=0)[5], NDCGs.mean(axis=0)[5], AUC.mean(axis=0)[0],
                    eval_losses.mean(axis=0)[0])
            )

        return HRs.mean(axis=0), NDCGs.mean(axis=0), AUC.mean(axis=0)[0], eval_losses.mean(axis=0)[0]

    def run(self):
        global_logger_path = self.train_args['logger_folder']
        if not os.path.exists(global_logger_path):
            os.makedirs(global_logger_path, exist_ok=True)
        global_logger_file_path = os.path.join(global_logger_path, 'global_logger.pkl')
        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, train_loss_per_run_np, eval_loss_per_run_np, last_run = \
            load_global_logger(global_logger_file_path)

        logger_file_path = os.path.join(global_logger_path, 'logger_file.txt')
        with open(logger_file_path, 'w') as logger_file:
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
                    model_input = self.prepare_model_input(data, if_use_features=self.model_args['if_use_features'])

                    # Create model and optimizer
                    if self.model_args['if_use_features']:
                        self.model_args['emb_dim'] = data.x.shape[1]
                    self.model_args['num_nodes'] = data.num_nodes[0]
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
                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np = \
                        rec_metrics

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    start_epoch = last_epoch + 1
                    if start_epoch == 1 and self.init_eval:
                        HRs_before_np, NDCGs_before_np, AUC_before_np, eval_loss_before_np = self.metrics(
                            run,
                            start_epoch,
                            model(**model_input),
                            train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map
                        )
                        print(
                            'Initial performance HR@10: {:.4f}, NDCG@10: {:.4f}, '
                            'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                                HRs_before_np[5], NDCGs_before_np[5], AUC_before_np, eval_loss_before_np
                            )
                        )
                        logger_file.write(
                            'Initial performance HR@10: {:.4f}, NDCG@10: {:.4f}, '
                            'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                                HRs_before_np[5], NDCGs_before_np[5], AUC_before_np, eval_loss_before_np
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
                                train_neg_i_nids = self.train_negative_sampling(
                                    u_nid,
                                    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map,
                                    data
                                )
                                train_pos_i_nids = [pos_i_nids for _ in range(self.train_args['num_negative_samples'])]
                                train_pos_i_nids = list(itertools.chain.from_iterable(train_pos_i_nids))
                                train_u_nids = [u_nid for _ in range(len(pos_i_nids) * self.train_args['num_negative_samples'])]

                                if len(train_pos_i_nids) == 0 or len(train_neg_i_nids) == 0:
                                    continue

                                propagated_node_emb = model(**model_input)

                                u_node_emb = propagated_node_emb[train_u_nids]
                                pos_i_node_emb = propagated_node_emb[train_pos_i_nids]
                                neg_i_node_emb = propagated_node_emb[train_neg_i_nids]
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
                            HRs, NDCGs, AUC, eval_loss = self.metrics(
                                run,
                                epoch,
                                model(**model_input),
                                train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map
                            )
                            HRs_per_epoch_np = np.vstack([HRs_per_epoch_np, HRs])
                            NDCGs_per_epoch_np = np.vstack([NDCGs, HRs])
                            AUC_per_epoch_np = np.vstack([AUC_per_epoch_np, AUC])
                            train_loss_per_epoch_np = np.vstack([train_loss_per_epoch_np, np.array([train_loss])])
                            eval_loss_per_epoch_np = np.vstack([eval_loss_per_epoch_np, np.array([eval_loss])])

                            if epoch in self.train_args['save_epochs']:
                                weightpath = os.path.join(weights_path, '{}.pkl'.format(epoch))
                                save_model(
                                    weightpath,
                                    model, optimizer, epoch,
                                    rec_metrics=(
                                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np)
                                )
                            if epoch > self.train_args['save_every_epoch']:
                                weightpath = os.path.join(weights_path, 'latest.pkl')
                                save_model(
                                    weightpath,
                                    model, optimizer, epoch,
                                    rec_metrics=(
                                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np)
                                )
                            logger_file.write(
                                'Run: {}, epoch: {}, HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, '
                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                    run, epoch, HRs[5], NDCGs[5], AUC, train_loss, eval_loss
                                )
                            )

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    t_end = time.perf_counter()

                    HRs_per_run_np = np.vstack([HRs_per_run_np, HRs_per_epoch_np[-1]])
                    NDCGs_per_run_np = np.vstack([NDCGs_per_run_np, NDCGs_per_epoch_np[-1]])
                    AUC_per_run_np = np.vstack([AUC_per_run_np, AUC_per_epoch_np[-1]])
                    train_loss_per_run_np = np.vstack([train_loss_per_run_np, train_loss_per_epoch_np[-1]])
                    eval_loss_per_run_np = np.vstack([eval_loss_per_run_np, eval_loss_per_epoch_np[-1]])

                    save_global_logger(
                        global_logger_file_path,
                        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
                        train_loss_per_run_np, eval_loss_per_run_np
                    )
                    print(
                        'Run: {}, Duration: {:.4f}, HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, '
                        'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            run, t_end - t_start, HRs_per_epoch_np[-1][5], NDCGs_per_epoch_np[-1][5],
                            AUC_per_epoch_np[-1][0], train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                    )
                    logger_file.write(
                        'Run: {}, Duration: {:.4f}, HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, '
                        'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            run, t_end - t_start, HRs_per_epoch_np[-1][5], NDCGs_per_epoch_np[-1][5],
                            AUC_per_epoch_np[-1][0], train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                    )
            print(
                'Overall HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                    HRs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[5], AUC_per_run_np.mean(axis=0)[0], train_loss_per_run_np.mean(axis=0)[0],
                    eval_loss_per_run_np.mean(axis=0)[0]
                )
            )
            logger_file.write(
                'Overall HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                    HRs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[5], AUC_per_run_np.mean(axis=0)[0], train_loss_per_run_np.mean(axis=0)[0],
                    eval_loss_per_run_np.mean(axis=0)[0]
                )
            )


class BaseMFSolver(object):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        self.model_class = model_class
        self.dataset_args = dataset_args
        self.model_args = model_args
        self.train_args = train_args

        self.init_eval = train_args['init_eval']

    @abstractmethod
    def train_negative_sampling(self, u_nid, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, data):
        """
        Extract the negative samples from data
        :param u_nid:
        :param train_pos_unid_inid_map:
        :param test_pos_unid_inid_map:
        :param neg_unid_inid_map:
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def generate_candidates(self, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid):
        """
        Generate the candidates for evaluation
        :param train_pos_unid_inid_map:
        :param test_pos_unid_inid_map:
        :param neg_unid_inid_map:
        :param u_nid:
        :return:
        """
        pass

    def metrics(
            self,
            run,
            epoch,
            model,
            train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map,
            data
    ):
        """
        Compute the usual metrics for recommendation system (Hit rate, NDCG and AUC)
        :param run:
        :param epoch:
        :param propagated_node_emb:
        :param train_pos_unid_inid_map:
        :param test_pos_unid_inid_map:
        :param neg_unid_inid_map:
        :return:
        """
        model.eval()
        HRs, NDCGs, AUC, eval_losses = np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1))

        u_nids = list(test_pos_unid_inid_map.keys())
        test_bar = tqdm.tqdm(u_nids, total=len(u_nids))
        for u_idx, u_nid in enumerate(test_bar):
            pos_i_nids, neg_i_nids = self.generate_candidates(
                train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid
            )

            pos_i_nid_df = pd.DataFrame({'u_nid': [u_nid for _ in range(len(pos_i_nids))], 'pos_i_nid': pos_i_nids})
            neg_i_nid_df = pd.DataFrame({'u_nid': [u_nid for _ in range(len(neg_i_nids))], 'neg_i_nid': neg_i_nids})
            pos_neg_pair_np = pd.merge(pos_i_nid_df, neg_i_nid_df, how='inner', on='u_nid').to_numpy()
            eval_u_nids = torch.from_numpy(pos_neg_pair_np[:, 0]).to(self.train_args['device'])
            eval_pos_i_nids = torch.from_numpy(pos_neg_pair_np[:, 1]).to(self.train_args['device']) - data.e2nid[0]['iid'][0]
            eval_neg_i_nids = torch.from_numpy(pos_neg_pair_np[:, 2]).to(self.train_args['device']) - data.e2nid[0]['iid'][0]

            pred_pos = model(eval_u_nids, eval_pos_i_nids)
            pred_neg = model(eval_u_nids, eval_neg_i_nids)

            loss = - (pred_pos - pred_neg).sigmoid().log().mean().item()

            pos_u_nids_t = torch.from_numpy(np.array([u_nid for _ in range(len(pos_i_nids))])).to(self.train_args['device'])
            neg_u_nids_t = torch.from_numpy(np.array([u_nid for _ in range(len(neg_i_nids))])).to(self.train_args['device'])
            pos_i_nids_t = torch.from_numpy(np.array(pos_i_nids)).to(self.train_args['device']) - data.e2nid[0]['iid'][0]
            neg_i_nids_t = torch.from_numpy(np.array(neg_i_nids)).to(self.train_args['device']) - data.e2nid[0]['iid'][0]
            pred_pos = model(pos_u_nids_t, pos_i_nids_t).reshape(-1)
            pred_neg = model(neg_u_nids_t, neg_i_nids_t).reshape(-1)
            _, indices = torch.sort(torch.cat([pred_pos, pred_neg]), descending=True)
            hit_vec = (indices < len(pos_i_nids)).cpu().detach().numpy()
            pred_pos = pred_pos.detach().cpu().numpy()
            pred_neg = pred_neg.detach().cpu().numpy()

            HRs = np.vstack([HRs, hit(hit_vec)])
            NDCGs = np.vstack([NDCGs, ndcg(hit_vec)])
            AUC = np.vstack([AUC, auc(pred_pos, pred_neg)])
            eval_losses = np.vstack([eval_losses, loss])
            test_bar.set_description(
                'Run {}, epoch: {}, HR@10: {:.4f}, NDCG@10: {:.4f}, '
                'AUC: {:.4f}, eval loss: {:.4f}, '.format(
                    run, epoch,
                    HRs.mean(axis=0)[5], NDCGs.mean(axis=0)[5], AUC.mean(axis=0)[0],
                    eval_losses.mean(axis=0)[0])
            )

        return HRs.mean(axis=0), NDCGs.mean(axis=0), AUC.mean(axis=0)[0], eval_losses.mean(axis=0)[0]

    def run(self):
        global_logger_path = self.train_args['logger_folder']
        if not os.path.exists(global_logger_path):
            os.makedirs(global_logger_path, exist_ok=True)
        global_logger_file_path = os.path.join(global_logger_path, 'global_logger.pkl')
        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, train_loss_per_run_np, eval_loss_per_run_np, last_run = \
            load_global_logger(global_logger_file_path)

        logger_file_path = os.path.join(global_logger_path, 'logger_file.txt')
        with open(logger_file_path, 'w') as logger_file:
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

                    # Create model and optimizer
                    self.model_args['num_users'] = data.users[0].shape[0]
                    self.model_args['num_items'] = data.items[0].shape[0]
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
                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np = \
                        rec_metrics

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    start_epoch = last_epoch + 1
                    if start_epoch == 1 and self.init_eval:
                        HRs_before_np, NDCGs_before_np, AUC_before_np, eval_loss_before_np = self.metrics(
                            run,
                            start_epoch,
                            model,
                            train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map,
                            data
                        )
                        print(
                            'Initial performance HR@10: {:.4f}, NDCG@10: {:.4f}, '
                            'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                                HRs_before_np[5], NDCGs_before_np[5], AUC_before_np, eval_loss_before_np
                            )
                        )
                        logger_file.write(
                            'Initial performance HR@10: {:.4f}, NDCG@10: {:.4f}, '
                            'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                                HRs_before_np[5], NDCGs_before_np[5], AUC_before_np, eval_loss_before_np
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
                                train_neg_i_nids = self.train_negative_sampling(
                                    u_nid,
                                    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map,
                                    data
                                )
                                train_pos_i_nids = [pos_i_nids for _ in range(self.train_args['num_negative_samples'])]
                                train_pos_i_nids = list(itertools.chain.from_iterable(train_pos_i_nids))
                                train_u_nids = [u_nid for _ in range(len(pos_i_nids) * self.train_args['num_negative_samples'])]
                                train_u_nids = torch.from_numpy(np.array(train_u_nids)).long().to(self.train_args['device'])
                                train_pos_i_nids = torch.from_numpy(np.array(train_pos_i_nids)).long().to(self.train_args['device']) - data.e2nid[0]['iid'][0]
                                train_neg_i_nids = torch.from_numpy(np.array(train_neg_i_nids)).long().to(self.train_args['device']) - data.e2nid[0]['iid'][0]

                                pred_pos = model(train_u_nids, train_pos_i_nids)
                                pred_neg = model(train_u_nids, train_neg_i_nids)
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
                            HRs, NDCGs, AUC, eval_loss = self.metrics(
                                run,
                                epoch,
                                model,
                                train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map,
                                data
                            )
                            HRs_per_epoch_np = np.vstack([HRs_per_epoch_np, HRs])
                            NDCGs_per_epoch_np = np.vstack([NDCGs, HRs])
                            AUC_per_epoch_np = np.vstack([AUC_per_epoch_np, AUC])
                            train_loss_per_epoch_np = np.vstack([train_loss_per_epoch_np, np.array([train_loss])])
                            eval_loss_per_epoch_np = np.vstack([eval_loss_per_epoch_np, np.array([eval_loss])])

                            if epoch in self.train_args['save_epochs']:
                                weightpath = os.path.join(weights_path, '{}.pkl'.format(epoch))
                                save_model(
                                    weightpath,
                                    model, optimizer, epoch,
                                    rec_metrics=(
                                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np)
                                )
                            if epoch > self.train_args['save_every_epoch']:
                                weightpath = os.path.join(weights_path, 'latest.pkl')
                                save_model(
                                    weightpath,
                                    model, optimizer, epoch,
                                    rec_metrics=(
                                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np)
                                )
                            logger_file.write(
                                'Run: {}, epoch: {}, HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, '
                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                    run, epoch, HRs[5], NDCGs[5], AUC, train_loss, eval_loss
                                )
                            )

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    t_end = time.perf_counter()

                    HRs_per_run_np = np.vstack([HRs_per_run_np, HRs_per_epoch_np[-1]])
                    NDCGs_per_run_np = np.vstack([NDCGs_per_run_np, NDCGs_per_epoch_np[-1]])
                    AUC_per_run_np = np.vstack([AUC_per_run_np, AUC_per_epoch_np[-1]])
                    train_loss_per_run_np = np.vstack([train_loss_per_run_np, train_loss_per_epoch_np[-1]])
                    eval_loss_per_run_np = np.vstack([eval_loss_per_run_np, eval_loss_per_epoch_np[-1]])

                    save_global_logger(
                        global_logger_file_path,
                        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
                        train_loss_per_run_np, eval_loss_per_run_np
                    )
                    print(
                        'Run: {}, Duration: {:.4f}, HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, '
                        'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            run, t_end - t_start, HRs_per_epoch_np[-1][5], NDCGs_per_epoch_np[-1][5],
                            AUC_per_epoch_np[-1][0], train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                    )
                    logger_file.write(
                        'Run: {}, Duration: {:.4f}, HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, '
                        'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            run, t_end - t_start, HRs_per_epoch_np[-1][5], NDCGs_per_epoch_np[-1][5],
                            AUC_per_epoch_np[-1][0], train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                    )
            print(
                'Overall HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                    HRs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[5], AUC_per_run_np.mean(axis=0)[0], train_loss_per_run_np.mean(axis=0)[0],
                    eval_loss_per_run_np.mean(axis=0)[0]
                )
            )
            logger_file.write(
                'Overall HR@10: {:.4f}, NDCG@10: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                    HRs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[5], AUC_per_run_np.mean(axis=0)[0], train_loss_per_run_np.mean(axis=0)[0],
                    eval_loss_per_run_np.mean(axis=0)[0]
                )
            )

