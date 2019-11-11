from torch_geometric.datasets import MovieLens

def get_dataset(dataset_args):
    if dataset_args['dataset'] == 'movielens':
        return MovieLens(**dataset_args)
    else:
        raise NotImplemented('{} has not been implemented!'.format(dataset_args['dataset']))
