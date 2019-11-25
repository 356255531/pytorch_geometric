import os.path as osp


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
