from au2actr.data.datasets.xxxx import XXXXDataset

_SUPPORTED_DATASETS = {
    'xxxx': XXXXDataset
}


def dataset_factory(params):
    """
    Factory that generate dataset
    :param params:
    :return:
    """
    dataset_name = params['dataset'].get('name', 'xxxx')
    try:
        dataset = _SUPPORTED_DATASETS[dataset_name](params)
        data = dataset.fetch_data()
        return data
    except KeyError:
        raise KeyError(f'Not support {dataset_name} dataset')
