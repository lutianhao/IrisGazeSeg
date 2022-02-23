from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'CityscapesTrain': {
            'id': 'IrisGazeSeg',
            'data_root': 'tmp/leftImg8bit',
            'ann_file': ('tmp/Anno/train', 'tmp/Anno/train'),
            'split': 'train'
        },
        'CityscapesVal': {
            'id': 'IrisGazeSeg',
            'data_root': 'tmp/leftImg8bit',
            'ann_file': 'tmp/Anno/val',
            'split': 'val'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()

