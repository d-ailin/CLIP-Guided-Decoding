import torchvision.datasets as dset

class CocoDataset(dset.CocoCaptions):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super(CocoDataset, self).__init__(root, annFile, transform, target_transform)

    def __getitem__(self, index):
        img, target = super(CocoDataset, self).__getitem__(index)
        return img, target
    
    def get_image_id(self, index):
        return self.ids[index]
    
    def get_index_by_image_id(self, image_id):
        return self.ids.index(image_id)
    
    def get_sample_by_id(self, image_id):
        index = self.ids.index(image_id)
        return self.__getitem__(index)