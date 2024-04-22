import torchvision.datasets as dset
from PIL import Image
from torch.utils.data import Dataset
import json

class MMVetDataset(Dataset):
    def __init__(self, data_path, qa_file):
        # get image ids from qa_file
        
        self.image_ids, self.image_infos = self.load_image_infos(qa_file)
        self.data_path = data_path
    
    # def parse_image_id(self, image_path):
    #     return int(image_path.split('_')[-1].rstrip('.jpg'))
    
    def load_image_infos(self, qa_file):
        qa_info = json.load(open(qa_file))
        image_ids = []
        image_infos = {}
        for k, qa_item in qa_info.items():
            image_id = k
            image_ids.append(image_id)
            image_infos[image_id] = qa_item['imagename']
        
        
        image_ids = list(set(image_ids)) # remove duplicates
        image_ids.sort()
        return image_ids, image_infos

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        # image_path = self.image_infos[image_id]
        image_path = '{}/{}'.format(self.data_path, self.image_infos[image_id])

        image = Image.open(image_path).convert('RGB')
        
        return image, image_id
    
    def get_image_id(self, index):
        return self.image_ids[index]
    
    def get_index_by_image_id(self, image_id):
        return self.image_ids.index(image_id)
    
    def get_sample_by_id(self, image_id):
        index = self.get_index_by_image_id(image_id)
        return self.__getitem__(index)
    
def process_output(out_path):
    out = json.load(open(out_path, 'r'))
    new_out = {}
    for item in out:
        # new_out[item['image_id']] = item['answer']
        new_out[item['question_index']] = item['answer']
    return new_out
        