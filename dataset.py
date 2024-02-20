import json
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, test_file=None):
        self.dataset = []
        self.dataset = self._load_data(test_file)
        print(f'Number of test dataset: {len(self.dataset)}.')

    def __getitem__(self, idx):
        idx = self.dataset[idx]['idx']
        dialogue_history = self.dataset[idx]['dialogue_history'] 
        dialogue_summary = self.dataset[idx]['dialogue_summary']
        return idx, dialogue_history, dialogue_summary

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_function(batch):
        idx, dialogue_history, dialogue_summary = zip(*batch)
        return idx, dialogue_history, dialogue_summary
    
    def _load_data(self, file_name):
        dataset = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                json_dict = json.loads(line)
                dialogue_history, dialogue_summary = json_dict['dialogue_history'], json_dict['dialogue_summary']
                dataset.append({"idx": idx, "dialogue_history": dialogue_history, "dialogue_summary": dialogue_summary})
        return dataset
