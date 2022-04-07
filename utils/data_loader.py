import torch
from torch.utils.data import Dataset

from torchtext import data


class BertDataset(Dataset):

    def __init__(self,
                 texts: 'list of str',
                 entity_mentions: 'list of list',
                 labels: 'list of int' = None
                 ):

        self.texts = texts
        self.entity_mentions = entity_mentions
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        entity_mentions_per_text = self.entity_mentions[item]
        if self.labels is not None:
            label = self.labels[item]
        else:
            label = None

        return {
            'text': text,
            'entity_mentions_per_text': entity_mentions_per_text,
            'label': label,
        }


class TokenizerWrapper:

    def __init__(self, tokenizer, max_length, max_entity_num, entity_to_index):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.max_entity_num = max_entity_num
        self.entity_to_index = entity_to_index
        self.entity_list_from_kg = set(self.entity_to_index.keys())

    def collate(self, samples):
        output_dict = {}

        texts = [s['text'] for s in samples]
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        output_dict['input_ids'] = torch.tensor(encoding['input_ids'], dtype=torch.long)
        output_dict['attention_mask'] = torch.tensor(encoding['attention_mask'], dtype=torch.long)

        entity_mentions = [s['entity_mentions_per_text'] for s in samples]
        entity_ids = [self.convert_entity_mentions_to_entity_ids(entity_mentions_per_text) for entity_mentions_per_text in entity_mentions]
        output_dict['entity_ids'] = torch.tensor(entity_ids, dtype=torch.long)

        if samples[0]['label'] is not None:
            labels = [s['label'] for s in samples]
            output_dict['labels'] = torch.tensor(labels, dtype=torch.long)

        return output_dict

    def convert_entity_mentions_to_entity_ids(self, entity_mentions_per_text: 'list of str'):

        # padding
        if len(entity_mentions_per_text) < self.max_entity_num:
            pad_len = self.max_entity_num - len(entity_mentions_per_text)
            entity_mentions_per_text = entity_mentions_per_text + (['<pad>'] * pad_len)
        else:
            entity_mentions_per_text = entity_mentions_per_text[:self.max_entity_num]

        # convert keywords_to_ids
        entity_ids_per_text = []
        for entity in entity_mentions_per_text:
            if entity not in self.entity_list_from_kg:
                entity_ids_per_text.append(self.entity_to_index['<unk>'])
            else:
                entity_ids_per_text.append(self.entity_to_index[entity])

        return entity_ids_per_text
