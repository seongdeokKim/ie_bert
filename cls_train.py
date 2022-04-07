import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from transformers import AutoTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from utils.models.ie_bert import Informative_Entity_BERT as IE_BERT
from utils.trainer import Trainer
from utils.entity_mention_extractor import EntityMentionExtractor
from utils.data_loader import BertDataset, TokenizerWrapper


class Arguments:
    def __init__(self):
        self.model_fn = './simple_ntc/models/test.pth'
        self.pretrained_model_name = 'bert-base-uncased'

        # self.train_fn = './data/LitCovidForSingleLabel_for_train.txt'
        # self.test_fn = './data/LitCovidForSingleLabel_for_test.txt'
        self.train_fn = './data/_dataset_for_train.txt'
        self.test_fn = './data/_dataset_for_test.txt'


        self.gpu_id = -1
        self.verbose = 2

        self.batch_size = 8
        self.n_epochs = 10
        self.bert_dropout_p = .1

        self.lr = 5e-5
        self.warmup_ratio = .1
        self.adam_epsilon = 1e-8
        self.use_radam = True
        self.max_length = 512

        self.max_entity_num = 10
        self.entity_vector_size = 512
        self.kge_path = './knowledge_graph_embedding/snomed/transe/'
        self.mlp_hidden_size = 1536
        self.hidden_dropout_p = .1


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--test_fn', required=True)

    p.add_argument('--pretrained_model_name', type=str, default='bert-base-cased')

    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--bert_dropout_p', type=float, default=.1)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_true')

    p.add_argument('--max_length', type=int, default=100)

    p.add_argument('--max_entity_num', type=int, default=8)
    p.add_argument('--entity_vector_size', type=int, default=512)
    p.add_argument('--kge_path', type=str, default='./knowledge_graph_embedding/hetionet/transe')
    p.add_argument('--mlp_hidden_size', type=int, default=2560)
    p.add_argument('--hidden_dropout_p', type=float, default=.1)

    config = p.parse_args()

    return config


def read_text(fn):

    with open(fn, 'r', encoding='utf-8') as f:
        labels, texts = [], []
        for line in f:
            if line.strip() != '' and len(line.strip().split('\t')) > 1:
                # The file should have tab delimited three columns.
                # First column indicates label field,
                # and second column indicates text field.

                field_line = line.split('\t')
                label, text = field_line[0].strip(), field_line[1].strip()

                labels += [label]
                texts += [text]

    return labels, texts


def get_knowledge_graph_embedding(config):

    embedding_index = dict()
    with open(config.kge_path + 'ent_labels.tsv', 'r', encoding='utf-8', errors='ignore') as fr1:
        with open(config.kge_path + 'ent_embedding.tsv', 'r', encoding='utf-8', errors='ignore') as fr2:
            for ent_label, embedding in zip(fr1, fr2):
                embedding = embedding.strip().split('\t')
                embedding_index[ent_label.strip().lower().replace(' ', '_')] = list(map(float, embedding))

    entity_to_index = {'<unk>': 0,
                       '<pad>': 1}
    index_to_entity = {0: '<unk>',
                       1: '<pad>'}

    entity_matrix = np.zeros((len(embedding_index.keys()), config.entity_vector_size), dtype='float32')
    for i, entity in enumerate(embedding_index.keys()):
        entity_to_index[entity] = i
        index_to_entity[i] = entity

        embedding_vector = embedding_index.get(entity)
        if embedding_vector is not None:
            entity_matrix[i] = embedding_vector
        else:
            entity_matrix[i] = np.random.uniform(-0.25, 0.25, config.entity_vector_size)
    print(f'number of entities in knowledge graph is {len(embedding_index)}')

    return torch.from_numpy(entity_matrix), entity_to_index, index_to_entity


def get_loaders(config, tokenizer, entity_to_index):
    # Get list of labels and list of texts.
    labels, texts = read_text(config.train_fn)
    print(f'avg length of abstracts: {sum([len(text.split()) for text in texts])/len(texts)}')

    em_extractor = EntityMentionExtractor(entity_to_index)
    entity_mentions = em_extractor.generate_entity_mentions(texts)

    print('labels :', labels[0])
    print('texts :', texts[0])
    print('entity_mentions :', entity_mentions[0])

    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, entity_mentions, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    entity_mentions = [e[1] for e in shuffled]
    labels = [e[2] for e in shuffled]

    valid_ratio = 0.175
    idx = int(len(texts) * (1-valid_ratio))

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        BertDataset(texts[:idx], entity_mentions[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length,
                                    config.max_entity_num,
                                    entity_to_index).collate,
    )
    valid_loader = DataLoader(
        BertDataset(texts[idx:], entity_mentions[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length,
                                    config.max_entity_num,
                                    entity_to_index).collate,
    )


    labels_for_test, texts_for_test = read_text(config.test_fn)
    entity_mentions_for_test = em_extractor.generate_entity_mentions(texts_for_test)
    # Convert label text to integer value.
    labels_for_test = list(map(label_to_index.get, labels_for_test))
    test_loader = DataLoader(
        BertDataset(texts_for_test, entity_mentions_for_test, labels_for_test),
        batch_size=config.batch_size,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length,
                                    config.max_entity_num,
                                    entity_to_index).collate,
    )

    return train_loader, valid_loader, index_to_label, test_loader


def main(config):
    # Get pretrained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    kg_emb, entity_to_index, index_to_entity = get_knowledge_graph_embedding(config)

    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label, test_loader = get_loaders(config, tokenizer, entity_to_index)

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
        '|test| =', len(test_loader) * config.batch_size,
    )

    # Get model with specified softmax layer.
    model = IE_BERT(
        pretrained_model_name=config.pretrained_model_name,
        bert_hidden_size=768 if 'base' in config.pretrained_model_name else 1024,
        n_classes=len(index_to_label),
        bert_dropout_p=config.bert_dropout_p,
        knowledge_graph_embedding=kg_emb,
        entity_vector_size=config.entity_vector_size,
        mlp_hidden_size=config.mlp_hidden_size,
        hidden_dropout_p=config.hidden_dropout_p,
    )

    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id > -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.gpu_id))
        print('GPU on')
        print('Count of using GPUs:', torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print('No GPU')

    model.to(device)

    # Start train.
    bert_trainer = Trainer(config)
    model = bert_trainer.train(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        index_to_label,
        device,
    )

    bert_trainer.test(
        model,
        test_loader,
        index_to_label,
        device,
    )

    torch.save({
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'entity_to_index': entity_to_index,
        'knowledge_graph_embedding': kg_emb,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':
    # config_for_debug = Arguments()
    # main(config_for_debug)

    config = define_argparser()
    main(config)
