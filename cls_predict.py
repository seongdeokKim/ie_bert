import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import AutoTokenizer

from utils.models.ie_bert import Informative_Entity_BERT as IE_BERT
from torch.utils.data import DataLoader
from utils.entity_mention_extractor import EntityMentionExtractor
from utils.data_loader import BertDataset, TokenizerWrapper


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config


def read_text():
    '''
    Read text from standard input for inference.
    '''
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip()]

    return lines


def main(config):

    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']
    entity_to_index = saved_data['entity_to_index']
    kg_emb = saved_data['knowledge_graph_embedding']

    # Get abstracts
    # lines = load_text(config.input_file)
    texts = read_text()

    # Extract entities mentioned in abstracts
    em_extractor = EntityMentionExtractor(entity_to_index)
    entity_mentions = em_extractor.generate_entity_mentions(texts)

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(train_config.pretrained_model_name)
    loaded_tokenizer = saved_data['tokenizer']

    predict_loader = DataLoader(
        BertDataset(texts, entity_mentions),
        batch_size=config.batch_size,
        collate_fn=TokenizerWrapper(loaded_tokenizer,
                                    train_config.max_length,
                                    train_config.max_entity_num,
                                    entity_to_index).collate,
    )

    with torch.no_grad():

        # Declare model and load pre-trained weights.
        model = IE_BERT(
            pretrained_model_name=train_config.pretrained_model_name,
            bert_hidden_size=768 if 'base' in train_config.pretrained_model_name else 1024,
            n_classes=len(index_to_label),
            bert_dropout_p=train_config.bert_dropout_p,
            knowledge_graph_embedding=kg_emb,
            entity_vector_size=train_config.entity_vector_size,
            mlp_hidden_size=train_config.mlp_hidden_size,
            hidden_dropout_p=train_config.hidden_dropout_p,
        )

        model.load_state_dict(bert_best)

        if config.gpu_id > -1:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for _, mini_batch in enumerate(predict_loader):
            input_ids = mini_batch['input_ids']
            input_ids = input_ids.to(device)
            attention_mask = mini_batch['attention_mask']
            attention_mask = attention_mask.to(device)

            entity_ids = mini_batch['entity_ids']
            entity_ids = entity_ids.to(device)

            # Take feed-forward
            logits = model(input_ids,
                           attention_mask=attention_mask,
                           entity_ids=entity_ids)[0]

            #current_y_hats = np.argmax(logits, axis=-1)
            current_y_hats = F.softmax(logits, dim=-1)
            y_hats += [current_y_hats]

        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        probs, indices = y_hats.cpu().topk(k=len(index_to_label))

        for i in range(len(texts)):
            sys.stdout.write('{}\t{}\t{}\n'.format(
                ",".join([index_to_label.get(int(j)) for j in indices[i][:config.top_k]]),
                ",".join([str(float(j))[:6] for j in probs[i][:config.top_k]]),
                texts[i],
            ))


if __name__ == '__main__':

    import time
    start = time.time()

    config = define_argparser()
    main(config)

    sys.stdout.write('\n{:.4f} seconds\n'.format(time.time()-start))