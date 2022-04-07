from transformers import BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class Informative_Entity_BERT(nn.Module):
    def __init__(
            self,
            pretrained_model_name,
            bert_hidden_size,
            n_classes,
            bert_dropout_p,
            knowledge_graph_embedding,
            entity_vector_size,
            mlp_hidden_size,
            hidden_dropout_p,
    ):
        super(Informative_Entity_BERT, self).__init__()

        self.bert_hidden_size = bert_hidden_size
        self.bert_dropout_p = bert_dropout_p
        self.bert = BertModel.from_pretrained(
            pretrained_model_name,
        )
        self.bert_dropout = nn.Dropout(self.bert_dropout_p)

        self.knowledge_graph_embedding = knowledge_graph_embedding
        self.entity_vector_size = entity_vector_size
        self.n_classes = n_classes

        if self.knowledge_graph_embedding is not None:
            self.knowledge_graph_emb = nn.Embedding.from_pretrained(
                self.knowledge_graph_embedding, padding_idx=0
            )
        else:
            raise Exception("there is no embedding matrix from knowledge graph.")
        #     self.emb = nn.Embedding(input_size, word_vec_size, padding_idx=0)
        #     nn.init.uniform_(self.emb.weight, -0.25, 0.25)

        self.mlp_hidden_size = mlp_hidden_size
        #self.mlp_hidden_size = 2 * (self.bert_hidden_size + self.entity_vector_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.bert_hidden_size + self.entity_vector_size, self.mlp_hidden_size),
            nn.Dropout(hidden_dropout_p),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.bert_hidden_size + self.entity_vector_size),
            nn.Dropout(hidden_dropout_p),
            nn.ReLU(),
            nn.Linear(self.bert_hidden_size + self.entity_vector_size, self.n_classes)
        )

        ## We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        #self.activation = nn.LogSoftmax(dim=-1)

    def forward(
            self,
            input_ids,
            attention_mask,
            entity_ids=None,
            labels=None,
    ):
        #|e_emb| = (batch_size, entity_vector_siz

        # |input_ids| = (batch_size, max_seq_length)
        _, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        # |pooled_output| = (batch_size, bert_hidden_size)

        # |entity_ids| = (batch_size, num_of_entities)
        entity_emb = self.knowledge_graph_emb(entity_ids)
        # |e_emb| = (batch_size, num_of_entities, entity_vector_size)
        entity_emb = entity_emb.sum(dim=1)

        concat = torch.cat((pooled_output, entity_emb), dim=1)
        # |concat| = (batch_size, bert_hidden_size + entity_vector_size)
        logits = self.mlp(concat)
        # |logits| = (batch_size, n_classes)
        #logits = self.activation(concat)
        ## |logits| = (batch_size, n_classes)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))

        output = (logits,)
        return ((loss,) + output) if loss is not None else output