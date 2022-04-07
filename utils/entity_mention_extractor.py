from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag


class EntityMentionExtractor:
    def __init__(self, entity_to_index):
        self.entity_set = set(list(entity_to_index.keys()))

    def generate_entity_mentions(self, documents: 'list of str') -> 'list of list of str':
        entity_mentions = []
        for document in documents:
            # Extract n_grams on a document
            n_grams = self.extract_n_grams(max_n=5,
                                           doc=document)
            # Match n_grams in a document with entities in a knowledge graph
            entity_mentions_per_doc = [n_gram for n_gram in n_grams if n_gram in self.entity_set]
            # print('doc: ', document)
            # print('n_grams: ', n_grams)
            # print('entity_mention in document:', entity_mentions_per_doc)
            # print()

            entity_mentions.append(entity_mentions_per_doc)
            #print('entity_mentions_per_doc :', entity_mentions_per_doc)

        self.print_info(entity_mentions)

        return entity_mentions

    def extract_n_grams(self, max_n: int, doc: str) -> 'list of str':

        n_grams = []

        prepocessed_doc = self.preprocess(doc)
        for k in range(max_n):
            current_n = k + 1

            if len(prepocessed_doc) - current_n < 1:
                return n_grams

            for i in range(len(prepocessed_doc) - current_n + 1):
                n_gram_span = prepocessed_doc[i: i+current_n]
                n_gram = '_'.join(n_gram_span)
                n_grams.append(n_gram)

        return n_grams

    @staticmethod
    def preprocess(document: 'str') -> 'list of str':

        preprocessed_doc = []
        for sent in sent_tokenize(document.lower()):
            # tokenizing
            tokenized_sent = word_tokenize(sent)
            # pos tagging
            tokens_and_tags = pos_tag(tokenized_sent)
            # pos filtering
            tokens_and_tags = [tok_and_tag for tok_and_tag in tokens_and_tags if 'NN' in tok_and_tag[1]]
            # select token only
            tokens = [tok_and_tag[0] for tok_and_tag in tokens_and_tags]

            preprocessed_doc += tokens

        return preprocessed_doc

    @staticmethod
    def print_info(entity_mentions: 'list of list'):

        match_count = 0
        overlap_entity_list = set()
        for per_abstract in entity_mentions:
            for e in per_abstract:
                match_count += 1
                overlap_entity_list.add(e)

        print()
        print(f'number of unique entity occurred in dataset is {len(overlap_entity_list)}')
        print(f'number of entity matching is {match_count}')
        print(f'average number of entity matching per abstract is {match_count / len(entity_mentions)}')
        print()
