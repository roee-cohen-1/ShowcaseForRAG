import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _split_to_chunks(documents, chunking_method):
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    chunks = []
    for document in tqdm(documents, desc='Chunking'):
        chunks += re.split(sentence_endings, document)
    return documents


def load_and_embed(
        dataset_name='sujayC66/stocknews_summarization_1700',
        embedding_model='all-MiniLM-L6-v2',
        chunking_method='sentence',
        split='train',
        field='content'):

    if chunking_method not in ['sentence']:
        raise Exception('unknown chunking method')

    dataset = load_dataset(dataset_name, split=split)
    model = SentenceTransformer(embedding_model)
    documents = dataset[field]

    chunks = _split_to_chunks(documents, chunking_method)
    return model.encode(chunks, show_progress_bar=True), chunks, model


