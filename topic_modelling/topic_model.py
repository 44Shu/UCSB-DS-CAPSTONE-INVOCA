import pyLDAvis.gensim
import pickle
import pyLDAvis
import os
import logging
import re
from pprint import pprint
import gensim
from gensim.utils import simple_preprocess
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric
from gensim import corpora
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
from data_loading import data_process

stop_words = stopwords.words('english')
stop_words.extend(['from', 'ok', 'would', 'need', 'like', 'want', 'okay', 'yes', 'please', 'else', 'thanks', 'ye'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def preprocess_for_lda(convos):

    # Remove punctuation
    convos['convo_text_processed'] = \
    convos['text'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    convos['convo_text_processed'] = \
    convos['convo_text_processed'].map(lambda x: x.lower())

    # Print out the first rows of papers
    convos['convo_text_processed'].head()

    data = convos.convo_text_processed.values.tolist()
    data_words = list(sent_to_words(data))

    # remove stop words
    data_words = remove_stopwords(data_words)

    return data_words

def create_dict_corpus(data_words):

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return corpus, id2word

def process_convos(data):

    data_words = preprocess_for_lda(data)
    corpus, id2word = create_dict_corpus(data_words)

    return corpus, id2word


def train_lda(corpus, mapping_dict, num_topics, epochs, alpha, metric_log, filepath):

    logger = None

    if metric_log == True:
        logging.basicConfig(filename='model_callbacks.log',
                            format="%(asctime)s:%(levelname)s:%(message)s",
                            level=logging.NOTSET)

        # number of topics
        num_topics = 5

        perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
        convergence_logger = ConvergenceMetric(logger='shell')

        logger = [perplexity_logger, convergence_logger]

    # Build LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=mapping_dict,
                                       num_topics=num_topics,
                                       passes = epochs,
                                       alpha = alpha,
                                       eval_every=len(corpus),
                                       callbacks=logger)

    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    lda_model.save(filepath)

def save_display_lDAvis(model, corpus, mapping_dict, filepath, num_topics):
    # Visualize the topics
    pyLDAvis.enable_notebook()

    LDAvis_data_filepath = os.path.join(filepath + '_' + str(num_topics))

    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself

    LDAvis_prepared = pyLDAvis.gensim.prepare(model, corpus, mapping_dict)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, filepath + '_' + str(num_topics) +'.html')

def plot_metrics(lda_model):

    df = pd.DataFrame.from_dict(lda_model.metrics)

    df['topics'] = 5

    df = df.reset_index().rename(columns={'index': 'pass_num'})

    for metric in ['Perplexity', 'Convergence']:

        fig, axs = plt.subplots(1, 1, figsize=(20, 7))

        # Each plot to show results for all models with the same topic number
        for i, topic_number in enumerate([5]):
            filtered_topics = df[df['topics'] == topic_number]
            for label in filtered_topics:

                df.plot(x='pass_num', y=metric, ax=axs, label=label)

            axs.set_xlabel(f"Pass number")
            axs.legend()
            axs.set_ylim([df[metric].min() * 0.9, df[metric].max() * 1.1])


        fig.suptitle(metric, fontsize=20)
        plt.show()


def lda_pipeline(data, filepath='lda_model', num_topics=5, epochs=20, alpha='symmetric', metric_log=False):
    """Implement lda training on data, creates an interactive plot saved to an html file and if
    metri_log is true it also logs perplexity and coherence scores and plots them per epoch.
    """

    corpus, id2word = process_convos(data)

    model = train_lda(corpus, id2word, num_topics, epochs, alpha, metric_log, filepath)

    save_display_lDAvis(model, corpus, id2word, filepath, num_topics)

    if metric_log == True:
        plot_metrics(model)

data = data_process()

lda_pipeline(data, metric_log=True)

