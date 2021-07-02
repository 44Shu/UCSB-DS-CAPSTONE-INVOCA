# We can do all data loading and cleaning here

import pandas as pd
import itertools
<<<<<<< HEAD
from util_functions import preprocess
=======
from util_functions_RNNs import preprocess
from gensim.models import ldamodel
>>>>>>> 9dabfbb52d8e446e1d1beff51e8ea672a1b7847b



# make straight forward dataset by merging the text in each utterance cell and its corresponded label
# label is the first categories of the annotations, in this stage we will limit our label amount


woz_path = "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/woz-dialogs.json"
solo_path = "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/self-dialogs.json"

solo_data = pd.read_json(solo_path)
woz_data = pd.read_json(woz_path)

# Shuyun's data process
def data_process():

    text = []
    labels = []

    for utterance in solo_data['utterances']:
      text_merge = ''
      for line in utterance:
        text_merge += ' '
        text_merge += line['text']
        if 'segments' in line.keys():
          label = line['segments'][0]['annotations'][0]['name'].split(".")[0]
      text.append(text_merge)
      labels.append(label)

    data = pd.DataFrame(list(zip(text,labels)), columns = ['text','label'])

    cleanup_nums = {"label": {"restaurant_reservation": 1, "movie_ticket": 2, "pizza_ordering": 3,
                              "coffee_ordering": 4, "auto_repair": 5, "uber_lyft": 6}}

    data = data.replace(cleanup_nums)

    return data


# This part corresponds to Amy and Anton's update in Collab
def process_label(label, convo_len):
    """Take label of form subject-order-int and return each element separately
    in a list of lengtg convo_len"""

    label_tup = label.split('-')

    label_1 = list(itertools.repeat(label_tup[0], convo_len))
    label_2 = list(itertools.repeat(label_tup[1], convo_len))
    label_3 = list(itertools.repeat(label[2], convo_len))

    return label_1, label_2, label_3



def line_by_line(row, clean=True):
    conversation = row['utterances']

    # Generate lists for each column
    line_list = []
    index_list = []
    speaker_list = []
    # annotations = []
    for i in range(len(conversation)):
        line = conversation[i]['text']
        line_list.append(line)

        index = conversation[i]['index']
        index_list.append(index)

        speaker = conversation[i]['speaker']
        speaker_list.append(speaker)

    convo_id = list(itertools.repeat(row['conversation_id'], len(conversation)))

    label_1, label_2, label_3 = process_label(row['instruction_id'], len(conversation))

    data_dict = {
        'convo_id': convo_id,
        'convo_index': index_list,
        'lines': line_list,
        'speaker': speaker_list,
        'label_1': label_1,
        'label_2': label_2,
        'label_3': label_3
    }

    data = pd.DataFrame(data_dict)

    if clean:
        preprocess(data, 'lines')

    return data


# for each row in solo_data, define each convo, run line_by_line and then append to dataframe and concat
append_total = []

def anton_amy_data_process():
    for i in range(len(solo_data)):
        convo = solo_data.iloc[i]
        append_total.append(line_by_line(convo))
    final = pd.concat(append_total)
    user_lines = final.loc[final['speaker'] == 'USER']

    cleanup_nums = {"label_1": {"pizza": 1, "coffee": 2, "restaurant": 3,
                                "movie": 4, "auto": 5, "uber": 6}}

    final = user_lines.replace(cleanup_nums)

    final.head()

