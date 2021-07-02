import pandas as pd
from data_loading import solo_data


def get_convo_labels(conversation):
    label_list = list()
    for line in conversation:
        if 'segments' in line.keys():
            for segment in line['segments']:
                for name in segment['annotations']:
                    label_list.append(name['name'])
    return label_list


def get_all_labels(data):
    label_list = list()

    for conversation in data:
        label_list.extend(get_convo_labels(conversation))
    return label_list


def get_max_label_length(label_list):
    label_lengths = []
    for label in label_list:
        label_lengths.append(len(label.split('.')))
    return max(label_lengths)


def split_labels(label_list):
    labels_split = list()
    for label in label_list:
        list_of_labels = label.split('.')
        while len(list_of_labels) < 4:
            list_of_labels.append('None')
        labels_split.append(list_of_labels)

    return labels_split


# Get labels and create a nested list of each level
all_labels = get_all_labels(solo_data['utterances'])
labels = split_labels(all_labels)

# flatten the list to see how many unique labels there are
flat_label_list = [label for ls in labels for label in ls]
unique_labels = list(set(flat_label_list))

# Turn into dataframe and create a groupby object
label_frame = pd.DataFrame(labels)
label_frame['Full_label'] = all_labels
grouped = label_frame.groupby([0, 1, 2, 3])
grouped_label_counts = grouped.count()
grouped_label_counts.to_csv('Grouped_label_counts.csv')
