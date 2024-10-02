from datasets import DatasetDict, Dataset
import random
import json
import os

def load_ontology_rules_dataset():
    # Define the parent directory where the subdirectories are located
    parent_directory = 'ontology_rules'

    # Initialize an empty dictionary to store the organized data
    data_dict = {}

    # Define the dimensions and sentiments
    dimensions = ['train', 'test']
    sentiments = ['pos', 'neg']

    # Iterate through each subdirectory
    for subdir in os.listdir(parent_directory):
        subdir_path = os.path.join(parent_directory, subdir)

        # Check if the item in the parent directory is a directory
        if os.path.isdir(subdir_path):
            # Initialize a dictionary to store the content of text files in the current subdirectory
            subdir_dict = {}

            # Iterate through each dimension
            for dimension in dimensions:
                # Initialize a dictionary to store sentiment content in the current dimension
                sentiment_dict = {}

                # Iterate through each sentiment
                for sentiment in sentiments:
                    file_name = f"{sentiment}_{dimension}_rules.txt"
                    file_path = os.path.join(subdir_path, file_name)

                    # Check if the file exists
                    if os.path.exists(file_path):
                        # Read the content of the text file and split it based on the newline character
                        with open(file_path, 'r') as file:
                            content_lines = file.read().split('\n')
                            sentiment_dict[sentiment] = content_lines

                # Store the sentiment dictionary in the dimension dictionary
                subdir_dict[dimension] = sentiment_dict

            # Store the dimension dictionary in the main dictionary with the subdirectory name as the key
            data_dict[subdir] = subdir_dict

    # Processing for training and testing
    rows_train = []
    rows_test = []
    for theme in data_dict:
        # Extract data for the current theme
        pos_train = data_dict[theme]['train']['pos']
        neg_train = data_dict[theme]['train']['neg']
        pos_test = data_dict[theme]['test']['pos']
        neg_test = data_dict[theme]['test']['neg']

        random.seed(42)
        random.shuffle(pos_test)
        random.shuffle(neg_test)

        # Combine positive and negative examples for training and testing
        rows_train.extend([(rule, 1, theme) for rule in pos_train])
        rows_train.extend([(rule, 0, theme) for rule in neg_train])

        rows_test.extend([(rule, 1, theme) for rule in pos_test])
        rows_test.extend([(rule, 0, theme) for rule in neg_test])

    # Create DatasetDict with 'train' and 'test' splits
    dataset = DatasetDict({
        'train': Dataset.from_dict({'rules': [row[0] for row in rows_train],
                                    'labels': [row[1] for row in rows_train],
                                    'theme': [row[2] for row in rows_train]}),
        'test': Dataset.from_dict({'rules': [row[0] for row in rows_test],
                                   'labels': [row[1] for row in rows_test],
                                   'theme': [row[2] for row in rows_test]}),
    })

    return dataset

def load_gcn_not_predicted_dataset():
    parent_directory = 'ontology_rules_are_not_predicted_by_gcn'
    data_dict_not_predicted = {}

    for file_name in os.listdir(parent_directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(parent_directory, file_name)
            with open(file_path, 'r') as file:
                content_lines = file.read().split('\n')
                data_dict_not_predicted[file_name.split("_")[0]] = {"rules": content_lines}

    rows_notPred = []
    for theme in data_dict_not_predicted:
        notPred = data_dict_not_predicted[theme]['rules']
        rows_notPred.extend([(rule, theme) for rule in notPred])

    dataset = DatasetDict({
        'notPred': Dataset.from_dict({'rules': [row[0] for row in rows_notPred],
                                      'theme': [row[1] for row in rows_notPred]})
    })

    return dataset

def load_split_types_dataset():
    parent_directory = 'datasets_split_types/untyped'
    data_dict_test = {}

    for subdir in os.listdir(parent_directory):
        subdir_path = os.path.join(parent_directory, subdir)
        subdir_dict = {}
        for file_name in os.listdir(subdir_path):
            if file_name.startswith('test_') and file_name.endswith('.json'):
                split_type = file_name.split('_')[1].split('.')[0]
                file_path = os.path.join(subdir_path, file_name)
                with open(file_path, 'r') as file:
                    content_lines = [json.loads(line) for line in file]
                    if 'test' not in subdir_dict:
                        subdir_dict['test'] = []
                    for entry in content_lines:
                        entry['split-type'] = split_type
                    subdir_dict['test'].extend(content_lines)
        data_dict_test[subdir] = subdir_dict

    rows_test = []
    for theme in data_dict_test:
        test_data = data_dict_test[theme]['test']
        random.seed(42)
        random.shuffle(test_data)
        rows_test.extend([(entry['rule'], entry['label'], theme, entry['split-type']) for entry in test_data])

    dataset = DatasetDict({
        'test': Dataset.from_dict({'rules': [row[0] for row in rows_test],
                                   'labels': [row[1] for row in rows_test],
                                   'theme': [row[2] for row in rows_test],
                                   'split-type': [row[3] for row in rows_test]}),
    })

    return dataset['test']

def load_untyped_dataset(splited_types):
    # Define the parent directory where the JSON files are located
    parent_directory = 'untyped'

    # Initialize an empty dictionary to store the organized data
    data_dict = {}

    # Define the dimensions
    dimensions = ['train', 'test', 'dev']

    # Iterate through each subdirectory
    for subdir in os.listdir(parent_directory):
        subdir_path = os.path.join(parent_directory, subdir)

        # Initialize a dictionary to store the content of JSON files in the current subdirectory
        subdir_dict = {}

        # Iterate through each dimension
        for dimension in dimensions:
            file_name = f"{dimension}.json"
            file_path = os.path.join(subdir_path, file_name)

            # Check if the file exists
            if os.path.exists(file_path):
                # Read the content of the JSON file
                with open(file_path, 'r') as file:
                    content_lines = [json.loads(line) for line in file]
                    subdir_dict[dimension] = content_lines

        # Store the dimension dictionary in the main dictionary with the subdirectory name as the key
        data_dict[subdir] = subdir_dict

    # Create lists to store rows for training, testing, and dev
    rows_train = []
    rows_test = []
    rows_dev = []

    # Iterate through each theme in data_dict
    for theme in data_dict:
        # Extract data for the current theme
        train_data = data_dict[theme]['train']
        test_data = data_dict[theme]['test']
        dev_data = data_dict[theme]['dev']

        random.seed(42)
        random.shuffle(train_data)
        random.shuffle(test_data)
        random.shuffle(dev_data)

        # Combine examples for training, testing, and dev
        rows_train.extend([(entry['rule'], entry['label'], theme) for entry in train_data])
        rows_test.extend([(entry['rule'], entry['label'], theme) for entry in test_data])
        rows_dev.extend([(entry['rule'], entry['label'], theme) for entry in dev_data])

    # Create DatasetDict with 'train', 'test', and 'dev' splits
    dataset = DatasetDict({
        'train': Dataset.from_dict({'rules': [row[0] for row in rows_train],
                                    'labels': [row[1] for row in rows_train],
                                    'theme': [row[2] for row in rows_train]}),
        'dev': Dataset.from_dict({'rules': [row[0] for row in rows_dev],
                                  'labels': [row[1] for row in rows_dev],
                                  'theme': [row[2] for row in rows_dev]}),
        'test': load_split_types_dataset() if splited_types else Dataset.from_dict({
            'rules': [row[0] for row in rows_test],
            'labels': [row[1] for row in rows_test],
            'theme': [row[2] for row in rows_test]})
    })

    return dataset

# Main function to select and load the correct dataset
def load_dataset(dataset_name, splited_types):
    if dataset_name == "ontology_rules":
        dataset = load_ontology_rules_dataset()
    elif dataset_name == "gcn_not_predicted":
        dataset = load_gcn_not_predicted_dataset()
    elif dataset_name == "untyped":
        dataset = load_untyped_dataset(splited_types)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Save dataset
    dataset.save_to_disk(f"saved_datasets/{dataset_name}_dataset")