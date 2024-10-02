from transformers import EarlyStoppingCallback, BitsAndBytesConfig, DataCollatorForLanguageModeling, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from datasets import DatasetDict, Dataset, load_from_disk
from peft import get_peft_model, LoraConfig
from unsloth import FastLanguageModel
import random
import torch
import re
import os

def setup_model(model_name, theme, output_dir, prompt, quantiz, dataset_name, splited):

    # List of prompts
    prompts = [
        ('Classify the text into True or False. Reply with only one word: True or False.', 'Determine if the following statement is valid:'),
        ('Assess the validity of the following statement. Reply with only one word: True or False.', 'Determine if the following statement is valid:'),
        ('Assess the validity of the following rule. Reply with only one word: True or False.', 'Determine if the following rule is valid:'),
        ('Classify the text into True or False. Reply with only one word: True or False.', 'Determine if the following is a valid rule:'),
        ('Classify the text into True or False. Reply with only one word: True or False.', 'Determine if the following is valid statement:')
    ]

    ################################################################################################################################################################

    # Function to load the dataset from disk
    def load_dataset_from_disk(dataset_name):
        dataset_path = f"saved_datasets/{dataset_name}_dataset"
        if os.path.exists(dataset_path):
            return load_from_disk(dataset_path)
        else:
            raise FileNotFoundError(f"The dataset {dataset_name} does not exist in {dataset_path}.")

    dataset = load_dataset_from_disk(dataset_name)

    print(dataset)

    ################################################################################################################################################################

    def transform(raw_line):
        # Separate 'body' and 'head'
        body, head = raw_line.replace("body=", "").split(", head=")

        def transform_term(term):
            # Remove the prefix and handle parentheses
            term = term.split(".")[-1]
            elements = re.findall(r'[^()\[\]]+|\([^()]*\)', term)  # Capture elements and contents within parentheses

            # Reverse the order of elements within parentheses and place them in front
            elements_with_parentheses = [el for el in elements if '(' in el][::-1]
            elements_without_parentheses = [el for el in elements if '(' not in el]
            ordered_elements = elements_with_parentheses + elements_without_parentheses

            # Clean and format the elements
            formatted_elements = []
            for element in ordered_elements:
                # Remove parentheses and split by commas
                sub_elements = re.sub(r'[()]', '', element).split(',')
                # Add spaces before uppercase letters and remove excess spaces
                sub_elements = [re.sub(r'(?<=[a-z0-9A-Z])(?=[A-Z])', ' ', sub_el).strip() for sub_el in sub_elements]
                formatted_elements.extend(sub_elements)

            # Construct the final string for the term
            return ' '.join(formatted_elements)
        
        body_terms = [transform_term(part) for part in body.split(', ')]

        if "owl.Bottom" in head:
            return f"{body_terms[0]} implies {body_terms[1]}: contradiction"
        else:
            head_readable = ' '.join(transform_term(part) for part in head.split(', '))
            body_readable = ' and '.join(body_terms)
            return f"{body_readable} implies {head_readable}"

    ################################################################################################################################################################

    # Function to choose the prompt template based on the model
    def chose_prompt(example, label=""):
        if model_name in ["mistralai/Mistral-7B-v0.3", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "mistralai/Mistral-7B-v0.1","lmsys/vicuna-13b-v1.5", "lmsys/vicuna-13b-v1.5-16k", "unsloth/mistral-7b-v0.3-bnb-4bit", "unsloth/mistral-7b-bnb-4bit", "unsloth/gemma-7b-bnb-4bit"]:
            return f'''### Instruction:
{prompts[prompt][0]}

### Question:
{prompts[prompt][1]}
{example}

### Answer:
{label}'''

        elif model_name in ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf']:
            return f'''<s>[INST] <<SYS>>
{prompts[prompt][0]}
<</SYS>>

{prompts[prompt][1]}
{transform(example)}
[/INST]
{label}
{'</s>' if label != "" else ''}'''

        elif model_name == 'togethercomputer/Llama-2-7B-32K-Instruct':
            return f'''[INST]
{prompts[prompt][0]}
{prompts[prompt][1]}
{transform(example)}
[/INST]
{label}'''

        elif model_name in ['mistralai/Mistral-7B-Instruct-v0.2', "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"]:
            return f'''<s>[INST]
{prompts[prompt][0]}
{prompts[prompt][1]}
{transform(example)}
[/INST]
{label}
{'</s>' if label != "" else ''}'''
        
        elif model_name in ["meta-llama/Meta-Llama-3-8B-Instruct", "unsloth/llama-3-8b-Instruct-bnb-4bit"]:
            return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{prompts[prompt][0]}<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompts[prompt][1]}
{example}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{label}{'<|eot_id|>' if label != "" else ''}'''

        elif model_name in ["meta-llama/Meta-Llama-3-8B", "unsloth/llama-3-8b-bnb-4bit"]:
            return f'''<|begin_of_text|>### Instruction:
{prompts[prompt][0]}

### Question:
{prompts[prompt][1]}
{transform(example)}

### Answer:
{label}'''

        elif model_name in ["unsloth/Phi-3-mini-4k-instruct", "unsloth/Phi-3-medium-4k-instruct", "unsloth/Phi-3-medium-4k-instruct-bnb-4bit"]:
            return f'''<s><|user|>
{prompts[prompt][0]}
{prompts[prompt][1]}
{transform(example)}<|end|>
<|assistant|>
{label}{'<|end|>' if label != "" else ''}'''

        elif model_name in ["unsloth/gemma-7b-it-bnb-4bit"]:
            return f'''<start_of_turn>user
{prompts[prompt][0]}
{prompts[prompt][1]}
{transform(example)}<end_of_turn>
<start_of_turn>model
{label}{'<end_of_turn>' if label != "" else ''}'''

    def restructure_and_rename(example, part_name):
        if part_name == "train":
            if example['labels'] == 0:
                example["text"] = chose_prompt(example['rules'],'False')
            else:
                example["text"] = chose_prompt(example['rules'],'True')
        else:
            example["text"] = chose_prompt(example['rules'])
        return example

    for part_name in dataset.keys():
        part_dataset = dataset[part_name] 
        dataset[part_name] = part_dataset.map(lambda example: restructure_and_rename(example, part_name))

    train_dataset = dataset['train']

    # Separate data by labels
    positives = [example for example in train_dataset if example['labels'] == 1]
    negatives = [example for example in train_dataset if example['labels'] == 0]

    # Randomly shuffle the data
    random.seed(42)  # For reproducibility
    random.shuffle(positives)
    random.shuffle(negatives)

    positives_reduced = positives
    negatives_reduced = negatives[:len(positives_reduced)]

    # Merge the reduced data
    train_balanced_data = positives_reduced + negatives_reduced
    random.shuffle(train_balanced_data)  # Shuffle again after merging

    dataset = DatasetDict({
        'train': Dataset.from_dict({'rules': [ex['rules'] for ex in train_balanced_data], 'labels': [ex['labels'] for ex in train_balanced_data], 'theme': [ex['theme'] for ex in train_balanced_data], 'text': [ex['text'] for ex in train_balanced_data]}),
        'dev': dataset['dev'],
        'test': dataset['test'],
    })

    # Check dataset sizes
    print("Total size of balanced training dataset:", len(dataset['train']))
    print("Number of positive examples:", sum(1 for ex in dataset['train'] if ex['labels'] == 1))
    print("Number of negative examples:", sum(1 for ex in dataset['train'] if ex['labels'] == 0))

    ########################################################################################################################### Load the QLoRA model

    if quantiz == "qlora":
        bnb_4bit_compute_dtype = "float16"
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            task_type="CAUSAL_LM",
        )

        model_name = model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
        )

        model = get_peft_model(
            model,
            peft_config,
        )

        output_dir = f"{output_dir}/{model_name.replace('/', '_')}_theme_{theme}_prompt_{prompt}"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    ########################################################################################################################### Load the unsLoth model

    elif quantiz == "unsloth":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 128,
            dtype = None,
            load_in_4bit = True,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0.1, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        output_dir = f"{output_dir}/{model_name.replace('/', '_')}_theme_{theme}_prompt_{prompt}"

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    else:
        raise ValueError("Quantization method not recognized")

    ###########################################################################################################################

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=128)

    dataset = dataset.map(tokenize_function, batched=True)

    train_batch_size = 8
    test_batch_size = 8

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        num_train_epochs=1,
        seed=0,
        do_predict=True,
        predict_with_generate=True,
        generation_max_length=128,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        save_total_limit=2,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train pre-trained model
    print('###Train')
    trainer.train()
    trainer.save_model(output_dir)

    ###########################################################################################################################

    print('###Evaluation')
    # Generate predictions with predict()
    generated_outputs = trainer.predict(dataset['test'], max_new_tokens=1)

    predictions = []
    for elt in generated_outputs.predictions:
        # Filter the array to remove -100
        filtered_array = elt[elt != -100]
        predictions.append(filtered_array)

    # Use the tokenizer to decode the filtered tokens
    decoded_responses = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    preds = []
    for response in decoded_responses:
        last_word = response.strip().split()[-1].lower()
        if last_word == "true":
            preds.append(1)
        elif last_word == "false":
            preds.append(0)
        else:
            preds.append(2)

    # Get the true labels from your test dataset
    labels = dataset['test']['labels']
    themes = dataset['test']['theme']
    if splited:
        split_types = dataset['test']['split-type']

    # Function to compute global metrics
    def compute_global_metrics(predictions, labels):

        accuracy = accuracy_score(labels, predictions)
        recall_w = recall_score(labels, predictions, zero_division=0, average='weighted')
        precision_w = precision_score(labels, predictions, zero_division=0, average='weighted')
        f1_w = f1_score(labels, predictions, zero_division=0, average='weighted')

        recall_mi = recall_score(labels, predictions, zero_division=0, average='micro')
        precision_mi = precision_score(labels, predictions, zero_division=0, average='micro')
        f1_mi = f1_score(labels, predictions, zero_division=0, average='micro')

        recall_ma = recall_score(labels, predictions, zero_division=0, average='macro')
        precision_ma = precision_score(labels, predictions, zero_division=0, average='macro')
        f1_ma = f1_score(labels, predictions, zero_division=0, average='macro')

        return {
            "accuracy": accuracy,
            "precision_w": precision_w,
            "recall_w": recall_w,
            "f1_w": f1_w,
            "precision_mi": precision_mi,
            "recall_mi": recall_mi,
            "f1_mi": f1_mi,
            "precision_ma": precision_ma,
            "recall_ma": recall_ma,
            "f1_ma": f1_ma
        }

    # Function to compute metrics per theme
    def compute_metrics_per_theme(preds, labels, themes):
        metrics_per_theme = {}
        preds_per_theme = {}
        labels_per_theme = {}

        for theme in set(themes):
            theme_indices = [i for i, t in enumerate(themes) if t == theme]
            theme_preds = [preds[i] for i in theme_indices]
            theme_labels = [labels[i] for i in theme_indices]

            # Calculate metrics
            accuracy = accuracy_score(theme_labels, theme_preds)
            recall_w = recall_score(theme_labels, theme_preds, zero_division=0, average='weighted')
            precision_w = precision_score(theme_labels, theme_preds, zero_division=0, average='weighted')
            f1_w = f1_score(theme_labels, theme_preds, zero_division=0, average='weighted')

            recall_mi = recall_score(theme_labels, theme_preds, zero_division=0, average='micro')
            precision_mi = precision_score(theme_labels, theme_preds, zero_division=0, average='micro')
            f1_mi = f1_score(theme_labels, theme_preds, zero_division=0, average='micro')

            recall_ma = recall_score(theme_labels, theme_preds, zero_division=0, average='macro')
            precision_ma = precision_score(theme_labels, theme_preds, zero_division=0, average='macro')
            f1_ma = f1_score(theme_labels, theme_preds, zero_division=0, average='macro')

            # Store metrics, predictions, and labels
            metrics_per_theme[theme] = {
                "accuracy": accuracy,
                "precision_w": precision_w,
                "recall_w": recall_w,
                "f1_w": f1_w,
                "precision_mi": precision_mi,
                "recall_mi": recall_mi,
                "f1_mi": f1_mi,
                "precision_ma": precision_ma,
                "recall_ma": recall_ma,
                "f1_ma": f1_ma
            }
            preds_per_theme[theme] = theme_preds
            labels_per_theme[theme] = theme_labels

        return metrics_per_theme

    # Function to compute metrics per theme_split
    def compute_metrics_per_theme_split(preds, labels, themes, split_types):
        metrics_per_theme = {}

        for theme in set(themes):
            theme_metrics_per_split = {}
            theme_preds = [p for p, t in zip(preds, themes) if t == theme]
            theme_labels = [l for l, t in zip(labels, themes) if t == theme]
            if theme_labels:
                metrics_per_theme[theme] = compute_global_metrics(theme_preds, theme_labels)
            
            for split_type in set(split_types):
                theme_split_preds = [p for p, t, s in zip(preds, themes, split_types) if t == theme and s == split_type]
                theme_split_labels = [l for l, t, s in zip(labels, themes, split_types) if t == theme and s == split_type]
                if theme_split_labels:
                    theme_metrics_per_split[split_type] = compute_global_metrics(theme_split_preds, theme_split_labels)
            metrics_per_theme[theme]['split_types'] = theme_metrics_per_split

        return metrics_per_theme

    # Calculate global metrics
    global_metrics = compute_global_metrics(preds, labels)

    if splited:
        # Calculate global metrics by split-type
        global_metrics_per_split = {}
        for split_type in set(split_types):
            split_type_preds = [p for p, s in zip(preds, split_types) if s == split_type]
            split_type_labels = [l for l, s in zip(labels, split_types) if s == split_type]
            global_metrics_per_split[split_type] = compute_global_metrics(split_type_preds, split_type_labels)

        # Calculate metrics for each theme
        metrics_per_theme = compute_metrics_per_theme_split(preds, labels, themes, split_types)

    # Calculate metrics for each theme
    metrics_per_theme = compute_metrics_per_theme(preds, labels, themes)

    with open(f"result_metric/{theme}_theme_{model_name.replace('/', '_')}.txt", "a", encoding='UTF-8') as fichier:

        print(model_name, " / ", "prompt", prompt, " / ", "dataset size: ", len(positives_reduced) * 2, " / ", "batch size: ", train_batch_size)
        fichier.write(model_name + ' / ' + "prompt " + str(prompt + 1) + ' / ' + "dataset size: " + str(len(positives_reduced) * 2) + ' / ' + "batch size: " + str(train_batch_size) + '\n\n')

        # Display global metrics
        print("Global metrics:")
        fichier.write("Global metrics:\n")
        for metric, value in global_metrics.items():
            print(f"  {metric}: {value}")
            fichier.write(f"  {metric}: {value}" + '\n')
        print()
        fichier.write("\n")

        # Display the results
        for theme in metrics_per_theme:
            print(f"Theme: {theme}")
            fichier.write(f"Theme: {theme}" + '\n')
            print("Metrics:")
            fichier.write("Metrics:\n")
            for metric, value in metrics_per_theme[theme].items():
                print(f"  {metric}: {value}")
                fichier.write(f"  {metric}: {value}" + '\n')
            print()
            fichier.write('\n')

        if splited:
            # Display global metrics by split-type
            print("Global metrics per Split-Type:")
            fichier.write("Global metrics per Split-Type:\n")
            for split_type, metrics in global_metrics_per_split.items():
                print(f" Split Type: {split_type}")
                fichier.write(f" Split Type: {split_type}" + '\n')
                for metric, value in metrics.items():
                    print(f"   {metric}: {value}")
                    fichier.write(f"   {metric}: {value}" + '\n')
                print()
                fichier.write("\n")

            # Display metrics per theme and split-type
            for theme, theme_metrics in metrics_per_theme.items():
                print(f"Theme: {theme}")
                fichier.write(f"Theme: {theme}" + '\n')
                for metric, value in theme_metrics.items():
                    if metric != 'split_types':
                        print(f"  {metric}: {value}")
                        fichier.write(f"  {metric}: {value}" + '\n')
                print("\n  Split-Types:\n")
                fichier.write("\n  Split-Types:\n\n")
                for split_type, metrics in theme_metrics['split_types'].items():
                    print(f"    Split Type: {split_type}")
                    fichier.write(f"    Split Type: {split_type}" + '\n')
                    for metric, value in metrics.items():
                        print(f"      {metric}: {value}")
                        fichier.write(f"      {metric}: {value}" + '\n')
                    print()
                    fichier.write("\n")

        try:
            report = classification_report(labels, preds, target_names=['False', 'True'], zero_division=0)
        except:
            # If the model predicts a third class
            report = classification_report(labels, preds, target_names=['False', 'True', 'Other'], zero_division=0)
        
        print(report)
        fichier.write(report)
        fichier.write('\n\n')