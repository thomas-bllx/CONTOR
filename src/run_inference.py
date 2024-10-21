import os.path

import numpy as np
from deeponto.probe.ontolama.inference import *
from datasets import load_dataset
from openprompt.config import *
from openprompt.data_utils import InputExample
import argparse


def add_label_to_dataset(dataset, task_name):
    premise_name = "v_sub_concept"
    hypothesis_name = "v_super_concept"
    # different data fields for the bimnli dataset
    if "bimnli" in task_name:
        premise_name = "premise"
        hypothesis_name = "hypothesis"

    prompt_samples = []
    for samp in dataset:
        inp = InputExample(text_a=samp[premise_name], text_b=samp[hypothesis_name], label=samp["label"])
        prompt_samples.append(inp)
    return prompt_samples


def run(config, args):
    """Main entry for running the OpenPrompt script.
    """
    # load dataset. The valid_dataset can be None
    Processor = OntoLAMADataProcessor()
    datasets = load_dataset(config.dataset.path, data_dir=config.dataset.path,
                            data_files={'train': 'train.json', 'validation': 'dev.json', 'test': 'test.json'})
    train_dataset = add_label_to_dataset(datasets['train'], config.dataset.task_name)
    valid_dataset = add_label_to_dataset(datasets['validation'], config.dataset.task_name)
    test_dataset = add_label_to_dataset(datasets['test'], config.dataset.task_name)
    del datasets

    # train_dataset, valid_dataset, test_dataset, Processor = OntoLAMADataProcessor.load_inference_dataset(
    #     config, test=args.test is not None or config.learning_setting == "zero_shot"
    # )

    # main
    if config.learning_setting == "full":
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            resume=args.resume,
            test=args.test,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )
    elif config.learning_setting == "few_shot":
        if config.few_shot.few_shot_sampling is None:
            raise ValueError("use few_shot setting but config.few_shot.few_shot_sampling is not specified")
        seeds = config.sampling_from_train.seed
        res = 0
        for seed in seeds:
            if not args.test:
                sampler = FewShotSampler(
                    num_examples_per_label=config.sampling_from_train.num_examples_per_label,
                    also_sample_dev=config.sampling_from_train.also_sample_dev,
                    num_examples_per_label_dev=config.sampling_from_train.num_examples_per_label_dev,
                )
                train_sampled_dataset, valid_sampled_dataset = sampler(
                    train_dataset=train_dataset, valid_dataset=valid_dataset, seed=seed
                )
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    resume=args.resume,
                    test=args.test,
                    train_dataset=train_sampled_dataset,
                    valid_dataset=valid_sampled_dataset,
                    test_dataset=test_dataset,
                )
            else:
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    test=args.test,
                    test_dataset=test_dataset,
                )
            res += result
        res /= len(seeds)
    elif config.learning_setting == "zero_shot":
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            zero=True,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    return res, config, CUR_TEMPLATE, CUR_VERBALIZER


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", help='dataset path', default='../datasets/wine/untyped')
    parser.add_argument("--config_yaml", default='../scripts/config.yaml', type=str, help='the configuration file for this experiment.')
    parser.add_argument("--resume", type=str, help='a specified logging path to resume training.\
               It will fall back to run from initialization if no lastest checkpoint are found.')
    parser.add_argument("--test", type=str, help='a specified logging path to test')
    parser.add_argument("-res_fn", help='result path', default='./result/wine.txt')

    args, _ = parser.parse_known_args()

    config = get_user_config(args.config_yaml)

    add_cfg_to_argparser(config, parser)
    args = parser.parse_args()

    update_cfg_with_argparser(config, args)
    check_config_conflicts(config)

    if not os.path.exists(config.logging.path_base):
        os.makedirs(config.logging.path_base)

    if not os.path.exists(os.path.dirname(args.res_fn)):
        os.makedirs(os.path.dirname(args.res_fn))

    global CUR_TEMPLATE, CUR_VERBALIZER
    # exit()
    # init logger, create log dir and set log level, etc.
    if args.resume and args.test:
        raise Exception("cannot use flag --resume and --test together")
    if args.resume or args.test:
        config.logging.path = EXP_PATH = args.resume or args.test
    else:
        EXP_PATH = config_experiment_dir(config)
        init_logger(
            os.path.join(EXP_PATH, "log.txt"),
            config.logging.file_level,
            config.logging.console_level,
        )
        # save config to the logger directory
        save_config_to_yaml(config)

    config.dataset.path = args.data_path
    res = run(config, args)[0]
    out_str = 'precison = ' + str(np.round(res['precision'], 4)) + '\n' + \
              'recall = ' + str(np.round(res['recall'], 4)) + '\n' + \
              'f1 = ' + str(np.round(res['binary-f1'], 4)) + '\n' + \
              'acc = ' + str(np.round(res['accuracy'], 4)) + '\n\n'
    out_str += str(config) + '\n'
    out_str += str(args) + '\n'

    with open(args.res_fn, 'w', encoding='utf-8') as f:
        f.write(out_str)