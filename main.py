import argparse
from load_my_dataset import load_dataset
from finetuning_LLMs import setup_model

def main():
    parser = argparse.ArgumentParser(description='Finetune and Evaluate LLMs')
    parser.add_argument('--theme', type=str, default='all', help='Choose theme or all')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--quantize', type=str, choices=['unsloth', 'qlora'], required=True, help='unsloth or qlora')
    parser.add_argument('--prompt', type=int, required=True, help='Prompt number')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--splited', type=int, required=True, help='Splited types or not')
    parser.add_argument('--output', type=str, required=True, help='Output directory')

    args = parser.parse_args()

    print(f"Arguments received: theme={args.theme}, model_name={args.model_name}, quantize={args.quantize}, prompt={args.prompt}, dataset={args.dataset}, splited={args.splited}, output={args.output}")

    theme = args.theme
    model_name = args.model_name
    quantize = args.quantize
    prompt = args.prompt
    dataset_name = args.dataset
    splited = args.splited
    output_dir = args.output

    print("Loading the dataset...")
    load_dataset(dataset_name, splited_types=splited)
    print(f"Dataset {dataset_name} loaded successfully.")

    print("Initializing the model...")
    setup_model(model_name, theme, output_dir, prompt, quantize, dataset_name, splited)
    print("Model initialized successfully.")

if __name__ == "__main__":
    print("Starting the script...")
    main()