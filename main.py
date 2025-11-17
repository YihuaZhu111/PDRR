import os
import logging
from datetime import datetime
from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import json
from qa_system import QA_system

def setup_main_logger(args):
    """Set the main program's logger"""
    # Create a log directory dynamically based on the model type and name
    if args.LLM_type == 'gpt':
        model_name = args.engine
    else:
        model_name = args.llama_model_name.split('/')[-1]
    
    log_dir = f'logs/{model_name}'
    os.makedirs(log_dir, exist_ok=True)
   
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'{log_dir}/main_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    # Keep existing arguments
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.6, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0.1, help="the temperature in reasoning stage.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--n_hops", type=int,
                        default=1, help="number of hops for KG expansion")
    parser.add_argument("--llama_model_name", type=str,
                        default="meta-llama/Llama-3.3-70B-Instruct", help="name of the LLM model to use.")  # meta-llama/Llama-3.3-70B-Instruct, meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument("--huggingface_api_keys", type=str,
                        default="", help="if the LLM_type is llama, you need add your own huggingface api keys.")
    parser.add_argument('--openai_api_key', type=str, default='', help='if the LLM_type is gpt, you need add your own openai api keys.')
    parser.add_argument('--engine', type=str, default='gpt-4o', help='if the LLM_type is gpt, you need add your own openai model name.')
    parser.add_argument('--LLM_type', type=str, default='gpt', choices=['gpt', 'llama'])
    parser.add_argument('--question_type_from', type=str, default='LLM', choices=['LLM', 'dataset'], help='If you use question type from LLM.')
    args = parser.parse_args()

    # Set the logger
    logger = setup_main_logger(args)
    
    # Set the API token
    if args.LLM_type == 'llama':    
        if args.huggingface_api_keys:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.huggingface_api_keys
    elif args.LLM_type == 'gpt':
        if args.openai_api_key:
            os.environ["OPENAI_API_KEY"] = args.openai_api_key
    else:
        raise ValueError("Invalid LLM type. Please choose 'gpt' or 'llama'.")
    
    logger.info(f"args: {args}")
    # Initialize the QA system
    qa_system = QA_system(args, logger)
   
    # Generate a filename with timestamp and model information
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_info = args.engine if args.LLM_type == 'gpt' else args.llama_model_name.split('/')[-1]
    
    reasoning_triples_dir = 'reasoning_triples'
    os.makedirs(reasoning_triples_dir, exist_ok=True)
    output_filename_reasoning_triples = os.path.join(reasoning_triples_dir, f"results_{args.dataset}_{args.LLM_type}_{model_info}_{timestamp}_reasoning_triples.jsonl")
  
    # Read the dataset
    with open(output_filename_reasoning_triples, 'a+', encoding="UTF-8") as out_triples:
        datas, question_string = prepare_dataset(args.dataset)
        
        for i in tqdm(datas, total=len(datas)):
            question = i[question_string]
            logger.info(f"question: {question}")
            if 'topic_entity' in i:
                topic_entity = i['topic_entity']
                logger.info(f"topic_entity: {topic_entity}")
            else:
                topic_entity = None
            if 'compositionality_type' in i:
                question_type = i['compositionality_type']
                logger.info(f"question_type: {question_type}")
            else:
                question_type = None

            result, reasoning_triples, question_type_llm = qa_system.answer_question(question, topic_entity, question_type)
            # Use the KG-enhanced QA system
           
            if args.question_type_from == 'LLM':
                out_triples.write(json.dumps({
                    "question": question,
                    "question_type": question_type_llm,
                    "kg_enhanced_result": result,
                    "reasoning_triples": reasoning_triples
                })+'\n')
            elif args.question_type_from == 'dataset':
                out_triples.write(json.dumps({
                    "question": question,
                    "kg_enhanced_result": result,
                    "reasoning_triples": reasoning_triples
                })+'\n')

if __name__ == "__main__":
    main()
