import argparse
from utils import *
import logging
import os
from datetime import datetime

if __name__ == '__main__':
    # Set up logging
    log_dir = "logs/evaluation"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default="ToG_cwq.json", help="the output file name.")
    parser.add_argument("--constraints_refuse", type=bool,
                        default=True, help="LLM may have refuse erorr, enable this option to skip current sample.")
    parser.add_argument("--method", type=str,
                        default="new_method", help="evaluation method name")
    args = parser.parse_args()

    logging.info(f"Starting evaluation on dataset: {args.dataset}")
    logging.info(f"Output file: {args.output_file}")
    
    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)
    logging.info(f"Total samples: {len(output_datas)}")
    
    # Initialize statistics for different question types
    type_stats = {
        "Chain Structure": {"right": 0, "total": 0},
        "Parallel Structure": {"right": 0, "total": 0},
        "Comparative Structure": {"right": 0, "total": 0},
        "Superlative Structure": {"right": 0, "total": 0}
    }
    
    num_right = 0
    num_error = 0
    for idx, data in enumerate(output_datas, 1):
        logging.info(f"\nSample {idx}:")
        logging.info(f"Question: {data.get('question', 'N/A')}")
        answers = align(args.dataset, question_string, data, ground_truth_datas)
        question_type = ground_truth_datas[idx-1]["compositionality_type"]

        results = data["kg_enhanced_result"]

        logging.info(f"Expected answer: {answers}")
        logging.info(f"Model result: {results}")

        if question_type:
            if question_type == 'composition':
                question_type = "Chain Structure"
            elif question_type == 'conjunction':
                question_type = "Parallel Structure"
            elif question_type == 'comparative':
                question_type = "Comparative Structure"
            elif question_type == 'superlative':
                question_type = "Superlative Structure"
            else:
                question_type = "Chain Structure"
        logging.info(f"Question type: {question_type}")     
     
        # Update total count for question type
        if question_type in type_stats:
            type_stats[question_type]["total"] += 1

        is_correct = False
        
        # Check if result is None
        if results is None:
            logging.info("Result is None, marking as error and skipping current sample")
            continue
            
        if args.constraints_refuse and check_string(results) and "refuse" in results.lower():
            logging.info("Skipping current sample (constraint refusal)")
            continue
           
        response = clean_results(results)
        logging.info(f"Cleaned result: {response}")
        if response == "NULL":
            response = results
            logging.info("Cleaned result is NULL, using original result")
            
        if exact_match(response, answers):
            num_right += 1
            is_correct = True
            logging.info("✓ Match successful")
        else:
            num_error += 1
            logging.info("✗ Match failed")

        # Update correct count for question type
        if is_correct and question_type in type_stats:
            type_stats[question_type]["right"] += 1
        
        logging.info(f"Current accuracy: {float(num_right/(idx)):.4f}")

    final_accuracy = float(num_right/len(output_datas))
    logging.info("\nFinal evaluation results:")
    logging.info(f"Accuracy: {final_accuracy:.4f}")
    logging.info(f"Correct: {num_right}, Error: {num_error}")
    
    # Log accuracy for different question types
    logging.info("\nAccuracy for different question types:")
    for q_type, stats in type_stats.items():
        if stats["total"] > 0:
            type_accuracy = float(stats["right"] / stats["total"])
            logging.info(f"{q_type} accuracy: {type_accuracy:.4f} ({stats['right']}/{stats['total']})")
        else:
            logging.info(f"{q_type} accuracy: N/A (0/0)")

    save_result2json(args.dataset, num_right, num_error, len(output_datas), method=args.method)
    logging.info("Evaluation completed, results saved")  