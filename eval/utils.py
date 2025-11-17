import json
import re

def prepare_dataset_for_eval(dataset_name, output_file):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    else:
        print("dataset not found, you should pick from {cwq, webqsp}.")
        exit(-1)
    
    output_datas = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Ensure each line's content is properly stripped of whitespace
                    line = line.strip()
                    if line:  # Ensure it's not an empty line
                        data = json.loads(line)
                        output_datas.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:100]}...")
                    print(f"Error message: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

    return datas, question_string, output_datas


def align(dataset_name, question_string, data, ground_truth_datas):

    answer_list = []
    matching_data = [j for j in ground_truth_datas if j[question_string] == data[question_string]]
    if not matching_data:
        return []
    origin_data = matching_data[0]
    
    if dataset_name == 'cwq':
        if 'answers' in origin_data:
            answers = origin_data["answers"]
        else:
            answers = origin_data["answer"]
        if isinstance(answers, str):
            answer_list.append(answers)
        else:
            for answer in answers:
                if isinstance(answer, dict) and 'aliases' in answer:
                    alias = answer['aliases']
                    ans = answer['answer']
                    alias.append(ans)
                    answer_list.extend(alias)
                elif isinstance(answer, str):
                    answer_list.append(answer)
                else:
                    pass
               
    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

    return list(set(answer_list))
    
    
def check_string(string):
    return "{" in string


def clean_results(string):

    nested_matches = re.findall(r'\{\"(.*?)\"\}', string)
    if nested_matches:
        return nested_matches[-1]
    

    matches = re.findall(r'\{(.*?)\}', string)
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    elif len(matches) >= 1:
        return matches[-1]  
    

    lines = string.strip().split('\n')  
    last_line = lines[-1].strip() if lines else ""

    quote_matches = re.findall(r'\"(.*?)\"', last_line)
    if quote_matches:
        return quote_matches[-1]

    if last_line and not re.search(r'[.:]$', last_line) and len(last_line.split()) <= 3:
        return last_line

    return "NULL"




def check_refuse(string):
    refuse_words = ["however", "sorry"]
    return any(word in string.lower() for word in refuse_words)



def exact_match(response, answers):
    
    # Clean response text: remove spaces, convert to lowercase, replace special characters
    clean_result = response.strip().lower()
    clean_result = normalize_text(clean_result)  # Normalize text (handle accent marks, etc.)
    
    # Save a version with spaces for phrase matching
    spaced_result = re.sub(r'[^\w\s]', '', clean_result)  # Remove all non-alphanumeric and non-space characters
    result_words = set(spaced_result.split())
    
    # Version with all spaces removed for string matching
    clean_result = re.sub(r'[^\w\s]', '', clean_result)  # Remove all non-alphanumeric and non-space characters
    clean_result = re.sub(r'\s+', '', clean_result)  # Remove all spaces

    
    for answer in answers:
        # Apply the same cleaning to answers
        clean_answer = answer.strip().lower()
        clean_answer = normalize_text(clean_answer)  # Normalize text
        
        # Save a version with spaces for phrase matching
        spaced_answer = re.sub(r'[^\w\s]', '', clean_answer)  # Remove all non-alphanumeric and non-space characters
        answer_words = set(spaced_answer.split())
        
        # Version with all spaces removed for string matching
        clean_answer = re.sub(r'[^\w\s]', '', clean_answer)  # Remove all non-alphanumeric and non-space characters
        clean_answer = re.sub(r'\s+', '', clean_answer)  # Remove all spaces

        
        # Original matching logic
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
            
        # New phrase matching logic
        # Check if there's sufficient word overlap
        # Overlap module is for the case: e.g., "Peruvian nuevo sol" and "Peruvian sol".
        common_words = result_words.intersection(answer_words)
        if len(common_words) >= 1:  # At least one common word
            # Calculate overlap ratio
            overlap_ratio_result = len(common_words) / len(result_words) if result_words else 0
            overlap_ratio_answer = len(common_words) / len(answer_words) if answer_words else 0
            
            
            # If overlap ratio in either direction exceeds 70%, consider it a successful match
            if overlap_ratio_result >= 0.7 or overlap_ratio_answer >= 0.7:

                return True
    
    return False

def normalize_text(text):
    """
    Convert accent marks and special characters in text to basic ASCII characters
    For example: convert Québec to Quebec
    """
    import unicodedata
    # Decompose Unicode characters into base characters and combining characters
    # Then keep only base characters (ASCII characters)
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def save_result2json(dataset_name, num_right, num_error, total_nums, method):
    results_data = {
        'dataset': dataset_name,
        'method': method,
        'Exact Match': float(num_right/total_nums),
        'Right Samples': num_right,
        'Error Sampels': num_error
    }
    with open('cot_{}_results.json'.format(dataset_name), 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)


def extract_content(s):
    matches = re.findall(r'\{(.*?)\}', s)
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    elif len(matches) >= 1:
        return matches[0]
    else:
        return 'NULL'

