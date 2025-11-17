from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sentence_transformers import SentenceTransformer
from freebase_func import *
from utils import *
from typing import Dict, List, Set
import json
import re
from new_prompt import *


class QA_system:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.n_hops = args.n_hops
        self.question_type_from = args.question_type_from # llm or dataset

        # Initialize Llama model
        self.LLM_type = args.LLM_type
        if self.LLM_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(args.llama_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                args.llama_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif self.LLM_type == 'gpt':
            self.engine = args.engine
       
        # self.opeani_api_keys = args.opeani_api_keys
        self.bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def log_step(self, step_name: str, data: any):
        """Log step results to logger"""
        try:
            if isinstance(data, (list, dict)):
                log_content = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                log_content = str(data)
            
            self.logger.info(f"\n{'='*50}\n{step_name}:\n{log_content}\n{'='*50}")
        except Exception as e:
            self.logger.error(f"Error logging: {str(e)}")

    def get_question_type_llm(self, question: str) -> str:
        """Analyze question type"""
        """
        Input: question: str
        Output: question_type: str
               Chain Structure or Parallel Structure
        """
        prompt = get_question_type_prompt(question)
 
        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant that analyzes the question and determines its type."},
            *message_question_type,
            {"role": "user", "content": prompt}
        ]
       

        if self.LLM_type == 'llama':
            response = run_llm_llama(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens=self.args.max_length,
                tokenizer=self.tokenizer,
                model=self.model
            )
        elif self.LLM_type == 'gpt':
            response = run_llm_gpt(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens=self.args.max_length,
                engine=self.engine
            )
        self.log_step("Question type analysis response", response)  

        pattern = r"\{([^}]*)\}"
        matches = re.findall(pattern, response)
        
        # If matches found and contains chain or parallel (case insensitive)
        for match in matches:
            if "chain" in match.lower():
                return "Chain Structure"
            elif "parallel" in match.lower():
                return "Parallel Structure"
            else:
                return "Chain Structure"

    def question_decomposition_llm(self, question: str, question_type: str) -> List[Dict]:
        """Decompose question"""
        """
        Input: question: str
        Output: question_decomposition format: List[Dict]
        Output example:
        [
            {"head": "entity1", "relation": "relation", "tail": "entity2"}
            ...
        ]
        Input example:
        Parallel Structure:
        [
            {"head": "country#1", "relation": "borders", "tail": "France"},
            {"head": "country#1", "relation": "contains an airport that serves", "tail": "Nijmegen"}
            ...
        ]
        Chain Structure:
        [
            {"head": "Rift Valley Province", "relation": "is located in", "tail": "nation#1"},
            {"head": "nation#1", "relation": "uses currency", "tail": "currency#1"}
            ...
        ]
        """ 
        prompt = get_question_decomposition_prompt(question, question_type)
     
        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant that decomposes the question into triples."},
            *message_question_decomposition,
            {"role": "user", "content": prompt}
        ]
      

        if self.LLM_type == 'llama':
            response = run_llm_llama(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens=self.args.max_length,
                tokenizer=self.tokenizer,
                model=self.model
            )
        elif self.LLM_type == 'gpt':
            response = run_llm_gpt(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens=self.args.max_length,
                engine=self.engine
            )
        self.log_step("Question decomposition response", response)
        
        # Parse triples from response
        try:
            # Find square brackets and their content
            triples_pattern = r'\[\s*\{.*?\}\s*\]'
            triples_match = re.search(triples_pattern, response, re.DOTALL)
            
            if triples_match:
                triples_str = triples_match.group(0)
                # Try to parse JSON
                triples = json.loads(triples_str)
                self.log_step("Modified format triples", triples)
                return triples
            else:
                # If no triple format found, try to find single triples
                single_triple_pattern = r'\{\s*"head":\s*"[^"]*",\s*"relation":\s*"[^"]*",\s*"tail":\s*"[^"]*"\s*\}'
                single_matches = re.findall(single_triple_pattern, response)
                
                if single_matches:
                    triples = [json.loads(match) for match in single_matches]
                    self.log_step("Parsed single triples", triples)
                    return triples
                else:
                    self.logger.warning("Unable to parse triples from response")
                    return []
        except Exception as e:
            self.logger.error(f"Error parsing triples: {str(e)}")
            return []

    def single_chain_reasoning_triples_search_prune(self, triple: Dict, topic_entity: List[Dict]) -> List[Dict]:
        """Single chain reasoning triple search pruning"""
        """
        Input: triple: Dict
        Output: triples: List[Dict]
        Input triple format:
            First triple format: {"head": "entity1", "relation": "relation", "tail": "entity2"}
            Second triple format: {
                    'head_entity_id': 'xxx',
                    'head_entity_freebase_name': 'xxx',
                    'relation': 'xxx',
                    'tail': 'xxx'
                }
                or
                {
                    'head': 'xxx',
                    'relation': 'xxx',
                    'tail_entity_id': 'xxx',
                    'tail_entity_freebase_name': 'xxx'
                }
        Output triples format example: List[List[Dict]]
        """
        
        # Check input triple format
        if "head_entity_id" in triple:
            # Second format, directly use head_entity_id
            self.log_step("Using second format triple", triple)
            entity_start = [{
                'entity_id': triple['head_entity_id'],
                'entity_original_name': triple['head_entity_freebase_name'],
                'entity_freebase_name': triple['head_entity_freebase_name']
            }]
        elif "tail_entity_id" in triple:
            # Third format, directly use tail_entity_id
            self.log_step("Using third format triple", triple)
            entity_start = [{
                'entity_id': triple['tail_entity_id'],
                'entity_original_name': triple['tail_entity_freebase_name'],
                'entity_freebase_name': triple['tail_entity_freebase_name']
            }]
        else:
            # First format, need to query entity_id
            self.log_step("Using first format triple", triple)
            # Extract entities from triple, prioritize entities without # symbol
            head = triple.get("head", "")
            tail = triple.get("tail", "")
           
            # Check if head and tail contain # symbol
            head_has_hash = "#" in head
            tail_has_hash = "#" in tail
            
            # If only one entity doesn't contain # symbol, use that entity
            if head_has_hash and not tail_has_hash:
                entity_str = tail
            elif not head_has_hash and tail_has_hash:
                entity_str = head
            else:
                # If both have or both don't have # symbol, use head
                entity_str = head
            self.log_step("entity_str", entity_str)
            
            entity_start = []
            if topic_entity:
                # Convert all topic entities to list format
                sorted_topic_entity = []
                for topic_entity_item in topic_entity:
                    sorted_topic_entity.append({
                        'topic_entity_id': topic_entity_item['topic_entity_id'],
                        'topic_entity_name': topic_entity_item['topic_entity_name']
                    })

                # Find the most similar topic entity
                max_similarity = 0
                most_similar_topic_entity = None
                
                for topic_entity_item in sorted_topic_entity:
                    similarity = compute_similarity(entity_str.lower(), topic_entity_item['topic_entity_name'].lower())
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_topic_entity = topic_entity_item
                
                # If found topic entity with sufficient similarity, add to entity_start
                if max_similarity > 0.3 and most_similar_topic_entity:
                    entity_start.append(
                        {
                            'entity_id': most_similar_topic_entity['topic_entity_id'],
                            'entity_original_name': most_similar_topic_entity['topic_entity_name'],
                            'entity_freebase_name': most_similar_topic_entity['topic_entity_name'],
                            'score': max_similarity
                        }
                    )
                    self.log_step("topic_entity search result entity_start", entity_start)
                else:
                    entity_start = self.entity_id_search(entity_str)
                    self.log_step("SPARQL search serial entity ID search result", entity_start)
            else:
                entity_start = self.entity_id_search(entity_str)
                self.log_step("SPARQL search serial entity ID search result", entity_start)
            
        self.log_step("entity_start", entity_start)
        
        all_relation_infos = []
        total_relations = []
        if entity_start:
            entity_relations = self.relation_search(entity_start, parallel_reasoning = False, is_head = True, args = self.args)
            #self.log_step("entity_relations", entity_relations)
            if entity_relations:
                # Add relation info to total list
                all_relation_infos.extend(entity_relations)
                # Also maintain a list containing only relations for filtering
                total_relations.extend([info['relation'] for info in entity_relations])
            
            self.log_step("total_relations", total_relations)
            # Get filtered relations

            filtered_relations = self.relations_prune_with_triple(triple, total_relations, parallel_reasoning=False, args=self.args)
            self.log_step("filtered_relations", filtered_relations)

            filtered_entity_relation = [
                info for info in all_relation_infos 
                if info['relation'] in filtered_relations
            ]
            self.log_step("filtered_entity_relation", filtered_entity_relation)

            connected_triples = self.triple_search(filtered_entity_relation)  
            if connected_triples and len(connected_triples) > 2:
            # new_triples: List[List[Dict]]
                new_triples = self.triples_prune_with_triple(triple, connected_triples, is_unnamed_entity=False, args=self.args) 
                new_triples = [[element_triple] for element_triple in new_triples]
                self.log_step("new_triples", new_triples)
            elif len(connected_triples) ==1 or len(connected_triples) == 2:
                new_triples = connected_triples
                new_triples = [[element_triple] for element_triple in new_triples]  
                self.log_step("new_triples", new_triples)
            else:
                return []
        
            if new_triples:    # List[List[Dict]]
                for i, list_triple in enumerate(new_triples):      # List[Dict]
                    # For each triple chain, we take the first triple
                    if list_triple:
                        unnamed_entity = {}
                        dict_triple = list_triple[0]  # Dict
                        if dict_triple['head_entity_freebase_name'] == "UnName_Entity":
                            unnamed_entity = {
                                'entity_id': dict_triple['head_entity_id'],
                                'entity_freebase_name': dict_triple['head_entity_freebase_name'],
                                'original_entity': dict_triple['tail_entity_freebase_name']

                            }
                        elif dict_triple['tail_entity_freebase_name'] == "UnName_Entity":
                            unnamed_entity = {
                                'entity_id': dict_triple['tail_entity_id'],
                                'entity_freebase_name': dict_triple['tail_entity_freebase_name'],
                                'original_entity': dict_triple['head_entity_freebase_name']
                            }
                        else:
                            pass
                        if len(unnamed_entity) > 0:
                            self.log_step("unnamed_entity", unnamed_entity)
                            all_unnamed_entity_relation_infos = []
                            total_unnamed_entity_relations = []
                            #is_head = unnamed_entity['is_head']
                            
                            unnamed_entity_relations = self.relation_search(unnamed_entity, parallel_reasoning = False, is_head = True, args = self.args)
                            self.log_step("unnamed_entity_relation_search", unnamed_entity_relations)
                            
                            if unnamed_entity_relations:
                                    # Add relation info to total list
                                    all_unnamed_entity_relation_infos.extend(unnamed_entity_relations)
                                    # Also maintain a list containing only relations for filtering
                                    total_unnamed_entity_relations.extend([info['relation'] for info in unnamed_entity_relations])
                            self.log_step("total_unnamed_entity_relations", total_unnamed_entity_relations)
                            
                            filtered_unnamed_entity_relations = self.relations_prune_with_triple(triple, total_unnamed_entity_relations, parallel_reasoning = False, args=self.args)
                            self.log_step("filtered_unnamed_entity_relations", filtered_unnamed_entity_relations)
                            
                            filtered_unnamed_entity_relation = [
                                info for info in all_unnamed_entity_relation_infos 
                                if info['relation'] in filtered_unnamed_entity_relations
                            ]
                            self.log_step("filtered_unnamed_entity_relation", filtered_unnamed_entity_relation)
                            
                            unnamed_entity_triples = self.triple_search(filtered_unnamed_entity_relation)
                            self.log_step("unnamed_entity_triples", unnamed_entity_triples)
                            filter_original_unnamed_entity_triples = []
                            if unnamed_entity_triples:
                                filter_original_unnamed_entity_triples = [
                                    triple for triple in unnamed_entity_triples 
                                    if not (triple['head_entity_freebase_name'] == unnamed_entity['original_entity'] or 
                                            triple['tail_entity_freebase_name'] == unnamed_entity['original_entity'])
                                ]
                                self.log_step("filter_original_unnamed_entity_triples", filter_original_unnamed_entity_triples)
                            
                            # If unnamed_entity_triples is empty, return directly
                            if filter_original_unnamed_entity_triples and len(filter_original_unnamed_entity_triples) > 1:
                            # new_triples: List[List[Dict]] 
                                new_unnamed_entity_triples = self.triples_prune_with_triple(triple, filter_original_unnamed_entity_triples, is_unnamed_entity = True, args=self.args) 
                                self.log_step("new_unnamed_entity_triples", new_unnamed_entity_triples)
                            elif len(filter_original_unnamed_entity_triples) ==1:
                                new_unnamed_entity_triples = filter_original_unnamed_entity_triples
                                self.log_step("new_unnamed_entity_triples", new_unnamed_entity_triples)
                            else:
                                return []
                            if new_unnamed_entity_triples and len(new_unnamed_entity_triples) == 1:
                                new_unnamed_entity_triple = new_unnamed_entity_triples[0]
                                list_triple.append(new_unnamed_entity_triple)

                                new_triples[i] = list_triple
                                self.log_step(f"Updated triple chain {i}", list_triple) 
                            else:
                                pass
                        else:
                            pass
        else:
            return []
        
        return new_triples   # List[List[Dict]]
        """
        Now new_triples format is:
        [
            [
                {'head_entity_id': 'xxx',
                 'head_entity_original_name': 'xxx',
                 'head_entity_freebase_name': 'xxx',
                 'relation': 'xxx','tail_entity_id': 'xxx',
                 'tail_entity_freebase_name': 'xxx',
                 'score': x
                },
                {'head_entity_id': 'xxx',
                 'head_entity_freebase_name': 'UnName_Entity',
                 'relation': 'xxx','tail_entity_id': 'xxx',
                 'tail_entity_freebase_name': 'xxx',
                 'score': x
                }
                ...
            ]
            ...
        ]
        """

    def chain_reasoning(self, original_triples: List[Dict], topic_entity: List[Dict]) -> List[List[Dict]]:
        """
        Implement chain reasoning, replace placeholder entities in original triple chain with actual entities, and build complete reasoning chain
        
        Args:
            original_triples: Original triple chain, format:
                [
                    {"head": "Rift Valley Province", "relation": "is located in", "tail": "nation#1"},
                    {"head": "nation#1", "relation": "uses currency", "tail": "currency#1"},
                    ...
                ]
            topic_entity: Topic entity information
            
        Returns:
            reasoning_chains: Multiple complete reasoning chains, each chain contains a series of connected triples
            reasoning_chains format example:
            [
                [
                    {'head_entity_id': 'xxx','head_entity_original_name': 'xxx','head_entity_freebase_name': 'xxx','relation': 'xxx','tail_entity_id': 'xxx','tail_entity_freebase_name': 'xxx','score': x},
                    {'head_entity_id': 'xxx','head_entity_original_name': 'xxx','head_entity_freebase_name': 'xxx','relation': 'xxx','tail_entity_id': 'xxx','tail_entity_freebase_name': 'xxx','score': x},
                ]
                ...
            ]
        """
        # If original triples is empty, return empty list directly
        if not original_triples:
            return []
        
        # Get first triple
        first_triple = original_triples[0]
        self.log_step("First triple", first_triple)
        
        # Use single chain reasoning function to get possible results for first triple
        first_new_triples_lists = self.single_chain_reasoning_triples_search_prune(first_triple, topic_entity)
        
        """
        first_new_triples_lists format:  List[List[Dict]]
        [
            [
                {
                    'head_entity_id': 'xxx',
                    'head_entity_original_name': 'xxx',
                    'head_entity_freebase_name': 'xxx',
                    'relation': 'xxx',
                    'tail_entity_id': 'xxx',
                    'tail_entity_freebase_name': 'xxx',
                    'score': x
                },
                {
                    'head_entity_id': 'xxx',
                    'head_entity_original_name': 'xxx',
                    'head_entity_freebase_name': 'xxx',
                    'relation': 'xxx',
                    'tail_entity_id': 'xxx',
                    'tail_entity_freebase_name': 'xxx',
                    'score': x
                }
            ],
            ...
        ]
        """
        self.log_step("First triple reasoning result", first_new_triples_lists)
        
        # If no results found, return empty list
        if not first_new_triples_lists:
            return []
        
        # If original triple chain has only one triple, return results directly
        if len(original_triples) == 1:
            return first_new_triples_lists
        
        # Store all possible reasoning chains
        all_reasoning_chains = []
        
        # For each possible result list of first triple, build a reasoning chain
        for first_result_list in first_new_triples_lists:
            # Get last triple in list as basis for connecting next triple
            if not first_result_list:
                continue
                
            # Save complete first_result_list for final reasoning chain
            complete_first_result = first_result_list.copy()
            
            # Use last triple for connection
            first_result = first_result_list[-1]
            
            # Get entity names from first triple
            first_head = first_triple.get("head", "")
            first_tail = first_triple.get("tail", "")

            if first_result.get('tail_entity_original_name'):  
                new_entity_name = first_result.get('head_entity_freebase_name', '')
                new_entity_id = first_result.get('head_entity_id', '')
            elif first_result.get('head_entity_original_name'):  
                new_entity_name = first_result.get('tail_entity_freebase_name', '')
                new_entity_id = first_result.get('tail_entity_id', '')
            elif first_result['head_entity_freebase_name'] == "UnName_Entity":
                new_entity_name = first_result.get('tail_entity_freebase_name', '')
                new_entity_id = first_result.get('tail_entity_id', '')
            elif first_result['tail_entity_freebase_name'] == "UnName_Entity":
                new_entity_name = first_result.get('head_entity_freebase_name', '')
                new_entity_id = first_result.get('head_entity_id', '')
            else:
                new_entity_name = first_result.get('tail_entity_freebase_name', '')
                new_entity_id = first_result.get('tail_entity_id', '')
            
            # Get second triple
            second_triple = original_triples[1]
            
            # Determine which entity in second triple needs replacement (entity with # symbol)
            second_head = second_triple.get("head", "")
            second_tail = second_triple.get("tail", "")
           
            # Create new triple, replace entity with # symbol
            new_second_triple = second_triple.copy()
            if second_head == first_tail:  # If head entity of second triple is same as tail entity of first triple
                # Remove original head with # symbol and ignore return value
                new_second_triple.pop("head", None)
                # Add new entity information
                new_second_triple['head_entity_id'] = new_entity_id
                new_second_triple['head_entity_freebase_name'] = new_entity_name
            elif second_head == first_head:
                # Remove original head with # symbol and ignore return value
                new_second_triple.pop("head", None)
                # Add new entity information
                new_second_triple['head_entity_id'] = new_entity_id
                new_second_triple['head_entity_freebase_name'] = new_entity_name
            elif second_tail == first_tail:
                # Remove original tail with # symbol and ignore return value
                new_second_triple.pop("tail", None)
                # Add new entity information
                new_second_triple['tail_entity_id'] = new_entity_id
                new_second_triple['tail_entity_freebase_name'] = new_entity_name
            elif second_tail == first_head:  # If tail entity of second triple is same as head entity of first triple
                # Remove original tail with # symbol and ignore return value
                new_second_triple.pop("tail", None)
                # Add new entity information
                new_second_triple['tail_entity_id'] = new_entity_id
                new_second_triple['tail_entity_freebase_name'] = new_entity_name
            else:
                # Default case: replace head entity
                new_second_triple.pop("head", None)
                # Add new entity information
                new_second_triple['head_entity_id'] = new_entity_id
                new_second_triple['head_entity_freebase_name'] = new_entity_name
            self.log_step("new_second_triple", new_second_triple)
            # Recursively process remaining triple chain
            if len(original_triples) > 2:
                # Build new triple chain starting from second triple
                remaining_chains = self.chain_reasoning([new_second_triple] + original_triples[2:], topic_entity)
                self.log_step("remaining_chains", remaining_chains)
                
                if remaining_chains:
                    for chain in remaining_chains:
                        all_reasoning_chains.append(complete_first_result + chain)
                else:
                    
                    empty_chain = [{}] * (len(original_triples) - 1)
                    all_reasoning_chains.append(complete_first_result + empty_chain)
            else:
                # if len(original_triples) == 2, then process the second triple
                second_results_lists = self.single_chain_reasoning_triples_search_prune(new_second_triple, topic_entity)
                
                # if second_results_lists is not empty, then combine the current result with the second triple result
                if second_results_lists:
                    for second_result_list in second_results_lists:
                        all_reasoning_chains.append(complete_first_result + second_result_list)
                else:
                    # if second_results_lists is empty, then use empty object {} to replace
                    all_reasoning_chains.append(complete_first_result + [{}])
        
        self.log_step("all_reasoning_chains", all_reasoning_chains)
        return all_reasoning_chains 

    def best_chain_selection_llm(self, reasoning_chain: List[List[Dict]], question: str) -> List[Dict]:
        """
        Implement parallel reasoning, process multiple independent triples simultaneously
        
        Args:
            original_triples: Original triple list, format:
                [
                    {"head": "country#1", "relation": "borders", "tail": "France"},
                    {"head": "country#1", "relation": "contains an airport that serves", "tail": "Nijmegen"},
                    ...
                ]
            topic_entity: Topic entity information
            
        Returns:
            reasoning_results: Multiple reasoning result combinations, each combination contains results for all triples
            reasoning_results format example:
            [
                [
                    {'head_entity_id': 'xxx','head_entity_original_name': 'xxx','head_entity_freebase_name': 'xxx','relation': 'xxx','tail_entity_id': 'xxx','tail_entity_freebase_name': 'xxx','score': x},
                    {'head_entity_id': 'xxx','head_entity_original_name': 'xxx','head_entity_freebase_name': 'xxx','relation': 'xxx','tail_entity_id': 'xxx','tail_entity_freebase_name': 'xxx','score': x},
                ]
                ...
            ]
        """
        reasoning_chain_str = ""
        for i, chain in enumerate(reasoning_chain, 1):
            reasoning_chain_str += f"chain {i}: "
            
            chain_parts = []
            for triple in chain:
                # Extract head, relation and tail of the triple
                head = triple.get("head_entity_freebase_name", "")
                relation = triple.get("relation", "")
                tail = triple.get("tail_entity_freebase_name", "")
                
                # Format triple as {head, relation, tail}
                triple_str = f"{{{head}, {relation}, {tail}}}"
                chain_parts.append(triple_str)
            
            # Connect all triples in the same chain with commas
            reasoning_chain_str += ", ".join(chain_parts) + "\n"
        self.log_step("reasoning_chain_str", reasoning_chain_str)

        prompt = get_best_chain_selection_prompt(reasoning_chain_str, question)
        self.log_step("best_chain_selection_prompt", prompt)
        messages = [
            {"role": "system", "content": "You are a system designed to select the best reasoning chain to answer the question from the given reasoning chains."},
            *message_best_chain_selection,
            {"role": "user", "content": prompt}
        ]
        self.log_step("best_chain_selection_messages", messages)
        if self.LLM_type == 'llama':
            response = run_llm_llama(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens=self.args.max_length,
                tokenizer=self.tokenizer,
                model=self.model
            )
        elif self.LLM_type == 'gpt':
            response = run_llm_gpt(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens=self.args.max_length,
                engine=self.engine
            )
        self.log_step("best_chain_selection_response", response)

        # Use regular expression to match content within square brackets
        pattern = r'\[(.*?)\]'
        match = re.search(pattern, response)

        if match:
            # Extract matched content, including square brackets
            best_reasoning_chain_str = f"[{match.group(1)}]"
        else:
            # If no match found, try to find content after "chain"
            chain_pattern = r'chain \d+: (.*?)(?:\.|$)'
            chain_match = re.search(chain_pattern, response)
            if chain_match:
                best_reasoning_chain_str = chain_match.group(1)
            else:
                best_reasoning_chain_str = reasoning_chain_str

        self.log_step("extracted_best_chain", best_reasoning_chain_str)
        
        return best_reasoning_chain_str

    def chain_question_answer(self, best_reasoning_chain_str: str, question: str, topic_entity: List[Dict], question_decomposition_triples: List[Dict]) -> str:
        """
        Chain reasoning answer
        Input: best_reasoning_chain_str: str
        example:
        best_reasoning_chain_str:
        [{Rift Valley Province, is located in, France}, {France, uses currency, Euro}]
        question:
        What is the currency of France?
        """
        if topic_entity:
            # Extract 'topic_entity_name' from topic_entity and convert to str
            topic_entity_str = ', '.join([entity['topic_entity_name'] for entity in topic_entity])
        else:
            topic_entity_str = ''
            self.log_step("topic_entity_str", topic_entity_str)

        # Convert question_decomposition_triples to str
        question_decomposition_triples_str = ''
        for triple in question_decomposition_triples:
            question_decomposition_triples_str += f'{{{triple["head"]}, {triple["relation"]}, {triple["tail"]}}}, '
        
        # Remove the last comma and space
        if question_decomposition_triples_str.endswith(', '):
            question_decomposition_triples_str = question_decomposition_triples_str[:-2]
        self.log_step("chain_question_decomposition_triples_str", question_decomposition_triples_str)

        if best_reasoning_chain_str.startswith("["):
            best_reasoning_chain_str = best_reasoning_chain_str.replace("[", "").replace("]", "")
        else:
            best_reasoning_chain_str = best_reasoning_chain_str
        prompt = get_chain_question_answer_prompt(best_reasoning_chain_str, question, topic_entity_str, question_decomposition_triples_str)
        self.log_step("chain_question_answer_prompt", prompt)
        messages = [
            {"role": "system", "content": "You are a system designed to answer the question from the given reasoning chain and your own knowledge. Please think step by step and follow the reasoning steps carefully."},
            *message_chain_question_answer,
            {"role": "user", "content": prompt}
        ]
        #self.log_step("chain_question_answer_messages", messages)
        if self.LLM_type == 'llama':
            response = run_llm_llama(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens=2048,
                tokenizer=self.tokenizer,
                model=self.model
            )
        elif self.LLM_type == 'gpt':
            response = run_llm_gpt(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens=2048,
                engine=self.engine
            )
        self.log_step("chain_question_answer_response", response)
        return response

    def entity_id_search(self, entity: str):
        """Entity ID search"""
        """
        Input: entity: str
        Output: 
        align_entities example:
        [
            {
                'entity_id': 'm.02m4qz',
                'entity_original_name': 'mascot',
                'entity_freebase_name': 'mascot',
                'score': 0.9999999999999999
            }
        ]
        """
        #entity = list(entity)
        #entity = [word.lower().strip() for word in entity]  # for example: "Lou Seal" -> "lou seal"
        
        align_entities = query_freebase_entity(entity)
        return align_entities

    def relation_search(self, entity, parallel_reasoning: bool, is_head: bool, args):
        """
        Query all relations related to the entity

        Args:
            entity: entity information dictionary, format:
                {
                    'entity_id': 'm.xxx',
                    'entity_original_name': 'xxx',
                    'entity_freebase_name': 'xxx'
                }
            args: parameter configuration

        Returns:
            entity_and_relations: relation information list, format:
                [
                    {
                        'relation': 'xxx',
                        'is_head': 1,  # 1 represents head_relation, 0 represents tail_relation
                        'entity_id': 'm.xxx',
                        'entity_original_name': 'xxx',
                        'entity_freebase_name': 'xxx'
                    },
                    ...
                ]
        """
        
        # Check if entity is a list, if so, take the first element
        if isinstance(entity, list) and len(entity) > 0:
            entity = entity[0]  # Replace the entire entity with the first element of the list
        
        entity_id = entity['entity_id']
        entity_and_relations = []

        # Get head relations given head entity, find all relations pointing out from the head entity
        sparql_relations_extract_head = sparql_head_relations % (entity_id)
        head_relations = execurte_sparql(sparql_relations_extract_head)
        head_relations = replace_relation_prefix(head_relations)
        
        # Get tail relations given tail entity, find all relations pointing to the tail entity
        sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
        tail_relations = execurte_sparql(sparql_relations_extract_tail)
        tail_relations = replace_relation_prefix(tail_relations)

        # If you need to remove unnecessary relations
        if args.remove_unnecessary_rel:
            head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
            tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]

        # Remove duplicates
        head_relations = list(set(head_relations))
        tail_relations = list(set(tail_relations))

        if not parallel_reasoning:
            # If it is serial reasoning, output all head_relations and tail_relations
            # Add information to head relations
            for relation in head_relations:
                entity_and_relations.append({
                    'relation': relation,
                    'is_head': 1,
                    'entity_id': entity['entity_id'],
                    'entity_original_name': entity.get('entity_original_name', ''),
                    'entity_freebase_name': entity.get('entity_freebase_name', '')
                })

            # Add information to tail relations
            for relation in tail_relations:
                entity_and_relations.append({
                    'relation': relation,
                    'is_head': 0,
                    'entity_id': entity['entity_id'],
                    'entity_original_name': entity.get('entity_original_name', ''),
                    'entity_freebase_name': entity.get('entity_freebase_name', '')
                })
            self.log_step("chain reasoning entity_and_relations", entity_and_relations)
        else:
            if is_head:
                for relation in head_relations:
                    entity_and_relations.append({
                        'relation': relation,
                        'is_head': 1,
                        'entity_id': entity['entity_id'],
                        'entity_original_name': entity.get('entity_original_name', ''),
                        'entity_freebase_name': entity.get('entity_freebase_name', '')
                    })
            else:
                for relation in tail_relations:
                    entity_and_relations.append({
                        'relation': relation,
                        'is_head': 0,
                        'entity_id': entity['entity_id'],
                        'entity_original_name': entity.get('entity_original_name', ''),
                        'entity_freebase_name': entity.get('entity_freebase_name', '')
                    })
            self.log_step("parallel reasoning entity_and_relations", entity_and_relations)
        return entity_and_relations

    def relations_prune_with_triple(self, triple: Dict, relations: list, parallel_reasoning: bool, args: dict) -> list:
        """
        Use LLM to batch filter relations related to the question

        Input triple format 1: {"head": "entity1", "relation": "relation", "tail": "entity2"}
        Input triple format 2: 
        Second triple format:   {
                                    'head_entity_id': 'xxx',
                                    'head_entity_freebase_name': 'xxx',
                                    'relation': 'xxx',
                                    'tail': 'xxx'
                                }
                                or
                                {
                                    'head': 'xxx',
                                    'relation': 'xxx',
                                    'tail_entity_id': 'xxx',
                                    'tail_entity_freebase_name': 'xxx'
                                }
        
        Input relations format: ["relation1","relation2", "relation3",...]
        Returns:
            filtered_relations: ["relation1","relation3",...]
        """
        if not relations:
            return []
        
        # Before building the prompt for batch evaluation, first remove duplicates from relations
        unique_relations = list(set(relations))  # Use set to remove duplicates

        # Build the prompt for batch evaluation
        #relations_text = "\n".join([f"Relation {i+1}: {relation}" for i, relation in enumerate(relations)])
        relations_text = "\n".join([f"{i+1}. {relation}" for i, relation in enumerate(unique_relations)])
        if triple.get('head_entity_id'):
            filter_triple = f"{triple['head_entity_freebase_name']}, {triple['relation']}, {triple['tail']}"
        elif triple.get('tail_entity_id'):
            filter_triple = f"{triple['head']}, {triple['relation']}, {triple['tail_entity_freebase_name']}"
        else: 
            filter_triple = f"{triple['head']}, {triple['relation']}, {triple['tail']}"
        self.log_step("filter_triple", filter_triple)
        
        prompt = get_filter_relations_with_triple_prompt(filter_triple, relations_text)
        #self.log_step("relations_prune_with_triple prompt", prompt)
        messages = [
            {"role": "system", "content": "You are a system designed to evaluate the relevance of multiple relations to the triple."},
            *message_filter_relations_with_triple,
            {"role": "user", "content": prompt}
        ]
        #self.log_step("relation批量评估messages", messages)
        
        # Get LLM response
        if self.LLM_type == 'llama':
            response = run_llm_llama(
                messages=messages,
                temperature=args.temperature_reasoning,
                max_tokens= args.max_length,
                tokenizer=self.tokenizer,
                model=self.model
            )
        elif self.LLM_type == 'gpt':
            response = run_llm_gpt(
                messages=messages,
                temperature=args.temperature_reasoning,
                max_tokens=args.max_length,
                engine=self.engine
            )
        self.log_step("relation批量评估response", response)
        # Parse response to get scores
        def extract_scores(response_text):
            scores = {}
            
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check if it is a numbered line (more lenient match)
                if re.match(r'^\d+\.', line) or '(Score:' in line:
                    # Try multiple pattern matches
                    
                    # Pattern 1: Standard curly brace format {relation (Score: 0.5)}
                    pattern1 = r"\{([^}]+?)\s*\(Score:\s*([\d.]+)\)"
                    
                    # Pattern 2: Double asterisk + curly brace format **{relation (Score: X.X)}**
                    pattern2 = r"\*\*\{([^}]+?)\s*\(Score:\s*([\d.]+)\)\}\*\*"
                    
                    # Pattern 3: Unsigned format relation (Score: X.X)
                    pattern3 = r"([^{*]+?)\s*\(Score:\s*([\d.]+)\)"
                    
                    # Pattern 4: Double asterisk format **relation (Score: X.X)**
                    pattern4 = r"\*\*([^*]+?)\s*\(Score:\s*([\d.]+)\)\*\*"
                    
                    # Pattern 5: Number format 1. {relation} (Score: X.X)
                    pattern5 = r"\d+\.\s*\{([^}]+)\}\s*\(Score:\s*([\d.]+)\)"
                    
                    # Pattern 6: Colon format relation (Score: X.X): description
                    pattern6 = r"([^:]+?)\s*\(Score:\s*([\d.]+)\):"
                    
                    # Pattern 7: Simple format relation (Score: X.X)
                    pattern7 = r"^([^(]+?)\s*\(Score:\s*([\d.]+)\)"
                    
                    # Pattern 8: Star pattern variant: **relation** (Score: 0.5)
                    pattern8 = r"\*\*([^*]+?)\*\*\s*\(Score:\s*([\d.]+)\)"
                    
                    # Pattern 9: New - Star-surrounded content: **{relation (Score**: 0.2)}: description
                    pattern9 = r"\*\*\{([^}]+?)\s*\(Score\*\*:\s*([\d.]+)\)"
                    
                    # Pattern 10: New - Star-interrupted in front of score: **{relation (Score**: X.X)
                    pattern10 = r"\*\*\{([^}]+?)\s*\(Score\*\*:\s*([\d.]+)\)"

                    matched = False
                    # Try all patterns
                    for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9, pattern10]:
                        matches = re.findall(pattern, line)
                        if matches:
                            relation, score = matches[0]
                            # Clean the prefix number in the relation name
                            relation = re.sub(r'^\d+\.\s*', '', relation.strip())
                            # Remove curly braces
                            relation = relation.replace('{', '').replace('}', '')
                            
                            try:
                                score = float(score)
                                if 0 <= score <= 1:
                                    scores[relation.strip()] = score
                                    matched = True
                                    break  # Exit the pattern loop after finding a match
                            except (ValueError, AttributeError) as e:
                                self.logger.info(f"handle line '{line}' error: {str(e)}")
                    
                    # If none of the above patterns match, try a more general pattern
                    if not matched and "(Score:" in line:
                        # General pattern: try to extract any content containing (Score:X.X)
                        generic_pattern = r"(.*?)\s*\(Score:\s*([\d.]+)\)"
                        generic_matches = re.findall(generic_pattern, line)
                        if generic_matches:
                            relation = generic_matches[0][0].strip()
                            # If the content contains a colon, only take the part before the colon
                            if ":" in relation and not relation.startswith(":"):
                                relation = relation.split(":", 1)[0].strip()
                            # Clean the relation name
                            relation = re.sub(r'^\d+\.\s*', '', relation)
                            relation = re.sub(r'[{}*]', '', relation).strip()
                            try:
                                score = float(generic_matches[0][1])
                                if 0 <= score <= 1:
                                    scores[relation] = score
                            except (ValueError, AttributeError) as e:
                                self.logger.info(f"handle line '{line}' error: {str(e)}")
            return scores
           

        # Extract scores and filter relations
        scores = extract_scores(response)  # Call the function directly, do not call itself inside the function
        
        # Create a list of relations and scores, ensuring that the relations are not repeated
        relation_scores = []
        seen_relations = set()  # Used to track already added relations
        
        if parallel_reasoning:
            for relation in relations:
                # Check if the relation is in the score dictionary and has not been added to the result
                if relation in scores and scores[relation] > 0 and relation not in seen_relations:
                    relation_scores.append((relation, scores[relation]))
                    seen_relations.add(relation)  # Mark the relation as added
                    self.logger.info(f"relation '{relation}' score: {scores[relation]}")
            
            # Sort by score in descending order
            relation_scores.sort(key=lambda x: x[1], reverse=True)
            filtered_relations = [relation for relation, _ in relation_scores[:1]]  #Here modify top1
            self.logger.info(f"parallel reasoning, take the relation with the highest score: {filtered_relations}")
        else:
            for relation in relations:
                # Check if the relation is in the score dictionary and has not been added to the result
                if relation in scores and scores[relation] > 0 and relation not in seen_relations:
                    relation_scores.append((relation, scores[relation]))
                    seen_relations.add(relation)  # Mark the relation as added
                    self.logger.info(f"relation '{relation}' score: {scores[relation]}")
            
          
            relation_scores.sort(key=lambda x: x[1], reverse=True)
            filtered_relations = [relation for relation, _ in relation_scores[:4]]  
            self.logger.info(f"serial reasoning, take the relation with the highest score: {filtered_relations}")
        return filtered_relations
        """
        Example of the final return format:
        ['sports.sports_championship_event.champion', 'sports.sports_team.championships']
        """

    def triple_search(self, entity_and_relations: List[Dict]) -> List[Dict]:
        """
        Query all relations related to the entity
        Example of input:
        [
            {
                'relation': 'sports.sports_championship_event.champion',
                'is_head': 1,  # 1 represents head_relation, 0 represents tail_relation
                'entity_id': 'm.01xrsx',
                'entity_original_name': '2003 Rugby World Cup',
                'entity_freebase_name': '2003 Rugby World Cup'
            },
            ...
        ]
        """
        # head_entity_relations is (head_entity, relation, ?)
        head_entity_relations = [entity_relation for entity_relation in entity_and_relations if entity_relation['is_head'] == 1]
        # tail_entity_relations is (? , relation, tail_entity)
        tail_entity_relations = [entity_relation for entity_relation in entity_and_relations if entity_relation['is_head'] == 0]
        
        # Add the following code before returning total_relations to build triples
        triples = []
        # unname_entity_triples = []
        # Process the triples of head relations (entity_id as the subject)
        for head_entity_relation in head_entity_relations:
            entity_id = head_entity_relation['entity_id']
            relation = head_entity_relation['relation']
            tail_entities = entity_search(entity_id, relation, head=True) # Get all tail entities corresponding to the relation, return all entity_id
            logger.info(f"entity_search returned tail entity id: {tail_entities}")
            for tail_entity in tail_entities:
                tail_name = id2entity_name_or_type(tail_entity)
                if entity_id and relation and tail_entity:
                    #if tail_name != "UnName_Entity":
                    triples.append({
                        'head_entity_id': entity_id,
                        'head_entity_original_name': head_entity_relation['entity_original_name'],
                        'head_entity_freebase_name': id2entity_name_or_type(entity_id),
                        'relation': relation,
                        'tail_entity_id': tail_entity,
                        'tail_entity_freebase_name': tail_name
                    })
                    
        logger.info(f"After processing head_relations, the triples returned: {triples}")
        
        for tail_entity_relation in tail_entity_relations:
            entity_id = tail_entity_relation['entity_id']
            relation = tail_entity_relation['relation']
            head_entities = entity_search(entity_id, relation, head=False)
            logger.info(f"entity_search returned head entity id: {head_entities}")
            for head_entity in head_entities:
                head_name = id2entity_name_or_type(head_entity)
                if head_entity and relation and entity_id:
                    # if head_name != "UnName_Entity":
                    triples.append({
                        'head_entity_id': head_entity,
                        'head_entity_freebase_name': head_name,
                        'relation': relation,
                        'tail_entity_id': entity_id,
                        'tail_entity_original_name': tail_entity_relation['entity_original_name'],
                        'tail_entity_freebase_name': id2entity_name_or_type(entity_id)
                    })
                    
       
        logger.info(f"After processing tail_relations, the triples returned: {triples}")
        return triples
        """
        Example of the final return format:
        [
            {
                'head_entity_id': 'm.02m4qz',
                'head_entity_original_name': 'mascot',
                'head_entity_freebase_name': 'mascot',
                'relation': 'instance of',
                'tail_entity_id': 'm.02m4qz',
                'tail_entity_freebase_name': 'mascot'
            },
            {
                'head_entity_id': 'm.02m4qz',
                'head_entity_freebase_name': 'mascot',
                'relation': 'instance of',
                'tail_entity_id': 'm.02m4qz',
                'tail_entity_original_name': 'mascot',
                'tail_entity_freebase_name': 'mascot'
            },
            ...
        ]
        """

    def triples_prune_with_triple(self, filter_triple: Dict, triples: List[Dict], is_unnamed_entity:bool, args: Dict) -> List[Dict]:
        """Use LLM to filter the relevance of triples
        
        Args:
            Input filter_triple format 1: {"head": "entity1", "relation": "relation", "tail": "entity2"}
            Input filter_triple format 2: 
            Second triple format:   {
                                    'head_entity_id': 'xxx',
                                    'head_entity_freebase_name': 'xxx',
                                    'relation': 'xxx',
                                    'tail': 'xxx'
                                }
                                or
                                {
                                    'head': 'xxx',
                                    'relation': 'xxx',
                                    'tail_entity_id': 'xxx',
                                    'tail_entity_freebase_name': 'xxx'
                                }
            Input triples format: 
            [
                {
                    'head_entity_id': 'm.02m4qz',
                    'head_entity_original_name': 'mascot',
                    'head_entity_freebase_name': 'mascot',
                    'relation': 'instance of',
                    'tail_entity_id': 'm.02m4qz',
                    'tail_entity_freebase_name': 'mascot'
                }
                ...
            ]
            Output new_triples format:
            [
                {
                    'head_entity_id': 'm.02m4qz',
                    'head_entity_original_name': 'mascot',
                    'head_entity_freebase_name': 'mascot',
                    'relation': 'instance of',
                    'tail_entity_id': 'm.02m4qz',
                    'tail_entity_freebase_name': 'mascot',
                    'score': 0.6
                }
                ...
            ]
        """
        if filter_triple.get('head_entity_id'):
            filter_triple = f"{filter_triple['head_entity_freebase_name']}, {filter_triple['relation']}, {filter_triple['tail']}"
        elif filter_triple.get('tail_entity_id'):
            filter_triple = f"{filter_triple['head']}, {filter_triple['relation']}, {filter_triple['tail_entity_freebase_name']}"
        else: 
            filter_triple = f"{filter_triple['head']}, {filter_triple['relation']}, {filter_triple['tail']}"
        self.log_step("filter_triple", filter_triple)
        # Build the string of triples
        triple_strs = []
        for i, triple in enumerate(triples, 1):
            triple_str = f"{i}. {triple['head_entity_freebase_name']}, {triple['relation']}, {triple['tail_entity_freebase_name']}"
            triple_strs.append(triple_str)
        # Build the prompt for batch evaluation
        triples_text = "\n".join(triple_strs)
        self.log_step("triples_text", triples_text)
        prompt = get_filter_triples_with_triple_prompt(filter_triple, triples_text)
        #self.log_step("filter_triples_with_triple prompt", prompt)
        messages = [
            {"role": "system", "content": "You are a system designed to evaluate the relevance of triples to filter_triple. You must analyze how each triple relates to the filter_triple and provide a relevance score between 0 and 1 based on specific guidelines."},
            *message_filter_triples_with_triple,
            {"role": "user", "content": prompt}
        ]
        #self.log_step("filter_triples_with_triple messages", messages)
        # Get LLM response
        if self.LLM_type == 'llama':
            response = run_llm_llama(
                messages=messages,
                temperature=args.temperature_reasoning,
                max_tokens=1024,
                tokenizer=self.tokenizer,
                model=self.model
            )
        elif self.LLM_type == 'gpt':
            response = run_llm_gpt(
                messages=messages,
                temperature=args.temperature_reasoning,
                max_tokens=1024,
                engine=self.engine
            )
        self.log_step("filter_triples_with_triple response", response)
        
        # Check if response is None
        if response is None:
            self.logger.warning("LLM returned an empty response, unable to extract scores")
            # If there is no response, return the first two triples (if any)
            top_k = min(2, len(triples))
            new_triples = []
            for i in range(top_k):
                triple = triples[i]
                if 'head_entity_original_name' in triple and triple['head_entity_original_name']:
                    new_triples.append({
                        'head_entity_id': triple['head_entity_id'],
                        'head_entity_original_name': triple['head_entity_original_name'],
                        'head_entity_freebase_name': triple['head_entity_freebase_name'],
                        'relation': triple['relation'],
                        'tail_entity_id': triple['tail_entity_id'],
                        'tail_entity_freebase_name': triple['tail_entity_freebase_name'],
                        'score': 0.5  # Default medium score
                    })
                elif 'tail_entity_original_name' in triple and triple['tail_entity_original_name']:
                    new_triples.append({
                        'head_entity_id': triple['head_entity_id'],
                        'head_entity_freebase_name': triple['head_entity_freebase_name'],
                        'relation': triple['relation'],
                        'tail_entity_id': triple['tail_entity_id'],
                        'tail_entity_original_name': triple['tail_entity_original_name'],
                        'tail_entity_freebase_name': triple['tail_entity_freebase_name'],
                        'score': 0.5  # Default medium score
                    })
            return new_triples
        
        # Parse the response to get scores
        def extract_scores(response_text):
            # Store the content of triples and their scores
            triple_scores = {}
            
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # try multiple patterns to extract the content of triples and their scores
                try:
                    # Pattern 1: Standard curly brace format {head, relation, tail. (Score: X.X)}
                    pattern1 = r"\{([^}]+?)\.\s*\(Score:\s*([\d.]+)\)"
                    
                    # Pattern 2: Double asterisk + curly brace format **{head, relation, tail. (Score: X.X)}**
                    pattern2 = r"\*\*\{([^}]+?)\.\s*\(Score:\s*([\d.]+)\)\}\*\*"
                    
                    # Pattern 3: Format without symbols head, relation, tail. (Score: X.X)
                    pattern3 = r"([^{*]+?)\.\s*\(Score:\s*([\d.]+)\)"
                    
                    # Pattern 4: Double asterisk format **head, relation, tail. (Score: X.X)**
                    pattern4 = r"\*\*([^*]+?)\.\s*\(Score:\s*([\d.]+)\)\*\*"
                    
                    # Pattern 5: Format with number 1. {head, relation, tail} (Score: X.X)
                    pattern5 = r"\d+\.\s*\{([^}]+)\}\s*\(Score:\s*([\d.]+)\)"
                    
                    # Pattern 6: Colon format entity, relation, entity (Score: X.X): description
                    pattern6 = r"([^:]+?)\s*\(Score:\s*([\d.]+)\):"
                    
                    # Pattern 7: Simple format entity, relation, entity (Score: X.X)
                    pattern7 = r"^([^(]+?)\s*\(Score:\s*([\d.]+)\)"

                    # Pattern 8: New - Star-surrounded content: **{relation (Score**: 0.2)}: description
                    pattern8 = r"\*\*\{([^}]+?)\s*\(Score\*\*:\s*([\d.]+)\)"
                    
                    matched = False
                    triple_content = None
                    score = None
                    
                    # try all patterns
                    for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8]:
                        matches = re.findall(pattern, line)
                        if matches:
                            triple_content = matches[0][0].strip()
                            # Remove the possible prefix number
                            triple_content = re.sub(r'^\d+\.\s*', '', triple_content)
                            # Remove the curly braces
                            triple_content = triple_content.replace('{', '').replace('}', '')
                            score = float(matches[0][1])
                            if 0 <= score <= 1:
                                matched = True
                                break
                    
                    # If the triple content and score are found
                    if matched and triple_content and score is not None:
                        # Clean the triple content
                        triple_content = triple_content.strip()
                        # Store the content of triples and their scores
                        triple_scores[triple_content] = score
                        self.logger.info(f"Extracted the score of triple '{triple_content}': {score}")
                    
                    # If none of the above patterns match, try a more general pattern
                    if not matched and "(Score:" in line:
                        # General pattern: try to extract any content containing (Score:X.X)
                        generic_pattern = r"(.*?)\s*\(Score:\s*([\d.]+)\)"
                        generic_matches = re.findall(generic_pattern, line)
                        if generic_matches:
                            triple_content = generic_matches[0][0].strip()
                            # If the content contains a colon, only take the part before the colon
                            if ":" in triple_content and not triple_content.startswith(":"):
                                triple_content = triple_content.split(":", 1)[0].strip()
                            score = float(generic_matches[0][1])
                            if 0 <= score <= 1:
                                triple_scores[triple_content] = score
                                self.logger.info(f"Extracted the score of triple '{triple_content}': {score}")
                
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error processing line '{line}': {str(e)}")
            
            self.logger.info(f"Extracted scores dictionary: {triple_scores}")
            return triple_scores
        
        # Extract scores and filter triples
        triple_scores = extract_scores(response)
        if not triple_scores:
            return []
        
        scored_triples = []
        
        # Combine triples and their corresponding scores
        for triple in triples:
            # Build the string representation of the triple
            triple_str = f"{triple['head_entity_freebase_name']}, {triple['relation']}, {triple['tail_entity_freebase_name']}"
            
            # Find the score of the triple
            score = 0  # Default score is 0
            for content, content_score in triple_scores.items():
                # Check if the triple content matches
                if triple_str.lower() in content.lower() or content.lower() in triple_str.lower():
                    score = content_score
                    break
            
            # Add to the score list
            scored_triples.append((score, triple))
        
        # Sort the triples by score in descending order
        scored_triples.sort(key=lambda x: -x[0])
        self.log_step("Sorted triples", scored_triples)
        
        new_triples = []

        if not is_unnamed_entity:
            top_k = min(2, len(scored_triples))
            # If top_k is 1, only keep the triple with the highest score
            for score, triple in scored_triples[:top_k]:
                # Use the in operator to check if the key exists
                if 'head_entity_original_name' in triple and triple['head_entity_original_name']:
                    new_triples.append({
                        'head_entity_id': triple['head_entity_id'],
                        'head_entity_original_name': triple['head_entity_original_name'],
                        'head_entity_freebase_name': triple['head_entity_freebase_name'],
                        'relation': triple['relation'],
                        'tail_entity_id': triple['tail_entity_id'],
                        'tail_entity_freebase_name': triple['tail_entity_freebase_name'],
                        'score': score
                    })
                elif 'tail_entity_original_name' in triple and triple['tail_entity_original_name']:
                    new_triples.append({
                        'head_entity_id': triple['head_entity_id'],
                        'head_entity_freebase_name': triple['head_entity_freebase_name'],
                        'relation': triple['relation'],
                        'tail_entity_id': triple['tail_entity_id'],
                        'tail_entity_original_name': triple['tail_entity_original_name'],
                        'tail_entity_freebase_name': triple['tail_entity_freebase_name'],
                        'score': score
                    })
                else:
                    pass
        else:
            top_k = min(1, len(scored_triples))
            for score, triple in scored_triples[:top_k]:
                new_triples.append({
                    'head_entity_id': triple['head_entity_id'],
                    'head_entity_freebase_name': triple['head_entity_freebase_name'],
                    'relation': triple['relation'],
                    'tail_entity_id': triple['tail_entity_id'],
                    'tail_entity_freebase_name': triple['tail_entity_freebase_name'],
                    'score': score
                })
        self.log_step("Filtered triples", new_triples)
        return new_triples

    def single_parallel_reasoning_search_prune(self, triple: Dict, topic_entity: List[Dict]) -> List[Dict]:
        """Single reasoning chain search and filtering"""
        """
        Input triple example:
            {"head": "country#1", "relation": "borders", "tail": "France"},
        Output triples example:
            Output triples format: List[Dict] 
            example:
            [
                {
                    'head_entity_id': 'xxx',
                    'head_entity_freebase_name': 'Germany',
                    'relation': 'borders',
                    'tail_entity_id': 'xxx',
                    'tail_entity_freebase_name': 'France'
                }
                ...
            ]
        """
        # The first format, need to query entity_id 
        self.log_step("并行推理格式使用第一种格式的triple", triple)
        # Extract entities from the triple, prioritize entities without the # symbol
        head = triple.get("head", "")
        tail = triple.get("tail", "")
        
        # Check if head and tail contain the # symbol
        head_has_hash = "#" in str(head) if head is not None else False
        tail_has_hash = "#" in str(tail) if tail is not None else False
        
        is_head = False # Whether it is head
        # If only one entity does not contain the # symbol, use that entity
        if head_has_hash and not tail_has_hash:
            entity_str = tail
            is_head = False
        elif not head_has_hash and tail_has_hash:
            entity_str = head
            is_head = True
        else:
            # If both have or do not have the # symbol, use head
            entity_str = tail
            is_head = False
        self.log_step("entity_str", entity_str)
        self.log_step("is_head", is_head)
        new_triples = []
        entity_start = []
        if topic_entity:
            # Convert all topic entities to a list format
            sorted_topic_entity = []
            for topic_entity_item in topic_entity:
                sorted_topic_entity.append({
                    'topic_entity_id': topic_entity_item['topic_entity_id'],
                    'topic_entity_name': topic_entity_item['topic_entity_name']
                })

            # Find the topic entity with the highest similarity
            max_similarity = 0
            most_similar_topic_entity = None
            
            for topic_entity_item in sorted_topic_entity:
                if not isinstance(entity_str, str):
                    entity_str = str(entity_str)
                similarity = compute_similarity(entity_str.lower(), topic_entity_item['topic_entity_name'].lower())
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_topic_entity = topic_entity_item
           
            # If a topic entity with sufficient similarity is found, add it to entity_start
            if max_similarity > 0.3 and most_similar_topic_entity:
                entity_start.append(
                    {
                        'entity_id': most_similar_topic_entity['topic_entity_id'],
                        'entity_original_name': most_similar_topic_entity['topic_entity_name'],
                        'entity_freebase_name': most_similar_topic_entity['topic_entity_name']
                    }
                )
                self.log_step("entity_start", entity_start)
            else:
                entity_start = self.entity_id_search(entity_str)
                self.log_step("SPARQL search parallel_entity_start result", entity_start)
        else:
            entity_start = self.entity_id_search(entity_str)
            self.log_step("SPARQL search parallel_entity_start result", entity_start)
        
        all_relation_infos = []
        total_relations = []
        if entity_start:
            entity_relations = self.relation_search(entity_start, parallel_reasoning=False, is_head = is_head, args=self.args)
            self.log_step("entity_relations", entity_relations)
            if entity_relations:
                # Add the relation information to the total list
                all_relation_infos.extend(entity_relations)
                # Maintain a list containing only relations for filtering
                total_relations.extend([info['relation'] for info in entity_relations])
            
            self.log_step("total_relations", total_relations)
            # Get filtered relations
            filtered_relations = self.relations_prune_with_triple(triple, total_relations, parallel_reasoning=True, args=self.args)
            self.log_step("filtered_relations", filtered_relations)

            filtered_entity_relation = [
                info for info in all_relation_infos 
                if info['relation'] in filtered_relations
            ]
            self.log_step("filtered_entity_relation", filtered_entity_relation)

            new_triples = self.triple_search(filtered_entity_relation)   # List[Dict]
            
            if new_triples:
                # Check if "UnName_Entity" is included in the first five triples
                has_unnamed_entity = False
                check_count = min(10, len(new_triples))  # Ensure it does not exceed the list length
                
                for i, element_triple in enumerate(new_triples[:check_count]):
                    if (element_triple.get('head_entity_freebase_name') == "UnName_Entity" or 
                        element_triple.get('tail_entity_freebase_name') == "UnName_Entity"):
                        has_unnamed_entity = True
                        self.log_step(f"parallel_reasoning found UnName_Entity in the {i+1}/{len(new_triples)} triple", element_triple)
                        break
                if has_unnamed_entity:
                    new_triples = [[element_triple] for element_triple in new_triples]  # List[List[Dict]]
                    j = 0 # Control the search length, only search 10
                    for i, list_triple in enumerate(new_triples):      # List[Dict]
                        if list_triple and j < 10:
                            unnamed_entity = {}
                            dict_triple = list_triple[0]  # Dict
                            if dict_triple['head_entity_freebase_name'] == "UnName_Entity":
                                unnamed_entity = {
                                    'entity_id': dict_triple['head_entity_id'],
                                    'entity_freebase_name': dict_triple['head_entity_freebase_name'],
                                    'original_entity': dict_triple['tail_entity_freebase_name']
                                }
                            elif dict_triple['tail_entity_freebase_name'] == "UnName_Entity":
                                unnamed_entity = {
                                    'entity_id': dict_triple['tail_entity_id'],
                                    'entity_freebase_name': dict_triple['tail_entity_freebase_name'],
                                    'original_entity': dict_triple['head_entity_freebase_name']
                                }
                            else:
                                pass
                            if len(unnamed_entity) > 0:
                                j += 1
                                self.log_step("unnamed_entity", unnamed_entity)
                                all_unnamed_entity_relation_infos = []
                                total_unnamed_entity_relations = []
                                #is_head = unnamed_entity['is_head']
                                # Use parallel_reasoning to adjust so that unnamed_entity does not pass the relation back to the original entity
                                unnamed_entity_relations = self.relation_search(unnamed_entity, parallel_reasoning = False, is_head = True, args = self.args)
                                self.log_step("unnamed_entity_relation_search", unnamed_entity_relations)
                                
                                if unnamed_entity_relations:
                                    # Add the relation information to the total list
                                    all_unnamed_entity_relation_infos.extend(unnamed_entity_relations)
                                    # Maintain a list containing only relations for filtering
                                    total_unnamed_entity_relations.extend([info['relation'] for info in unnamed_entity_relations])
                                self.log_step("total_unnamed_entity_relations", total_unnamed_entity_relations)
                                
                                filtered_unnamed_entity_relations = self.relations_prune_with_triple(triple, total_unnamed_entity_relations, parallel_reasoning = False, args=self.args)
                                self.log_step("filtered_unnamed_entity_relations", filtered_unnamed_entity_relations)
                                
                                filtered_unnamed_entity_relation = [
                                    info for info in all_unnamed_entity_relation_infos 
                                    if info['relation'] in filtered_unnamed_entity_relations
                                ]
                                self.log_step("filtered_unnamed_entity_relation", filtered_unnamed_entity_relation)
                                
                                unnamed_entity_triples = self.triple_search(filtered_unnamed_entity_relation)
                                self.log_step("unnamed_entity_triples", unnamed_entity_triples)
                                filter_original_unnamed_entity_triples = []
                                if unnamed_entity_triples:
                                    filter_original_unnamed_entity_triples = [
                                        triple for triple in unnamed_entity_triples 
                                        if not (triple['head_entity_freebase_name'] == unnamed_entity['original_entity'] or 
                                                triple['tail_entity_freebase_name'] == unnamed_entity['original_entity'])
                                    ]
                                    self.log_step("filter_original_unnamed_entity_triples", filter_original_unnamed_entity_triples)
                                
                                # If unnamed_entity_triples is empty, return directly
                                if filter_original_unnamed_entity_triples and len(filter_original_unnamed_entity_triples) > 1:
                                # new_triples: List[List[Dict]] 
                                    new_unnamed_entity_triples = self.triples_prune_with_triple(triple, filter_original_unnamed_entity_triples, is_unnamed_entity = True, args=self.args) 
                                    self.log_step("new_unnamed_entity_triples", new_unnamed_entity_triples)
                                elif len(filter_original_unnamed_entity_triples) ==1:
                                    new_unnamed_entity_triples = filter_original_unnamed_entity_triples
                                    self.log_step("new_unnamed_entity_triples", new_unnamed_entity_triples)
                                else:
                                    return []
                                if new_unnamed_entity_triples and len(new_unnamed_entity_triples) == 1:
                                    new_unnamed_entity_triple = new_unnamed_entity_triples[0]
                                    list_triple.append(new_unnamed_entity_triple)

                                    new_triples[i] = list_triple
                                    self.log_step(f"Updated the {i}th triple chain", list_triple) 
                                else:
                                    pass
                            else:
                                pass
            """
            # Parallel Reasoning does not need to filter triples, all triples need to be retained
            if connected_triples:
                new_triples = self.triples_prune_with_triple(triple, connected_triples, self.args) 
                self.log_step("new_triples", new_triples)
            else:
                pass
            """
        else:
            pass
        if new_triples:
            self.log_step("new_triples", new_triples)
            return new_triples
        else:
            return []

    def parallel_reasoning(self, triples: List[Dict], topic_entity: List[Dict]) -> List[List[Dict]]:
        """Multiple reasoning chains search and filtering"""
        """
        Input triples example:
            [
                {"head": "country#1", "relation": "borders", "tail": "France"},
                {"head": "country#1", "relation": "contains an airport that serves", "tail": "Nijmegen"}
            ]
        Output triples format: List[List[Dict]] 
        example:
            [
                [
                    {
                    'head_entity_id': 'xxx',
                    'head_entity_freebase_name': 'Germany',
                    'relation': 'borders',
                    'tail_entity_id': 'xxx',
                    'tail_entity_original_name': 'France',
                    'tail_entity_freebase_name': 'France'
                    },
                    {
                        'head_entity_id': 'xxx',
                        'head_entity_freebase_name': 'Spain',
                        'relation': 'borders',
                        'tail_entity_id': 'xxx',
                        'tail_entity_original_name': 'France',
                        'tail_entity_freebase_name': 'France'
                    },
                ]
                ...
            ]
        """
        reasoning_triples = []
        for triple in triples:
            new_triples = self.single_parallel_reasoning_search_prune(triple, topic_entity)
            reasoning_triples.append(new_triples)
        self.log_step("reasoning_triples", reasoning_triples)
        return reasoning_triples

    def parallel_question_answer(self, reasoning_triples, question: str, topic_entity: List[Dict], question_decomposition_triples: List[Dict]) -> str:
        """Parallel reasoning chain answer"""
        """
        Input triples format: List[List[Dict]] or List[List[List[Dict]]]
        example:
            [
                [
                    {
                    'head_entity_id': 'xxx',
                    'head_entity_freebase_name': 'Germany',
                    'relation': 'borders',
                    'tail_entity_id': 'xxx',
                        'tail_entity_freebase_name': 'France'
                    },
                    {
                        'head_entity_id': 'xxx',
                        'head_entity_freebase_name': 'Spain',
                        'relation': 'borders',
                        'tail_entity_id': 'xxx',
                        'tail_entity_freebase_name': 'France'
                    },
                    ...
                ]
                ...
            ]
        """
        if topic_entity:
            # Extract the 'topic_entity_name' from topic_entity and convert it to a str
            topic_entity_str = ', '.join([entity['topic_entity_name'] for entity in topic_entity])
        else:
            topic_entity_str = ''
            self.log_step("topic_entity_str", topic_entity_str)

        # Convert question_decomposition_triples to str
        question_decomposition_triples_str = ''
        for triple in question_decomposition_triples:
            try:
                # Use the get method to safely get the value, if the key does not exist, return an empty string
                relation = triple.get('relation', '')
                head = triple.get('head', '')
                tail = triple.get('tail', '')
                
                if relation and head and tail:
                    question_decomposition_triples_str += f'{{{head}, {relation}, {tail}}}, '
                else:
                    question_decomposition_triples_str += '[], '
            except Exception as e:
                # Record the exception and continue processing the next triple
                self.logger.warning(f"Error processing triple: {str(e)}")
                question_decomposition_triples_str += '[], '
       
        # Remove the last comma and space
        if question_decomposition_triples_str.endswith(', '):
            question_decomposition_triples_str = question_decomposition_triples_str[:-2]
        self.log_step("parallel_question_decomposition_triples_str", question_decomposition_triples_str)

        # First convert the reasoning chain to the required format
        formatted_triples = "{"
        
        # Process the possible mixed nested structure
        for i, outer_item in enumerate(reasoning_triples):
            if i > 0:
                formatted_triples += ", "
            
            formatted_triples += "{"
            
            # Check if the current item is a list
            if isinstance(outer_item, list):
                for j, middle_item in enumerate(outer_item):
                    if j > 0:
                        formatted_triples += ", "
                    
                    # Check if the middle item is also a list (three-layer nested)
                    if isinstance(middle_item, list):
                        formatted_triples += "{"
                        for k, triple in enumerate(middle_item):
                            if k > 0:
                                formatted_triples += ", "
                            
                            # Extract the head, relation, and tail of the triple
                            head = triple.get("head_entity_freebase_name", "")
                            relation = triple.get("relation", "")
                            tail = triple.get("tail_entity_freebase_name", "")
                            
                            # Format the triple as {head, relation, tail}
                            formatted_triples += f"{{{head}, {relation}, {tail}}}"
                        
                        formatted_triples += "}"
                    else:
                        # The middle item is a dictionary (two-layer nested)
                        triple = middle_item
                        head = triple.get("head_entity_freebase_name", "")
                        relation = triple.get("relation", "")
                        tail = triple.get("tail_entity_freebase_name", "")
                        
                        formatted_triples += f"{{{head}, {relation}, {tail}}}"
            else:
                # The outer item is a dictionary (single layer)
                triple = outer_item
                head = triple.get("head_entity_freebase_name", "")
                relation = triple.get("relation", "")
                tail = triple.get("tail_entity_freebase_name", "")
                
                formatted_triples += f"{{{head}, {relation}, {tail}}}"
            
            formatted_triples += "}"
        
        formatted_triples += "}"
        
        self.log_step("formatted_triples", formatted_triples)

        prompt = get_parallel_question_answer_prompt(formatted_triples, question, topic_entity_str, question_decomposition_triples_str)
        self.log_step("parallel_question_answer_prompt", prompt)
        messages = [
            {"role": "system", "content": "You are a system designed to answer questions based on the provided triples and your own knowledge. Please think step by step and follow the reasoning steps carefully."},
            *message_parallel_question_answer,
            {"role": "user", "content": prompt}
        ]
        
        if self.LLM_type == 'llama':
            response = run_llm_llama(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens = 2048,
                tokenizer=self.tokenizer,
                model=self.model
            )
        elif self.LLM_type == 'gpt':
            response = run_llm_gpt(
                messages=messages,
                temperature=self.args.temperature_reasoning,
                max_tokens = 2048,
                engine=self.engine
            )
        self.log_step("parallel_question_answer_response", response)
        return response, formatted_triples

    def answer_question(self, question: str, topic_entity: Dict, question_type: str) -> str:
        """Answer questions"""
        # self.args.question = question
        self.logger.info(f"\nStart processing question: {question}")

        if topic_entity:
            sorted_topic_entity = []
            for entity_id, entity_name in topic_entity.items():
                sorted_topic_entity.append({
                    'topic_entity_id': entity_id,
                    'topic_entity_name': entity_name
                })
            self.log_step("Topic entities in the question", json.dumps(sorted_topic_entity, ensure_ascii=False, indent=2))
        else:
            sorted_topic_entity = []
        self.log_step("question_type_from", self.question_type_from)
        if self.question_type_from == 'dataset':

            if question_type:
                if question_type == 'composition':
                    question_type = "Chain Structure"
                elif question_type == 'conjunction' or question_type == 'comparative' or question_type == 'superlative':
                    question_type = "Parallel Structure"
                else:
                    question_type = "Chain Structure"
                self.log_step("Question type", question_type)
            else:

                question_type = self.get_question_type_llm(question)
                self.log_step("Question type", question_type)
        elif self.question_type_from == 'LLM':

            question_type = self.get_question_type_llm(question)
            self.log_step("Question type from LLMs", question_type)

        question_decomposition_triples = self.question_decomposition_llm(question, question_type)
        self.log_step("Question decomposition", question_decomposition_triples)

        # Initialize variables before using them
        chain_question_answer = None
        parallel_question_answer = None

        # Select different reasoning methods based on the question type
        if question_type == "Chain Structure":
            reasoning_triples = self.chain_reasoning(question_decomposition_triples, sorted_topic_entity)
            self.log_step("Reasoning triples", reasoning_triples)
            
            reasoning_chain_str = ""
            for i, chain in enumerate(reasoning_triples, 1):
                reasoning_chain_str += f"chain {i}: "
                
                chain_parts = []
                for triple in chain:
                    # Extract the head, relation, and tail of the triple
                    head = triple.get("head_entity_freebase_name", "")
                    relation = triple.get("relation", "")
                    tail = triple.get("tail_entity_freebase_name", "")
                    
                    # Format the triple as {head, relation, tail}
                    triple_str = f"{{{head}, {relation}, {tail}}}"
                    chain_parts.append(triple_str)
                
                # Use a comma to connect all triples in the same chain
                reasoning_chain_str += ", ".join(chain_parts) + "\n"
            self.log_step("Reasoning chain", reasoning_chain_str)
            
            chain_question_answer = self.chain_question_answer(reasoning_chain_str, question, sorted_topic_entity, question_decomposition_triples)
            self.log_step("Reasoning chain answer", chain_question_answer)
        elif question_type == "Parallel Structure":
            reasoning_triples = self.parallel_reasoning(question_decomposition_triples, sorted_topic_entity)
            self.log_step("Reasoning triples", reasoning_triples)

            parallel_question_answer, formatted_parallel_triples = self.parallel_question_answer(reasoning_triples, question, sorted_topic_entity, question_decomposition_triples)
            self.log_step("Parallel reasoning chain answer", parallel_question_answer)
            

        # Generate reasoning path based on reasoning triples
        if chain_question_answer:
            return chain_question_answer, reasoning_chain_str, question_type
        elif parallel_question_answer:
            return parallel_question_answer, formatted_parallel_triples, question_type
        else:
            return None, None, None
