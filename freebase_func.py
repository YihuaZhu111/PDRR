from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *
from typing import Dict, List
import torch
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed



# Get root logger
logger = logging.getLogger(__name__)

SPARQLPATH = "http://localhost:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md

# pre-defined sparqls
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}\n LIMIT 100"""
# Get head relations of entity, find all (specified entity, ?relation, ?x) format relations
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}\n LIMIT 100"""
# Get tail relations of entity, find all (?x, ?relation, specified entity) format relations
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}\n LIMIT 100""" 
# Get head entities of entity, input should be entity id first, then relation
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}\n LIMIT 100"""
# Get tail entities of entity, input should be relation first, then entity id
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}""" 
# sparql_id: explanation: Get tail entities of entity, through entity name or synonyms


def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation == "kg.object_profile.prominent_type" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


def execurte_sparql(sparql_query):
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        #sparql.setTimeout(40)
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        logger.error(f"SPARQL query timeout or error: {str(e)}")
        # Return empty list, let caller continue processing
        return []

def get_sliding_windows(words, window_size = 2):
    return [words[i:i+window_size] for i in range(len(words) - window_size + 1)]
    # return format: [['country', 'nation'], ['nation', 'world']]



def construct_query(search_words):
    """Build SPARQL query"""
    # Combine search words into bif:contains supported format
    logger.info(f"Building SPARQL query: {search_words}")
    search_string = ' AND '.join(f'"{word.lower()}~"' for word in search_words)
    logger.info(f"Building SPARQL query: {search_string}")
    return """ 
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?entity ?label
    WHERE {
        ?entity ns:type.object.name ?label .
        FILTER(LANG(?label) = "en") .
        FILTER(bif:contains(?label, '%s')) .
    }
    LIMIT 100
    """ % search_string
# Later determine if need to add ~, for fuzzy matching


def parallel_query_freebase_entities(texts: str, max_workers, timeout) -> List[Dict]:
    """
    Query multiple Freebase entities in parallel
    
    Args:
        texts: List of entity texts to query
        max_workers: Maximum number of parallel threads
        timeout: Timeout for each query (seconds)
    
    Returns:
        List of query results, each result format same as query_freebase_entity
    """
    def query_with_timeout(text: str) -> Dict:
        #result = None  # Initialize result variable
        try:
            result = query_freebase_entity(text)
            # Check if result is valid
            if result and isinstance(result, dict):
                if result.get('entity_id'):  # If entity found successfully
                    return {
                        'entity_id': result['entity_id'],
                        'entity_original_name': text,
                        'entity_freebase_name': result.get('entity_freebase_name'),
                        'score': result.get('score', 0)
                    }
            
            # If no valid entity found, skip directly
            logger.warning(f"Query entity '{text}' returned no valid result")
            pass
            
        except TimeoutError:
            logger.warning(f"Query entity '{text}' timeout")
            pass
        except Exception as e:
            # Only log error when real exception occurs
            logger.error(f"Error querying entity '{text}': {str(e)}")
            pass
            #return None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_text = {
            executor.submit(query_with_timeout, text): text 
            for text in texts
        }
        
        # Collect results
        for future in as_completed(future_to_text):
            text = future_to_text[future]
            try:
                result = future.result(timeout=timeout)
                # Add result directly, no additional type checking
                results.append(result)
            except TimeoutError:
                logger.warning(f"Query entity '{text}' timeout")
                pass
    # filter none in results
    results = [result for result in results if result is not None]
    logger.info(f"Parallel query entity results: {results}")
    return results



def query_freebase_entity(text: str):
    # Clean input text
    logger.info(f"Starting query_freebase_entity, input text: '{text}'")
    cleaned_text = text.lower().strip()  # for example: "Lou Seal" -> "lou seal"
    logger.info(f"Cleaned input text: '{cleaned_text}'")
    words = text.lower().strip().split()  # for example: "Lou Seal" -> ["lou", "seal"]
    #stop_words = ["what?","which?","who?", "when?", "where?", "the", "of", "in", "to", "from", "by", "with", "and", "or", "but", "as", "until", "while", "while", "as", 'for', 'is', 'a', 'an', 'on', ',']
    #words = [word for word in words if word not in stop_words]
    logger.info(f"Cleaned words: {words}")
    all_candidates = []
    
    query = construct_query(words)
    candidates = execurte_sparql(query)
    if candidates:
        all_candidates = candidates.copy()
    else:
        pass
    
    if not all_candidates:
        logger.info(f"No candidate entities found for text '{text}'")
        pass
    else:
        logger.info(f"Found candidate entities: {all_candidates}")
        
        # Calculate BERT similarity and sort
        scored_candidates = []
        for candidate in all_candidates:
            similarity = compute_similarity(cleaned_text, candidate['label']['value'].lower())
            scored_candidates.append({
                'entity_id': candidate['entity']['value'].split('/')[-1],
                'entity_original_name': text,
                'entity_freebase_name': candidate['label']['value'],
                'score': similarity
            })
        
        """
        candidates:
    [
        {
            "entity": {
                "type": "uri",
                "value": "http://rdf.freebase.com/ns/m.02m4qz"
            },
            "label": {
                "type": "literal",
                "xml:lang": "en",
                "value": "mascot"
            }
        }
        ...
    ]
        """
  
        if scored_candidates:
            # Sort by similarity
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
          
            # Only return result when score is above threshold
            if scored_candidates[0]['score'] > 0.5:
                best_candidate = scored_candidates[0]
                logger.info(f"Most similar entity that meets requirements: {best_candidate}")
                logger.info(f"Final output entity id: {best_candidate['entity_id']}")
                logger.info(f"Final output entity name: {best_candidate['entity_freebase_name']}")
                return best_candidate
            else:
                logger.info(f"Similarity below threshold, not returning result {scored_candidates[0]['entity_freebase_name']}")
                pass
        else:
            pass

BERT_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
def compute_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts"""
    embeddings = BERT_MODEL.encode([text1, text2])
    similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings[0]).unsqueeze(0),
        torch.tensor(embeddings[1]).unsqueeze(0)
    ).item()
    return similarity

def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = execurte_sparql(head_entities_extract)


    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity

def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]

def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']
    

import json
import re
from sentence_transformers import SentenceTransformer


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)
