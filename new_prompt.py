def get_question_type_prompt(question: str) -> str:
    return f"""
    Please analyze the following question and determine its type.
    Question Type:
    1. Chain Structure
    2. Parallel Structure
    Question: {question}
    Output the question type with "{'{question_type}'}", and provide explanation. Do NOT format into markdown or use headers.
    """

message_question_type = [
    {   "role": "user", 
        "content": """
        Please analyze the following question and determine its type.
        Question Type:
        1. Chain Structure
        2. Parallel Structure
        Question: {Who is the coach of the team owned by Steve Bisciotti?}
        Output the question type with "{'{question_type}'}", and provide explanation. Do NOT format into markdown or use headers.
        """
    },
    {   "role": "assistant", 
        "content": """    
        The type of this question is {Chain Structure}, the bridge entity is "team". We should first find the team owned by Steve Bisciotti. And then find the coach of the team.
        """
    },
    {   "role": "user", 
        "content": """
        Please analyze the following question and determine its type.
        Question Type:
        1. Chain Structure
        2. Parallel Structure
        Question: {Where did the "Country Nation World Tour" concert artist go to college? }
        Output the question type with "{'{question_type}'}", and provide explanation. Do NOT format into markdown or use headers.
        """
    },
    {   "role": "assistant", 
        "content": """
        The type of this question is {Chain Structure}, the bridge entity is "artist". We should first find the artist concert "Country Nation World Tour". And then find the college that the artist went to.
        """
    },
    {   "role": "user", 
        "content": """
        Please analyze the following question and determine its type.
        Question Type:
        1. Chain Structure
        2. Parallel Structure
        Question: {What country bordering France contains an airport that serves Nijmegen?}
        Output the question type with "{'{question_type}'}", and provide explanation. Do NOT format into markdown or use headers.
        """
    },
    {   "role": "assistant", 
        "content": """
        The type of this question is {Parallel Structure}, the core entity is "country". We should find the countries that borders France. And simultaneously find the countries that contains an airport that serves Nijmegen. Finally find the intersecrtion of these two sets of countries.
        """
    },
    {   "role": "user", 
        "content": """
        Please analyze the following question and determine its type.
        Question Type:
        1. Chain Structure
        2. Parallel Structure
        Question: {Which movie with a character called Ajila was directed by Angelina Jolie?}
        Output the question type with "{'{question_type}'}", and provide explanation. Do NOT format into markdown or use headers.
        """
    },
    {   "role": "assistant", 
        "content": """
        The type of this question is {Parallel Structure}, the core entity is "movie". We should find the movies that have a character called Ajila. And simultaneously find the movies that were directed by Angelina Jolie. Finally find the intersecrtion of these two sets of movies.
        """
    },
    {   "role": "user", 
        "content": """
        Please analyze the following question and determine its type.
        Question Type:
        1. Chain Structure
        2. Parallel Structure
        Question: {What country borders Germany with the smallest population?}
        Output the question type with "{'{question_type}'}", and provide explanation. Do NOT format into markdown or use headers.
        """
    },
    {   "role": "assistant", 
        "content": """
        The type of this question is {Parallel Structure}, the core entity is "country". We should find all the countries that borders Germany. And simultaneously find the population of these countries. Finally compare the populations to get the country with the smallest population.
        """
    },
    {   "role": "user", 
        "content": """
        Please analyze the following question and determine its type.
        Question Type:
        1. Chain Structure
        2. Parallel Structure
        Question: {Who was the president during WW2 that held his governmental position in the US before March 4, 1933?}
        Output the question type with "{'{question_type}'}", and provide explanation. Do NOT format into markdown or use headers.
        """
    },
    {   "role": "assistant", 
        "content": """
        The type of this question is {Parallel Structure}, the core entity is "person". We should find all the people who were the president during WW2. And find the date of these people when they held their governmental position in the US. Finally determine the people who held their governmental position in the US before March 4, 1933.
        """
    },
]

def get_question_decomposition_prompt(question: str, question_type: str) -> str:
    return f"""
    Please first determine the reasoning process of the question. Then decompose the question into triples following the reasoning process.
    Each triple should contain concise head entity, relation, and tail entity. The entity with "#_number" is what we need to find.
    Question: {question}
    Question Type: {question_type}
    Please output in JSON format as follows:
    [
        {{"head": "entity1", "relation": "relation", "tail": "entity2"}}
    ]
    """

message_question_decomposition = [
    {   "role": "user", 
        "content": """
        Please first determine the reasoning process of the question. Then decompose the question into triples following the reasoning process.
        Each triple should contain concise head entity, relation, and tail entity. The entity with "#_number" is what we need to find.
        Question: {Where did the "Country Nation World Tour" concert artist go to college?}
        Question Type: {Chain Structure}
        Please output in JSON format as follows:
        [
            {"head": "entity1", "relation": "relation", "tail": "entity2"}
        ]
        """
    },
    {   "role": "assistant", 
        "content": """
        Given the question type is chain structure, the sequence of the triples is important. The bridge entity is "artist". We should first find the artist concert "Country Nation World Tour". And then find the college that the artist went to.
        The output triples are:
        [
            {"head": "Country Nation World Tour", "relation": "was concert tour by", "tail": "artist#1"},
            {"head": "artist#1", "relation": "went to", "tail": "college#1"}
        ]
        """
    },
    {   "role": "user", 
        "content": """
        Please first determine the reasoning process of the question. Then decompose the question into triples following the reasoning process.
        Each triple should contain concise head entity, relation, and tail entity. The entity with "#_number" is what we need to find.
        Question: {Rift Valley Province is located in a nation that uses which form of currency?}
        Question Type: {Chain Structure}
        Please output in JSON format as follows:
        [
            {"head": "entity1", "relation": "relation", "tail": "entity2"}
        ]
        """
    },
    {   "role": "assistant", 
        "content": """
        Given the question type is chain structure, the sequence of the triples is important. The bridge entity is "nation". We should first find the nation that Rift Valley Province is located in. And then find the currency that the nation uses.
        The output triples are:
        [
            {"head": "Rift Valley Province", "relation": "is located in", "tail": "nation#1"},
            {"head": "nation#1", "relation": "uses currency", "tail": "currency#1"}
        ]
        """
    },
    {   "role": "user", 
        "content": """
        Please first determine the reasoning process of the question. Then decompose the question into triples following the reasoning process.
        Each triple should contain concise head entity, relation, and tail entity. The entity with "#_number" is what we need to find.
        Question: {What country bordering France contains an airport that serves Nijmegen?}
        Question Type: {Parallel Structure}
        Please output in JSON format as follows:
        [
            {"head": "entity1", "relation": "relation", "tail": "entity2"}
        ]
        """
    },
    {   "role": "assistant", 
        "content": """
        Given the question type is parallel structure, the sequence of the triples is parallel. The core entity is "country". We should find the countries that borders France. And simultaneously find the countries that contains an airport that serves Nijmegen. Finally find the intersecrtion of these two sets of countries.
        The output triples are:
        [
            {"head": "country#1", "relation": "borders", "tail": "France"},
            {"head": "country#1", "relation": "contains an airport that serves", "tail": "Nijmegen"}
        ]
        """
    },
    {   "role": "user", 
        "content": """
        Please first determine the reasoning process of the question. Then decompose the question into triples following the reasoning process.
        Each triple should contain concise head entity, relation, and tail entity. The entity with "#_number" is what we need to find.
        Question: {What educational institution has a football sports team named Northern Colorado Bears is in Greeley, Colorado?}
        Question Type: {Parallel Structure}
        Please output in JSON format as follows:
        [
            {"head": "entity1", "relation": "relation", "tail": "entity2"}
        ]
        """
    },
    {   "role": "assistant", 
        "content": """
        Given the question type is parallel structure, the sequence of the triples is parallel. The core entity is "educational institution". We should find the educational institutions that are in Greeley, Colorado. And simultaneously find the educational institutions that have a football sports team named Northern Colorado Bears. Finally find the intersecrtion of these two sets of educational institutions.
        The output triples are:
        [
            {"head": "educational institution#1", "relation": "is located in", "tail": "Greeley, Colorado"},
            {"head": "educational institution#1", "relation": "has a football sports team named", "tail": "Northern Colorado Bears"}
        ]
        """
    }
]

def get_filter_relations_with_triple_prompt(triple, relations_text):
    return f"""
    Please retrieve relations that relative to the triple and rate their relative on a scale from 0 to 1 (the sum of the scores of relations is 1). Do NOT format into markdown or use headers.
    triple: {triple}
    Relations: {relations_text}
    """

message_filter_relations_with_triple = [
    {   "role": "user", 
        "content": """
        Please retrieve relations that relative to the triple and rate their relative on a scale from 0 to 1 (the sum of the scores of relations is 1). Do NOT format into markdown or use headers.
        triple: {Mesih Pasha, uncle, people#1}
        Relations: {1. child
        2. country_of_citizenship
        3. date_of_birth
        4. family
        5. father
        6. languages_spoken, written_or_signed
        7. military_rank
        8. occupation
        9. place_of_death
        10. position_held
        11. religion_or_worldview
        12. sex_or_gender
        13. sibling
        14. significant_event}
        """
    },
    {   "role": "assistant", 
        "content": """
        1. {family (Score: 0.5)}: This relation is highly relevant as it can provide information about the family background of Mesih Pasha.
        2. {father (Score: 0.4)}: Uncle is father's brother, so father might provide some information as well.
        3. {position held (Score: 0.1)}: This relation is moderately relevant as it can provide information about any significant positions held by Mesih Pasha or his uncle that could be related to this triple.
        """
    },
    {   "role": "user", 
        "content": """
        Please retrieve relations that relative to the triple and rate their relative on a scale from 0 to 1 (the sum of the scores of relations is 1). Do NOT format into markdown or use headers.
        triple: {Van Andel Institute, founded in part by, American businessman#1}
        Relations: {1. affiliation
        2. country
        3. donations
        4. educated_at
        5. employer
        6. headquarters_location
        7. legal_form
        8. located_in_the_administrative_territorial_entity
        9. total_revenue}
        """
    },
    {   "role": "assistant", 
        "content": """
        1. {affiliation (Score: 0.4)}: This relation is relevant because it can provide information about the individuals or organizations associated with the Van Andel Institute, including the American businessman who co-founded the Amway Corporation.
        2. {donations (Score: 0.3)}: This relation is relevant because it can provide information about the financial contributions made to the Van Andel Institute, which may include donations from the American businessman in question.
        3. {educated_at (Score: 0.3)}: This relation is relevant because it can provide information about the educational background of the American businessman, which may have influenced his involvement in founding the Van Andel Institute.
        """
    }
]

def get_filter_triples_with_triple_prompt(filter_triple, triples_text):
    return f"""
    Please identify the triples that are relevant to the given filter_triple and rate their relevance on a scale from 0 to 1 (the sum of the scores of triples is 1). Do NOT include irrelevant triples. Do NOT format into markdown or use headers. You should choose at least 1 triple from the triples.
    Filter Triple: {filter_triple}
    Triples: {triples_text}
    """

message_filter_triples_with_triple = [
    {   "role": "user", 
        "content": """
        Please identify the triples that are relevant to the given filter_triple and rate their relevance on a scale from 0 to 1 (the sum of the scores of triples is 1). Do NOT include irrelevant triples. Do NOT format into markdown or use headers. You should choose at least 1 triple from the triples.
        Filter Triple: {Rift Valley Province, is located in, nation#1}
        Triples: {1. Rift Valley Province, is located in, Kenya
        2. Kenya, location.country.currency_used, Kenyan shilling
        3. San Antonio Spurs, home venue, AT&T Center
        4. Rift Valley Province, is located in, UnName_Entity
        5. UnName_Entity, education.education.institution, Castlemont High School
        6. Rift Valley Province, location.contains, Baringo County
        7. Rift Valley Province, location.contained_by, Kenya}
        """
    },
    {   "role": "assistant", 
        "content": """
        1. {Rift Valley Province, is located in, Kenya. (Score: 0.5)}: This triple provides significant information about Kenya's location, which relatives to the filter_triple.
        2. {Rift Valley Province, location.contained_by, Kenya. (Score: 0.4)}: This triple provides significant information about Kenya's location, which relatives to the filter_triple.
        3. {Rift Valley Province, location.contains, Baringo County. (Score: 0.1)}: This triple provides information about the place where Rift Valley Province contains, while cannot show us the location of it, so it is irrelevant.
        """
    },
    {   "role": "user", 
        "content": """
        Please identify the triples that are relevant to the given filter_triple and rate their relevance on a scale from 0 to 1 (the sum of the scores of triples is 1). Do NOT include irrelevant triples. Do NOT format into markdown or use headers. You should choose at least 1 triple from the triples.
        Filter Triple: {Country Nation World Tour, is concert tour by, artist#1}
        Triples: {1. Country Nation World Tour, is concert by, Brad Paisley
        2. Brad Paisley, went to, Belmont University
        3. San Antonio Spurs, home venue, AT&T Center
        4. Rift Valley Province, location.contains, Baringo County
        5. Country Nation World Tour, event.held_with, Randy Houser}
        """
    },
    {   "role": "assistant", 
        "content": """
        1. {Country Nation World Tour, is concert tour by, Brad Paisley. (Score: 0.9)}: This triple provides crucial information about the artist of the concert tour.
        2. {Country Nation World Tour, event.held_with, Randy Houser. (Score: 0.1)}: This triple provides partial information about the person who Country Nation World Tour held with, while it is unrelevant to the filter_triple.
        """
    },
    {   "role": "user", 
        "content": """
        Please identify the triples that are relevant to the given filter_triple and rate their relevance on a scale from 0 to 1 (the sum of the scores of triples is 1). Do NOT include irrelevant triples. Do NOT format into markdown or use headers. You should choose at least 1 triple from the triples.
        Filter Triple: {Květa Peschke, date of birth, birth time#1}
        Triples: {1. Květa Peschke, date of birth, +1975-07-09T00:00:00Z
        2. John Newcombe, date of birth, +1944-05-23T00:00:00Z
        3. John Newcombe, country of citizenship, Australia
        4. Květa Peschke, place of birth, Czech Republic}
        """
    },
    {   "role": "assistant", 
        "content": """
        1. {Květa Peschke, date of birth, +1975-07-09T00:00:00Z. (Score: 0.9)}: This triple provides key information about Květa Peschke's age, which is relevant to the filter_triple.
        2. {Květa Peschke, place of birth, Czech Republic. (Score: 0.1)}: This triple provides partial information about the place of birth of Květa Peschke, while it is unrelevant to the filter_triple.
        """
    }
]

def get_best_chain_selection_prompt(reasoning_chain_str: str, question: str) -> str:
    """
    example:
        reasoning_chain_str:
        chain 1: {Rift Valley Province, is located in, France}, {France, uses currency, Euro}
        chain 2: ...
        ...
    """    

    return f"""
    Please select the best reasoning chain to answer the question from the following chains:
    Reasoning Chains: {reasoning_chain_str}
    Question: {question}
    Do NOT format into markdown or use headers.
    """

message_best_chain_selection = [
    {   "role": "user", 
        "content": """
        Please select the best reasoning chain to answer the question from the following chains:
        Reasoning Chains: {
        chain 1: {Phillie Phanatic, sports.sports_team.team_mascot, Philadelphia Phillies}, {Philadelphia Phillies, sports.sports_team.arena_stadium, Bright House Field}
        chain 2: {Phillie Phanatic, sports.sports_team.team_mascot, Philadelphia Phillies}, {Philadelphia Phillies, sports.sports_team_location.teams, Philadelphia}
        chain 3: {Phillie Phanatic, sports.sports_team.team_mascot, Philadelphia Phillies}, {Philadelphia Phillies, sports.sports_team.teams, Phillie Phanatic}
        }
        Question: {What is mascot Phillie Phanatic’s team’s spring training stadium?}
        Do NOT format into markdown or use headers.
        """
    },
    {   "role": "assistant", 
        "content": """
        The best reasoning chain is chain 1: [{Phillie Phanatic, sports.sports_team.team_mascot, Philadelphia Phillies}, {Philadelphia Phillies, sports.sports_team.arena_stadium, Bright House Field}].
        It successfully finds the bridge entity "team", which refers to "Philadelphia Phillies", the team of Phillie Phanatic, and then finds the team's stadium, "Bright House Field".
        """
    },
    {   "role": "user", 
        "content": """
        Please select the best reasoning chain to answer the question from the following chains:
        Reasoning Chains: {
        chain 1: {Country Nation World Tour, music.concert_tour.artist, Brad Paisley}, {Brad Paisley, owns, Nashville Predators}
        chain 2: {Country Nation World Tour, music.concert_tour.artist, Brad Paisley}, {Brad Paisley, attended, Belmont University}
        chain 3: {Country Nation World Tour, is hold by , Steve Bisciotti}, {Steve Bisciotti, attended , University of Alabama at Birmingham}
        }
        Question: {Where did the "Country Nation World Tour" concert artist go to college?}
        Do NOT format into markdown or use headers.
        """
    },
    {   "role": "assistant", 
        "content": """
        The best reasoning chain is chain 2: [{Country Nation World Tour, music.concert_tour.artist, Brad Paisley}, {Brad Paisley, attended, Belmont University}].
        It successfully finds the bridge entity "artist", which refers to Brad Paisley, the artist of the Country Nation World Tour, and then finds the college he attended: Belmont University.
        """
    },
    {   "role": "user", 
        "content": """
        Please select the best reasoning chain to answer the question from the following chains:
        Reasoning Chains: {
        chain 1: {Rift Valley Province, location.contains, Baringo County}, {Baringo County, location.country.currency_used, Kenyan shilling}
        chain 2: {Rift Valley Province, location.administrative_division.country, Kenya}, {Kenya, location.contained_in, Africa}
        chain 3: {Rift Valley Province, location.administrative_division.country, Kenya}, {Kenya, location.country.currency_used, Kenyan shilling}
        }
        Question: {Rift Valley Province is located in a nation that uses which form of currency?}
        Do NOT format into markdown or use headers.
        """
    },
    {   "role": "assistant", 
        "content": """
        The best reasoning chain is chain 3: [{Rift Valley Province, location.administrative_division.country, Kenya}, {Kenya, location.country.currency_used, Kenyan shilling}].
        It successfully finds the bridge entity "nation", which refers to Kenya, the country where Rift Valley Province is located, and then finds the nation's currency: Kenyan shilling.
        """
    }
]

def get_chain_question_answer_prompt(best_reasoning_chain_str: str, question: str, topic_entity_str: str, question_decomposition_triples_str: str) -> str:
    return f"""
    Given a question and the associated information, you are asked to answer the question using the retrieved reasoning chain and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. Do NOT format into markdown or use headers. At the end, output the final answer in this format: "{'{answer}'}"
    Question: {question}
    Topic Entity: {topic_entity_str}
    Question Decomposition Triples: {question_decomposition_triples_str}
    Retrieved Reasoning Chain: {best_reasoning_chain_str}
    """

message_chain_question_answer = [
    {   "role": "user", 
        "content": """
        Given a question and the associated information, you are asked to answer the question using the retrieved reasoning chain and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. At the end, output the final answer in this format: "{'{answer}'}"
        Question: {Rift Valley Province is located in a nation that uses which form of currency?}
        Topic Entity: {Rift Valley Province}
        Question Decomposition Triples: {{Rift Valley Province, is located in, nation#1}, {nation#1, uses currency, currency#1}}
        Retrieved Reasoning Chain: {{Rift Valley Province, location.administrative_division.country, Kenya}, {Kenya, location.country.currency_used, Kenyan shilling}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition triples:
        Step 1: Identify the nation in which Rift Valley Province is located. According to the retrieved reasoning chain, Rift Valley Province is located in Kenya.
        Step 2: Determine the currency used by Kenya. The retrieved reasoning chain indicates that Kenya uses the Kenyan shilling.
        {Kenyan shilling}
        """
    },
    {   "role": "user", 
        "content": """
        Given a question and the associated information, you are asked to answer the question using the retrieved reasoning chain and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. At the end, output the final answer in this format: "{'{answer}'}"
        Question: {Where did the "Country Nation World Tour" concert artist go to college?}
        Topic Entity: {Country Nation World Tour}
        Question Decomposition Triples: {{Country Nation World Tour, is concert tour by, artist#1}, {artist#1, attended, college#1}}
        Retrieved Reasoning Chain: {{Country Nation World Tour, music.concert_tour.artist, Brad Paisley}, {Brad Paisley, attended, Belmont University}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition triples:
        Step 1: Identify the artist of the "Country Nation World Tour" concert. According to the retrieved reasoning chain, the artist is Brad Paisley.
        Step 2: Determine the college that Brad Paisley attended. The retrieved reasoning chain indicates that he attended Belmont University.
        {Belmont University}
        """
    },
    {   "role": "user", 
        "content": """
        Given a question and the associated information, you are asked to answer the question using the retrieved reasoning chain and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. At the end, only output the final answer in this format: "{'{answer}'}"
        Question: {What movies have the actress who is an alumni for West Reading El Ctr a part of?}
        Topic Entity: {West Reading El Ctr}
        Question Decomposition Triples: {{West Reading El Ctr, has alumni, actress#1}, {actress#1, acted in, movie#1}}
        Retrieved Reasoning Chain: {{West Reading El Ctr, people.person.education, unNamed_Entity},{unNamed_Entity, movie.actor, The Lorax}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the actress who is an alumni for West Reading El Ctr. According to the retrieved reasoning chain, the actress is unNamed_Entity, who is Taylor Swift based on my own knowledge.
        Step 2: Determine the movies that Taylor Swift acted in. According to the retrieved reasoning chain, the movies that Taylor Swift acted in are The Lorax.
        {The Lorax}
        """
    },
    {   "role": "user", 
        "content": """
        Given a question and the associated information, you are asked to answer the question using the retrieved reasoning chain and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. At the end, output the final answer in this format: "{'{answer}'}"
        Question: {What form of government is used in the country where the regional Omagua dialect is used?}
        Topic Entity: {Omagua dialect}
        Question Decomposition Triples: {{Omagua dialect, is used in, country#1}, {country#1, has form of government, government form#1}}
        Retrieved Reasoning Chain: {{Peru, location.country.languages_spoken, Omagua dialect}, {}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country where the regional Omagua dialect is used. According to the retrieved reasoning chain, the country is Peru.
        Step 2: Determine the form of government of Peru. Since the retrieved reasoning chain does not provide this information, I will rely on my own knowledge. Based on that, the form of government of Peru is Presidential system.
        {Presidential system}
        """
    }
]

def get_parallel_question_answer_prompt(formatted_triples: str, question: str, topic_entity_str: str, question_decomposition_triples_str: str) -> str:
    return f"""
    Given a question and the associated information, you are asked to answer the question with these Retrieved Triples and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. Do NOT format into markdown or use headers. At the end, output the final answer in this format: "{'{answer}'}".
    Question: {question}
    Topic Entity: {topic_entity_str}
    Question Decomposition Triples: {question_decomposition_triples_str}
    Retrieved Triples: {formatted_triples}
    """

message_parallel_question_answer = [
    {   "role": "user", 
        "content": """
        Given a question and the associated information, you are asked to answer the question with these Retrieved Triples and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. At the end, output the final answer in this format: "{'{answer}'}".
        Question: {What country bordering France contains an airport that serves Nijmegen?}
        Topic Entity: {France, Nijmegen}
        Question Decomposition Triples: {{country#1, borders, France}, {country#1, contains an airport that serves, Nijmegen}}
        Retrieved Triples: {{{{Belgium, borders, France}, {Germany, borders, France}, {Italy, borders, France}, {Switzerland, borders,France}},
        {{Germany, contains an airport that serves, Nijmegen}, {Netherlands, contains an airport that serves, Nijmegen}}}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country that borders France. According to the retrieved triples, the country are Belgium, Germany, Italy, and Switzerland.
        Step 2: Identify the country that contains an airport that serves Nijmegen. According to the retrieved triples, the country is Netherlands.
        Step 3: Find the intersection of the two sets, which is Germany.
        {Germany}
        """
    },
    {   "role": "user", 
        "content": """
        Given a question and the associated information, you are asked to answer the question with these Retrieved Triples and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. At the end, output the final answer in this format: "{'{answer}'}".
        Question: {In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?}
        Topic Entity: {Portuguese, 1.8}
        Question Decomposition Triples: {{country#1, location.country.spoken_languages, Portuguese}, {country#1, child_labor_percentage, 1.8}}
        Retrieved Triples: {{{{Portugal, location.country.spoken_languages, Portuguese}, {Brazil, location.country.spoken_languages, Portuguese}, {Angola, location.country.spoken_languages, Portuguese}, {Mozambique, location.country.spoken_languages, Portuguese}, {macau, location.country.spoken_languages, Portuguese}, {South Africa, location.country.spoken_languages, Portuguese}},
        {{El Salvador, child_labor_percentage, 1.8}, {Mozambique, child_labor_percentage, 1.8}}}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country that speaks Portuguese. According to the retrieved triples, the country are Portugal, Brazil, Angola, Mozambique, Macau, and South Africa.
        Step 2: Identify the country that has a child labor percentage of 1.8. According to the retrieved triples, the country is El Salvador and Mozambique.
        Step 3: Find the intersection of the two sets, which is Mozambique.
        {Mozambique}
        """
    },
    {   "role": "user", 
        "content": """
        Given a question and the associated information, you are asked to answer the question with these Retrieved Triples and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. At the end, output the final answer in this format: "{'{answer}'}".
        Question: {Which of the seven Central American countries had co2 emissions per capita once of 2009 metric ton?}
        Topic Entity: {Central American}
        Question Decomposition Triples: {{country#1, is one of, Seven Central American countries}, {country#1, co2_emissions_per_capita, 2009 metric ton}}
        Retrieved Triples: {{{{Costa Rica, location.location.containedby, Central American}, {Guatemala, location.location.containedby, Central American}, {Honduras, location.location.containedby, Central American},{El Salvador, location.location.containedby, Central American}, {Belize, location.location.containedby, Central American},{Panama, location.location.containedby, Central American},{Gran Colombia, location.location.containedby, Central American}},
        {}}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country that is one of the seven Central American countries. According to the retrieved triples, the country are Costa Rica, Guatemala, Honduras, El Salvador, Belize, Panama, and Gran Colombia.
        Step 2: Identify the country that has a co2 emissions per capita of 2009 metric ton. Since the retrieved reasoning chain does not provide this information, I will rely on my own knowledge. Based on that, the country are Guatemala, Qatar.
        Step 3: Find the intersection of the two sets, which is Guatemala.
        {Guatemala}
        """
    },
    {   "role": "user", 
        "content": """
        Given a question and the associated information, you are asked to answer the question with these Retrieved Triples and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. At the end, output the final answer in this format: "{'{answer}'}".
        Question: {Who was the 2009 Governor of Arizona and held his governmental position from before Janaury 6, 2003?}
        Topic Entity: {Arizona}
        Question Decomposition Triples: {{person#1, was governor of, Arizona}, {person#1, held governmental position from, before Janaury 6, 2003}}
        Retrieved Triples: {{{{Arizona, goverment.government_position.governor, Jan Brewer}, {Arizona, goverment.government_position.governor, Janet Napolitano}},{}}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the governor of Arizona. According to the retrieved triples, the governor is Jan Brewer and Janet Napolitano.
        Step 2: Identify the terms of office of these two governors. Since the retrieved reasoning chain does not provide this information, I will rely on my own knowledge. Based on my knowledge, Janet Napolitano was the Governor of Arizona from January 6, 2023 to January 21 2009, after which Jan Brewer succeeded her as Governor. 
        Step 3: Compare the date to find the governor who held the position from before January 6, 2003. From the information, Janet Napolitano was the governor who held the position from before January 6, 2003.
        {Janet Napolitano}
        """
    },
    {   "role": "user", 
        "content": """
        Given a question and the associated information, you are asked to answer the question with these Retrieved Triples and your own knowledge. Please think setep by step and follow the Question Decomposition Triples carefully. Do NOT output answer without reasoning steps. At the end, output the final answer in this format: "{'{answer}'}".
        Question: {What country borders Germany with the smallest population?}
        Topic Entity: {Germany}
        Question Decomposition Triples: {{Germany, borders, country#1}, {country#1, has population, population#1}}
        Retrieved Triples: {{{{Germany, borders, Austria}, {Germany, borders, Switzerland}, {Germany, borders, France}, {Germany, borders, Belgium}, {Germany, borders, Netherlands}, {Germany, borders, Luxembourg}, {Germany, borders, Poland}, {Germany, borders, Czech Republic}, {Germany, borders, Denmark}},
        {}}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country that borders Germany. According to the retrieved triples, the country are Austria, Switzerland, France, Belgium, Netherlands, Luxembourg, Poland, Czech Republic, and Denmark.
        Step 2: Identify the country with the smallest population. Since the retrieved reasoning chain does not provide this information, I will rely on my own knowledge. Based on my knowledge, Austria, Switzerland, France, Belgium, Netherlands, Luxembourg, Poland, Czech Republic, and Denmark have a population of 8.9 million, 8.6 million, 67.4 million, 11.5 million, 17.5 million, 0.64 million, 38.5 million, 10.7 million and 5.8 million respectively.
        Step 3: Find the country with the smallest population, which is Luxembourg.
        {Luxembourg}
        """
    }
]


def get_IO_prompt(question: str) -> str:
    return f"""
    Please answer the question, and output the answer in this format: "{'{answer}'}". Do NOT format into markdown or use headers
    Question: {question}
    """

message_IO = [
    {   "role": "user", 
        "content": """
        Please answer the question, and output the answer in this format: "{'{answer}'}".
        Question: {What state is home to the university that is represented in sports by George Washington Colonials men's basketball?}
        """
    },
    {   "role": "assistant", 
        "content": """
        {Washington, D.C.}
        """
    },
    {   "role": "user", 
        "content": """
        Please answer the question, and output the answer in this format: "{'{answer}'}".
        Question: {Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?}
        """
    },
    {   "role": "assistant", 
        "content": """
        {Bharoto Bhagyo Bidhata}
        """
    },
    {   "role": "user", 
        "content": """
        Please answer the question, and output the answer in this format: "{'{answer}'}".
        Question: {Who was the artist nominated for an award for You Drive Me Crazy?}
        """
    },
    {   "role": "assistant", 
        "content": """
        {Jason Allen Alexander}
        """
    },
    {   "role": "user", 
        "content": """
        Please answer the question, and output the answer in this format: "{'{answer}'}".
        Question: {What person born in Siegen influenced the work of Vincent Van Gogh?}
        """
    },
    {   "role": "assistant", 
        "content": """
        {Peter Paul Rubens}
        """
    },
    {   "role": "user", 
        "content": """
        Please answer the question, and output the answer in this format: "{'{answer}'}".
        Question: {What is the country close to Russia where Mikheil Saakashvii holds a government position?}
        """
    },
    {   "role": "assistant", 
        "content": """
        {Georgia}
        """
    },
    {   "role": "user", 
        "content": """
        Please answer the question, and output the answer in this format: "{'{answer}'}".
        Question: {What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?}
        """
    },
    {   "role": "assistant", 
        "content": """
        {Heroin}
        """
    }
]

def get_COT_prompt(question: str) -> str:
    return f"""
    Please think setep by step and answer the question. Output the answer in this format: "{'{answer}'}". Do NOT format into markdown or use headers
    Question: {question}
    """

message_COT = [
    {   "role": "user", 
        "content": """
        Please think setep by step and answer the question. Output the answer in this format: "{'{answer}'}".
        Question: {What state is home to the university that is represented in sports by George Washington Colonials men's basketball?}
        """
    },
    {   "role": "assistant", 
        "content": """
        First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.
        """
    },
    {   "role": "user", 
        "content": """
        Please think setep by step and answer the question. Output the answer in this format: "{'{answer}'}".
        Question: {Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?}
        """
    },
    {   "role": "assistant", 
        "content": """
        First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.
        """
    },
    {   "role": "user", 
        "content": """
        Please think setep by step and answer the question. Output the answer in this format: "{'{answer}'}".
        Question: {Who was the artist nominated for an award for You Drive Me Crazy?}
        """
    },
    {   "role": "assistant", 
        "content": """
        First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.
        """
    },
    {   "role": "user", 
        "content": """
        Please think setep by step and answer the question. Output the answer in this format: "{'{answer}'}".
        Question: {What person born in Siegen influenced the work of Vincent Van Gogh?}
        """
    },
    {   "role": "assistant", 
        "content": """
        First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.
        """
    },
    {   "role": "user", 
        "content": """
        Please think setep by step and answer the question. Output the answer in this format: "{'{answer}'}".
        Question: {What is the country close to Russia where Mikheil Saakashvii holds a government position?}
        """
    },
    {   "role": "assistant", 
        "content": """
        First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.
        """
    },
    {   "role": "user", 
        "content": """
        Please think setep by step and answer the question. Output the answer in this format: "{'{answer}'}".
        Question: {What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?}
        """
    },
    {   "role": "assistant", 
        "content": """
        First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}.
        """
    }
]

def get_chain_question_answer_without_retrieval_prompt(question: str, question_decomposition_triples_str: str) -> str:
    return f"""
    Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
    Question: {question}
    Question Decomposition Triples: {question_decomposition_triples_str}
    """

message_chain_question_answer_without_retrieval = [
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {Rift Valley Province is located in a nation that uses which form of currency?}
        Question Decomposition Triples: {{Rift Valley Province, is located in, nation#1}, {nation#1, uses currency, currency#1}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition triples:
        Step 1: Identify the nation in which Rift Valley Province is located. Based on my knowledge, Rift Valley Province is located in Kenya.
        Step 2: Determine the currency used by Kenya. Based on my knowledge, Kenya uses the Kenyan shilling.
        {Kenyan shilling}
        """
    },
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {Where did the "Country Nation World Tour" concert artist go to college?}
        Question Decomposition Triples: {{Country Nation World Tour, is concert tour by, artist#1}, {artist#1, attended, college#1}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition triples:
        Step 1: Identify the artist of the "Country Nation World Tour" concert. Based on my knowledge, the artist is Brad Paisley.
        Step 2: Determine the college that Brad Paisley attended. Based on my knowledge, he attended Belmont University.
        {Belmont University}
        """
    },
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {What movies have the actress who is an alumni for West Reading El Ctr a part of?}
        Question Decomposition Triples: {{West Reading El Ctr, has alumni, actress#1}, {actress#1, acted in, movie#1}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the actress who is an alumni for West Reading El Ctr. Based on my own knowledge, the actress is Taylor Swift.
        Step 2: Determine the movies that Taylor Swift acted in. Based on my own knowledge, the movies that Taylor Swift acted in are The Lorax.
        {The Lorax}
        """
    },
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {What is the type of government in the country where the newspaper "Haaretz" circulates?}
        Question Decomposition Triples: {{Haaretz, circulates in, country#1}, {country#1, has type of government, government type#1}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country where the newspaper "Haaretz" circulates. Based on my own knowledge, the country is Israel.
        Step 2: Determine the type of government of Israel. Based on my own knowledge, the type of government of Israel is Parliamentary system.
        {Parliamentary system}
        """
    },
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {What form of government is used in the country where the regional Omagua dialect is used?}
        Question Decomposition Triples: {{Omagua dialect, is used in, country#1}, {country#1, has form of government, government form#1}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country where the regional Omagua dialect is used. Based on my own knowledge, the country is Peru.
        Step 2: Determine the form of government of Peru. Based on my own knowledge, the form of government of Peru is Presidential system.
        {Presidential system}
        """
    }
]

def get_parallel_question_answer_without_retrieval_prompt(question: str, question_decomposition_triples_str: str) -> str:
    return f"""
    Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
    Question: {question}
    Question Decomposition Triples: {question_decomposition_triples_str}
    """

message_parallel_question_answer_without_retrieval = [
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {What country bordering France contains an airport that serves Nijmegen?}
        Question Decomposition Triples: {{country#1, borders, France}, {country#1, contains an airport that serves, Nijmegen}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country that borders France. Based on my own knowledge, the country are Belgium, Germany, Italy, and Switzerland.
        Step 2: Identify the country that contains an airport that serves Nijmegen. Based on my knowledge, the country which contains an airport that serves Nijmegen are Germany and Netherlands.
        Step 3: Find the intersection of the two sets, which is Germany.
        {Germany}
        """
    },
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?}
        Question Decomposition Triples: {{country#1, spoken_languages, Portuguese}, {country#1, child_labor_percentage, 1.8}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country that speaks Portuguese. Based on my own knowledge, the country are Portugal, Brazil, Angola, Mozambique, Macau, and South Africa.
        Step 2: Identify the country that has a child labor percentage of 1.8. Based on my own knowledge, the country is El Salvador and Mozambique.
        Step 3: Find the intersection of the two sets, which is Mozambique.
        {Mozambique}
        """
    },
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {Which of the seven Central American countries had co2 emissions per capita once of 2009 metric ton?}
        Question Decomposition Triples: {{country#1, is one of, Seven Central American countries}, {country#1, co2_emissions_per_capita, 2009 metric ton}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country that is one of the seven Central American countries. Based on my own knowledge, the country are Costa Rica, Guatemala, Honduras, El Salvador, Belize, Panama, and Gran Colombia.
        Step 2: Identify the country that has a co2 emissions per capita of 2009 metric ton. Based on my own knowledge, the country are Guatemala, Qatar.
        Step 3: Find the intersection of the two sets, which is Guatemala.
        {Guatemala}
        """
    },
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {Who was the 2009 Governor of Arizona and held his governmental position from before Janaury 6, 2003?}
        Question Decomposition Triples: {{person#1, was governor of, Arizona}, {person#1, held governmental position from, before Janaury 6, 2003}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the governor of Arizona. Based on my own knowledge, the governor is Jan Brewer and Janet Napolitano.
        Step 2: Identify the terms of office of these two governors. Based on my own knowledge, Janet Napolitano was the Governor of Arizona from January 6, 2023 to January 21 2009, after which Jan Brewer succeeded her as Governor. 
        Step 3: Compare the date to find the governor who held the position from before January 6, 2003. From the information, Janet Napolitano was the governor who held the position from before January 6, 2003.
        {Janet Napolitano}
        """
    },
    {   "role": "user", 
        "content": """
        Answer the question using the provided decomposition triples and your own knowledge. Think step by step, and strictly follow the triples. Do not skip reasoning, use markdown or headers. At the end, output the final answer as: "{'{answer}'}".
        Question: {What country borders Germany with the smallest population?}
        Question Decomposition Triples: {{Germany, borders, country#1}, {country#1, has population, population#1}}
        """
    },
    {   "role": "assistant", 
        "content": """
        Following the question decomposition Triples:
        Step 1: Identify the country that borders Germany. Based on my own knowledge, the country are Austria, Switzerland, France, Belgium, Netherlands, Luxembourg, Poland, Czech Republic, and Denmark.
        Step 2: Identify the country with the smallest population. Based on my knowledge, Austria, Switzerland, France, Belgium, Netherlands, Luxembourg, Poland, Czech Republic, and Denmark have a population of 8.9 million, 8.6 million, 67.4 million, 11.5 million, 17.5 million, 0.64 million, 38.5 million, 10.7 million and 5.8 million respectively.
        Step 3: Find the country with the smallest population, which is Luxembourg.
        {Luxembourg}
        """
    }
]