
Here is the code of the Paper: **Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering**.

# First, environment

please check the requirement.txt file.

# Second, Freebase

Please check freebase_readme.md file.


# Third, Run the code

```
python main.py \
    --dataset cwq \
    --remove_unnecessary_rel True \
    --LLM_type gpt \
    --engine gpt-4o \
    --question_type_from LLM \
    --openai_api_key your_openai_api_key
```

# Finally, Evaluation

```
python eval.py \
--dataset cwq \
--output_file your_output_file 
```