#run the model:

python main.py \
    --dataset cwq \
    --remove_unnecessary_rel True \
    --LLM_type gpt \
    --engine gpt-4o \
    --question_type_from LLM \
    --openai_api_key your_openai_api_key

#evaluation:

python eval.py \
--dataset cwq \
--output_file your_output_file 

