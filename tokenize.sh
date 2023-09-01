CUDA_VISIBLE_DEVICES=0 python tokenize_dataset_rows.py \
    --model_checkpoint chatglm2-6b \
    --input_file test.jsonl \
    --prompt_key q \
    --target_key a \
    --save_name test \
    --max_seq_length 2000 \
    --skip_overlength False

# THUDM/chatglm-6b
# THUDM/chatglm2-6b
# baichuan-inc/baichuan-7B
# internlm/internlm-chat-7b-8k
# internlm/internlm-chat-7b