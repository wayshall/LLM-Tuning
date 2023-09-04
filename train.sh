# WANDB_PROJECT=DCAI-ABSA-InternLM CUDA_VISIBLE_DEVICES=0,1 python internlm_lora_tuning.py \
#     --tokenized_dataset aspect_sentiment_train_orig_10k-internlm-chat-7b \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 2 \
#     --save_steps 100 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 100 \
#     --output_dir weights/aspect_sentiment_train_orig_10k-internlm-chat-7b \
#     --report_to wandb \
#     --run_name original-0809-gby


# WANDB_PROJECT=DCAI-ABSA-InternLM CUDA_VISIBLE_DEVICES=0 python internlm_lora_tuning.py \
#     --tokenized_train_dataset aspect_sentiment_train_orig_10k-internlm-chat-7b \
#     --tokenized_eval_dataset aspect_sentiment_test_orig-internlm-chat-7b \
#     --eval_size 2000 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 100 \
#     --output_dir weights/aspect_sentiment_train_orig_10k-internlm-chat-7b \
#     --report_to wandb \
#     --run_name original-eval2k-0811-gby

# transformers==4.29.2，不要使用4.30，否则会抛错
# learning_rate 通常，你可以尝试一些常见的初始学习率值，例如0.1、0.01、0.001等，并根据模型和任务的性质来选择。更复杂的模型可能需要较小的学习率，而更大的学习率可能适用于简单的模型。
# lora_rank 如果你有大量的训练数据，较低的秩可能更容易实现，因为模型可以从更多的数据中学习。然而，对于小数据集，可能需要更高的秩来保留更多的信息。
# num_train_epochs：训练的总 epoch 数。模型将在训练数据上迭代这么多次，每次迭代都被称为一个 epoch。
# 在深度学习中，多个 epoch 的训练通常是必要的，因为模型需要多次观察训练数据来学习不同的特征和模式。通常情况下，随着 epoch 数的增加，模型的性能会逐渐提高，直到达到一个性能收敛的稳定点。
# 需要注意的是，num_train_epochs 的值应该根据任务、数据集的大小和模型的复杂性来选择。通常，如果数据集较大或模型复杂，可能需要更多的 epoch 来让模型充分学习。一方面，训练太多的 epoch 可能会导致过拟合，另一方面，训练太少的 epoch 可能会导致模型未能充分学习数据。
# 因此，在选择 num_train_epochs 时，通常需要通过实验和验证集的性能来进行调整，以找到最佳的 epoch 数以获得最佳的模型性能。
CUDA_VISIBLE_DEVICES=0,1 python chatglm2_lora_tuning.py \
    --tokenized_dataset test \
    --lora_rank 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --num_train_epochs 10 \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir weights/test \
    --report_to none
    # --output_dir weights/sentiment_comp_ie_chatglm2


# CUDA_VISIBLE_DEVICES=3 python chatglm_lora_tuning.py \
#     --tokenized_dataset sentiment_comp_ie_shuffled \
#     --lora_rank 4 \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 2 \
#     --save_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir weights/temp


# CUDA_VISIBLE_DEVICES=3 python chatglm_lora_tuning.py \
#     --tokenized_dataset sentiment_comp_ie_shuffled \
#     --lora_rank 4 \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 2 \
#     --save_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir weights/temp


# CUDA_VISIBLE_DEVICES=0,1,2,3 python baichuan_lora_tuning.py \
#     --tokenized_dataset sentiment_comp_ie_shuffled_baichuan-7B \
#     --lora_rank 4 \
#     --per_device_train_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 3 \
#     --save_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir weights/sentiment_comp_ie_shuffled_baichuan-7B


# continue training with LoRA
# 使用 rulai_baichuan-7B 数据，在 weights/rulai_plus_baichuan-7B 的基础上继续训练 新的结果保存在 weights/rulai_plus_enhanced_baichuan-7B
# CUDA_VISIBLE_DEVICES=0,1,2,3 python baichuan_lora_tuning.py \
#     --tokenized_dataset rulai_enhance_baichuan-7B \
#     --previous_lora_weights weights/rulai_plus_baichuan-7B \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 5 \
#     --save_steps 200 \
#     --save_total_limit 2 \
#     --learning_rate 1e-5 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir weights/rulai_plus_enhanced_baichuan-7B
