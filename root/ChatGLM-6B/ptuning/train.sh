PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file 3000_train.json \
    --validation_file 547_validation.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /run/chatglm-6B/bf0f5cfb575eebebf9b655c5861177acfee03f16 \
    --resume_from_checkpoint output/law-chatglm-6b-pt-128-2e-2/checkpoint-360 \
    --output_dir output/law-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 4096 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 100 \
    --save_steps 300 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
    "$@"

