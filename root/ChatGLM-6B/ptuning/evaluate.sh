PRE_SEQ_LEN=128
CHECKPOINT=law-chatglm-6b-pt-128-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --validation_file 547_validation.json \
    --test_file 500_test.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /run/chatglm-6B/bf0f5cfb575eebebf9b655c5861177acfee03f16 \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 4096 \
    --max_target_length 512 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
