export CUDA_VISIBLE_DEVICES=2

model_name=TexFilter

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 16 \
  --patience 5 \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --hidden_size 128 \
  --train_epochs 15 \
  --batch_size 8 \
  --patience 5 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --hidden_size 128 \
  --train_epochs 15 \
  --batch_size 64 \
  --patience 5 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 64 \
  --patience 5 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1