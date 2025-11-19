export CUDA_VISIBLE_DEVICES=0

model_name=TexFilter

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 21 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --embed_size 128 \
  --hidden_size 128 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 128 \
  --patience 6 \
  --learning_rate 0.01 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 21 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --embed_size 256 \
  --hidden_size 1024 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 64 \
  --patience 6 \
  --learning_rate 0.01 \
  --des 'Exp' \
  --use_multi_gpu

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 21 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --embed_size 256 \
  --hidden_size 1024 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 64 \
  --patience 6 \
  --learning_rate 0.01 \
  --des 'Exp' \
  --use_multi_gpu

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 21 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --embed_size 256 \
  --hidden_size 1024 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 64 \
  --patience 6 \
  --learning_rate 0.01 \
  --des 'Exp' \
  --use_multi_gpu