
model_name=MultiPatchFormer

python -u run.py \
  --gpu 0 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_168 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 168 \
  --e_layers 1 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1