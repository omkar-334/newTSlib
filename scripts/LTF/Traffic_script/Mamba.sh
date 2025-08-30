model_name=Mamba

pred_len=168
python -u run.py \
  --gpu 0 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$pred_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $pred_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 862 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 862 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \

done