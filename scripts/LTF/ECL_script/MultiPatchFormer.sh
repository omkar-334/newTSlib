
model_name=MultiPatchFormer

python -u run.py \
  --gpu 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_168 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 168 \
  --e_layers 1 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1