
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model TimesNet \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 8 \
  --d_ff 16 \
  --e_layers 1 \
  --enc_in 55 \
  --c_out 55 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNet \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 25 \
  --c_out 25 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3


python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model TimesNet \
  --data SMAP \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 25 \
  --c_out 25 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model TimesNet \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 38 \
  --c_out 38 \
  --top_k 5 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 10

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model TimesNet \
  --data SWaT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 8 \
  --d_ff 8 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model TimesNet \
  --data SWaT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 16 \
  --d_ff 16 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model TimesNet \
  --data SWaT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 32 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model TimesNet \
  --data SWaT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model TimesNet \
  --data SWaT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 8 \
  --d_ff 8 \
  --e_layers 2 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model TimesNet \
  --data SWaT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 16 \
  --d_ff 16 \
  --e_layers 2 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model TimesNet \
  --data SWaT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 32 \
  --e_layers 2 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model TimesNet \
  --data SWaT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3