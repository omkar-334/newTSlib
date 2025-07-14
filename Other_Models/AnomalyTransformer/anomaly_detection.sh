model_name=AnomalyTransformer

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model $model_name \
  --data MSL \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 10


python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model $model_name \
  --data PSM \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model $model_name \
  --data SMAP \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model $model_name \
  --data SMD \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 38 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 10

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model $model_name \
  --data SWaT \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3