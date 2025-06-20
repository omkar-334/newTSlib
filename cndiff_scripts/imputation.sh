
model_name=CnDiff_imputation
device=1
cond=True
tphi=True
# ECL dataset

for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 10 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --gpu $device \
    --wandb False \
    --use_tphi $tphi \
    --use_cond $cond \
    --features M \
    --pred_len 96 \
    --c_out 321 \
    --d_model 96 \
done

# weather dataset


for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 21 \
    --gpu $device \
    --wandb False \
    --use_tphi True \
    --use_cond True \
    --d_model 96 \
done

# ETTh1

for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 7 \
    --gpu $device \
    --wandb False \
    --use_tphi $tphi \
    --use_cond $cond \
    --d_model 96 \
done

# ETTh2
for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 7 \
    --gpu $device \
    --wandb False \
    --use_tphi $tphi \
    --use_cond $cond \
    --d_model 96 \
done

# ETTm1
for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 7 \
    --gpu $device \
    --wandb False \
    --use_tphi $tphi \
    --use_cond $cond \
    --d_model 96 \
done


# ETTm2
for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 7 \
    --gpu $device \
    --wandb False \
    --use_tphi $tphi \
    --use_cond $cond \
    --d_model 96 \
done