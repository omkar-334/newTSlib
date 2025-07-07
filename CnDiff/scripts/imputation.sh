model_name=CnDiff
device=1
cond=1
tphi=True

attn_dropout=0.1
# 0.1
hidden_dim=512
# 512
mlp_ratio=3
# 1
n_depth=4
# 2
n_emb=2
# 2
n_heads=8
# 8
timesteps=100
# 100


# for rate in 0.125 0.25 0.375 0.5
# do
#   python -u run.py \
#     --task_name imputation \
#     --is_training 1 \
#     --root_path ./dataset/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_mask_$rate \
#     --mask_rate $rate \
#     --model $model_name \
#     --data custom \
#     --gpu $device \
#     --features M \
#     --pred_len 96 \
#     --c_out 321 \
#     --d_model 96 \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps 
# done


# for rate in 0.125 0.25 0.375 0.5
# do
#   python -u run.py \
#     --task_name imputation \
#     --is_training 1 \
#     --root_path ./dataset/weather/ \
#     --data_path weather.csv \
#     --model_id weather_mask_$rate \
#     --mask_rate $rate \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --pred_len 96 \
#     --c_out 21 \
#     --gpu $device \
#     --d_model 96 \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps 
# done

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
    --d_model 96 \
    --attn_dropout $attn_dropout \
    --hidden_dim $hidden_dim \
    --mlp_ratio $mlp_ratio \
    --n_depth $n_depth \
    --n_emb $n_emb \
    --n_heads $n_heads \
    --timesteps $timesteps
done

# # ETTh2
# for rate in 0.125 0.25 0.375 0.5
# do
#   python -u run.py \
#     --task_name imputation \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh2.csv \
#     --model_id ETTh2_mask_$rate \
#     --mask_rate $rate \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --pred_len 96 \
#     --c_out 7 \
#     --gpu $device \
#     --d_model 96 \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps 
# done

# # ETTm1
# for rate in 0.125 0.25 0.375 0.5
# do
#   python -u run.py \
#     --task_name imputation \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTm1.csv \
#     --model_id ETTm1_mask_$rate \
#     --mask_rate $rate \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --pred_len 96 \
#     --c_out 7 \
#     --gpu $device \
#     --d_model 96 \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps 
# done


# ETTm2
# for rate in 0.125 0.25 0.375 0.5
# do
#   python -u run.py \
#     --task_name imputation \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTm2.csv \
#     --model_id ETTm2_mask_$rate \
#     --mask_rate $rate \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --pred_len 96 \
#     --c_out 7 \
#     --gpu $device \
#     --d_model 96 \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps 
# done