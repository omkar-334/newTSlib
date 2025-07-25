model_name=CnDiff
device=0
tphi=1
cond=2
tphi_loss=True


attn_dropout=0.1
hidden_dim=768
mlp_ratio=3
n_depth=4
n_emb=2
n_heads=8
timesteps=100

des=transcond_tphi
filename=transformer_all


# # Weather
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
#     --use_tphi $tphi \
#     --tphi_loss $tphi_loss \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps \
#     --filename $filename \
#     --beta_schedule cosine \
#     --normalize True \
#     --des $des \
#     --use_cond $cond
# done

# # ETTh1
# for rate in 0.125 0.25 0.375 0.5
# do
#   python -u run.py \
#     --task_name imputation \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_mask_$rate \
#     --mask_rate $rate \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --pred_len 96 \
#     --c_out 7 \
#     --gpu $device \
#     --d_model 96 \
#     --use_tphi $tphi \
#     --tphi_loss $tphi_loss \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps \
#     --filename $filename \
#     --beta_schedule cosine \
#     --normalize True \
#     --des $des \
#     --use_cond $cond
# done

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
#     --use_tphi $tphi \
#     --tphi_loss $tphi_loss \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps  \
#     --filename $filename \
#     --beta_schedule cosine \
#     --normalize True \
#     --des $des  \
#     --use_cond $cond
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
#     --use_tphi $tphi \
#     --tphi_loss $tphi_loss \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps \
#     --filename $filename \
#     --beta_schedule cosine \
#     --normalize True \
#     --des $des  \
#     --use_cond $cond
# done


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
    --d_model 96 \
    --use_tphi $tphi \
    --tphi_loss $tphi_loss \
    --attn_dropout $attn_dropout \
    --hidden_dim $hidden_dim \
    --mlp_ratio $mlp_ratio \
    --n_depth $n_depth \
    --n_emb $n_emb \
    --n_heads $n_heads \
    --timesteps $timesteps \
    --filename $filename \
    --beta_schedule cosine \
    --normalize True \
    --des $des  \
    --use_cond $cond
done


# # ECL
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
#     --use_tphi $tphi \
#     --tphi_loss $tphi_loss \
#     --attn_dropout $attn_dropout \
#     --hidden_dim $hidden_dim \
#     --mlp_ratio $mlp_ratio \
#     --n_depth $n_depth \
#     --n_emb $n_emb \
#     --n_heads $n_heads \
#     --timesteps $timesteps \
#     --filename $filename \
#     --beta_schedule cosine \
#     --normalize True \
#     --des $des  \
#     --use_cond $cond
# done