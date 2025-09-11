
model_name=CnDiff
des=new
filename=weather_final

device=0
tphi=1
cond=2
tphi_loss=True
normalize=True

# imputation_weather_mask_0.25_CnDiff_attndropout0.2_hiddendim32_mlp-ratio2_n-depth2_n-emb2_n-heads8_epochs10_batch32_lr0.001_timesteps200_new_use_cond2_use_tphi1_normalizeTrue_tphi-lossTrue - 0.0312
rate=0.375
attn_dropout=0.2
hidden_dim=32
timesteps=200

mlp_ratio=2
n_depth=2
n_emb=2
n_heads=8

# for n_depth in 2 4; do
#   for n_emb in 2 4; do
#     for attn_dropout in 0.0 0.1 0.2; do
#       for mlp_ratio in 2 4; do     
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
  --normalize $normalize \
  --des $des  \
  --use_cond $cond
#       done
#     done
#   done
# done

