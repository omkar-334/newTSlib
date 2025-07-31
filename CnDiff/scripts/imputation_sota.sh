model_name=CnDiff
device=1
tphi=1
cond=2
tphi_loss=True
normalize=True

# imputation_ETTh1_mask_0.125_CnDiff_attndropout0.1_hiddendim768_mlp-ratio3_n-depth4_n-emb2_n-heads8_timesteps100_transcond_tphi_use_cond2_use_tphi1_normalizeTrue_tphi-lossTrue
attn_dropout=0.1
hidden_dim=256
mlp_ratio=3
n_depth=2
n_emb=2
n_heads=2
timesteps=100

des=reduce
filename=try_kan4
rate=0.125

# ETTh1
# for attn_dropout in 0.0 0.1; do
for hidden_dim in 128 256 512 768; do
# for mlp_ratio in 1 2 4; do
for n_depth in 1 2; do
for n_heads in 1 2 4; do
for timesteps in 100 200; do
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
      --normalize $normalize \
      --des $des \
      --use_cond $cond \
      --train_epochs 5
done
done
done
done
# done
# done
