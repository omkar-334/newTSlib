
model_name=CnDiff
des=new
filename=etth1_final

device=1
tphi=1
cond=2
tphi_loss=True
normalize=True

rate=0.5
hidden_dim=128
timesteps=200

# imputation_ETTh1_mask_0.375_CnDiff_attndropout0.0_hiddendim128_mlp-ratio2_n-depth2_n-emb2_n-heads4_epochs10_batch32_lr0.001_timesteps200_new_use_cond2_use_tphi1_normalizeTrue_tphi-lossTrue

# for n_heads in 2 4 8 16; do
#   for n_depth in 1 2 4; do
#     for n_emb in 2 4; do
#       for attn_dropout in 0.0 0.1 0.2; do
#         for mlp_ratio in 2 4; do     
n_heads=4
n_depth=2
n_emb=2
attn_dropout=0.0
mlp_ratio=2

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
  --normalize $normalize \
  --des $des  \
  --use_cond $cond
#         done
#       done
#     done
#   done
# done

