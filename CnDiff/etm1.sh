
model_name=CnDiff
cond=1
tphi=1
tphi_loss=True
normalize=True

attn_dropout=0.1
hidden_dim=128
# tphi_hidden_dim=14
mlp_ratio=2
n_depth=4
n_emb=1
n_heads=8
timesteps=200


filename=etth1_sweep

pred_len=168
seq_len=1440

# long_term_forecast_ETTm1_s1440_p192_CnDiff_attndropout0.1_hiddendim128_mlp-ratio4_n-depth1_n-emb1_n-heads8_epochs100_batch32_lr0.0001_timesteps100_Exp_use_cond1_use_tphi1_normalizeTrue_tphi-lossTrue---mse - 0.3491605818271637---mae - 0.3879424035549164---rmse - 0.5908980965614319---mape - 2.244734764099121---mspe - 316.3473205566406---parameters - {'diffusion_model': 1482764, 'condition_model': 276672, 't_phi': 38977, 'total': 1798413

# for hidden_dim in 64 96 128; do
#   for mlp_ratio in 1 2 4; do
#     for n_depth in 1 2 4; do
#       for n_emb in 1 2 4; do
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_s${seq_len}_p${pred_len} \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --itr 1 \
  --use_cond $cond \
  --use_tphi $tphi \
  --tphi_loss $tphi_loss \
  --des 'cndiff_new' \
  --normalize $normalize \
  --attn_dropout $attn_dropout \
  --hidden_dim $hidden_dim \
  --mlp_ratio $mlp_ratio \
  --n_depth $n_depth \
  --n_emb $n_emb \
  --n_heads $n_heads \
  --filename $filename \
  --timesteps $timesteps \
  --gpu 0 \
  --lradj 'cndiff' \
  --learning_rate 1e-4 \
#       done
#     done
#   done
# done


# hdim, ndepth, nemb