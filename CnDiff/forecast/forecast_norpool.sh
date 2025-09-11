
model_name=CnDiff
cond=1
tphi=1
tphi_loss=True
normalize=True

attn_dropout=0.1
hidden_dim=128
# tphi_hidden_dim=14
mlp_ratio=2
n_depth=1
n_emb=2
n_heads=1
timesteps=100
device=0

filename=final_forecast_norpool_new_final

# Norpool
pred_len=720

for seq_len in 1440 720; do
  for hidden_dim in 128 256 64; do
    for n_depth in 1 2 4; do
      for n_emb in 1 2 4; do
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path ./dataset/Norpool/ \
          --data_path Norpool.csv \
          --model_id Norpool_s${seq_len}_p${pred_len} \
          --model $model_name \
          --data Norpool \
          --features M \
          --seq_len $seq_len \
          --label_len 0 \
          --pred_len $pred_len \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in 18 \
          --dec_in 18 \
          --c_out 18 \
          --des 'Exp' \
          --itr 1 \
          --use_cond $cond \
          --use_tphi $tphi \
          --tphi_loss $tphi_loss \
          --des 'Exp' \
          --itr 1 \
          --normalize $normalize \
          --attn_dropout $attn_dropout \
          --hidden_dim $hidden_dim \
          --mlp_ratio $mlp_ratio \
          --n_depth $n_depth \
          --n_emb $n_emb \
          --n_heads $n_heads \
          --filename $filename \
          --gpu $device \
          --des 'non-adjust-new' \
          --timesteps $timesteps \
          --lradj 'cndiff'
      done
    done
  done
done

  --root_path ./dataset/Norpool/ \
  --data_path Norpool.csv \
  --model_id Norpool_96_720 \
  --model $model_name \
  --data Norpool \