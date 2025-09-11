
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
device=1

filename=final_forecast_caiso_new_final

# Caiso
pred_len=720
seq_len=336


for seq_len in 1440 720; do
  for hidden_dim in 128 256 64; do
    for n_depth in 1 2 4; do
      for n_emb in 1 2 4; do
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path ./dataset/Caiso/ \
          --data_path Caiso.csv \
          --model_id Caiso_s${seq_len}_p${pred_len} \
          --model $model_name \
          --data Caiso \
          --features M \
          --seq_len $seq_len \
          --label_len 0 \
          --pred_len $pred_len \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in 10 \
          --dec_in 10 \
          --c_out 10 \
          --des 'Exp' \
          --itr 1 \
          --use_cond $cond \
          --use_tphi $tphi \
          --tphi_loss $tphi_loss \
          --normalize $normalize \
          --attn_dropout $attn_dropout \
          --hidden_dim $hidden_dim \
          --mlp_ratio $mlp_ratio \
          --n_depth $n_depth \
          --n_emb $n_emb \
          --n_heads $n_heads \
          --filename $filename \
          --gpu $device \
          --timesteps $timesteps \
          
      done
    done
  done
done


  --root_path ./dataset/Caiso/ \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \