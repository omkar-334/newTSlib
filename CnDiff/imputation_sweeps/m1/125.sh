
model_name=CnDiff
des=new
filename=ettm1_sweep

device=0
tphi=1
cond=2
tphi_loss=True
normalize=True



# imputation_ETTm1_mask_0.5_CnDiff_attndropout0.1_hiddendim128_mlp-ratio2_n-depth4_n-emb2_n-heads8_epochs10_batch32_lr0.001_timesteps200_new_use_cond2_use_tphi1_normalizeTrue_tphi-lossTrue

rate=0.125
hidden_dim=128
timesteps=200
n_heads=8

# Ettm1
  for n_depth in 2 4; do
    for n_emb in 2 4; do
      for attn_dropout in 0.0 0.1 0.2; do
        for mlp_ratio in 2 4; do     
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
      done
    done
  done
done