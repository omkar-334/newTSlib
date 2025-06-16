
model_name=CnDiff
description="run"

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Handwriting/ \
  --model_id Handwriting \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model $model_name \
  --data UEA \
  --batch_size 16 \
  --des $description \
  --learning_rate 0.001 \
  --train_epochs 100 
