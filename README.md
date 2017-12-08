# nmt
---
#### AI-Challenger: EN-ZH Neural Machine Translation
---
## train
	python train.py train --ngpu=2 --eval_iter=10000 --lr=0.001 --model=Translate_lstm --batch_size=128
 	python train.py train --ngpu=2 --eval_iter=10000 --lr=0.001 --model=Translate_lstm_resconn --batch_size=128
	python train.py train --ngpu=2 --eval_iter=10000 --lr=0.001 --model=Translate_gru_layernorm --batch_size=128 --embeds_size 1024 --hidden_size=1024 --Ls=4 --Lt=8

---
## generate
	python generate.py generate --batch_size 128 --ngpu=7 --beam-size=5 --restore-file '[("Translate_lstm","checkpoint_xx"),("Translate_lstm_resconn","checkpoint_yy)]' --id 208






