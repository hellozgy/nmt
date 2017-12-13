# nmt
---
#### AI-Challenger: EN-ZH Neural Machine Translation
---
## train
#### for example:
	python train.py train --ngpu=2 --eval_iter=10000 --lr=0.001 --model=Translate_lstm --batch_size=128
 	python train.py train --ngpu=2 --eval_iter=10000 --lr=0.001 --model=Translate_lstm_resconn --batch_size=128 --embeds_size=1024
 	python train.py train --ngpu=2 --eval_iter=10000 --lr=0.001 --model=Sogou --batch_size=128

---
## generate
#### for example:
	python generate.py generate --batch_size 128 --ngpu=7 --beam-size=5 \
	--restore-file '[("Translate_lstm","checkpoint_best"),("Translate_lstm_resconn","checkpoint_best"),("Sogou","checkpoint_best")]' \
	--id fe32ffew






