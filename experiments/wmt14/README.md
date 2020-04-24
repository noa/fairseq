```bash
# Download a prepare the data
bash prepare-wmt14en2de.sh --icml17

# Move to $DATA
mv wmt14_en_de $DATA

# Binarize the dataset
fairseq-preprocess --source-lang en \
		   --target-lang de \
		   --trainpref $DATA/train \
		   --validpref $DATA/valid \
		   --testpref $DATA/test \
		   --destdir $DATA/bin/wmt14_en_de \
		   --thresholdtgt 0 \
		   --thresholdsrc 0 \
		   --workers 20

# Train the model
./train.sh

# (Optionally) average checkpoints
# See: https://github.com/pytorch/fairseq/issues/732
python scripts/average_checkpoints.py --inputs checkpoints/transformer_wmt16_en_de_bpe32k --num-epoch-checkpoints 10 --output averaged_model.pt

# Evaluate the model
./evaluate.sh
```