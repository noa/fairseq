```bash
# Download a prepare the data
bash prepare-wmt14en2de.sh --icml17

# Move to $DATA
mv wmt17_en_de $DATA

# Binarize the dataset
fairseq-preprocess --source-lang en \
		   --target-lang de \
		   --trainpref $DATA/train \
		   --validpref $DATA/valid \
		   --testpref $DATA/test \
		   --destdir $DATA/bin/wmt17_en_de \
		   --thresholdtgt 0 \
		   --thresholdsrc 0 \
		   --workers 20

# Train the model
./train.sh

# Evaluate the model
./evaluate.sh
```