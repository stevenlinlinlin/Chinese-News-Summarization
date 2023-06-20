# Chinese News Summarization
use Text-to-Text Transformer (T5) to do title generation

# Package
```
pip install -r requirements.txt
pip install -e tw_rouge
```

# Data
Example:
```json
{
    "date_publish": "2020-01-01 00:00:00",
    "title": "title",
    "source_domain": "source_domain",
    "maintext": "maintext......",
}
```

# Training
choose google/mt5-small model from HuggingFace
```
python run_summarization.py \
  --do_train \
  --do_eval \
  --model_name_or_path google/mt5-small \
  --train_file <train data path> \
  --validation_file <valid data path> \
  --output_dir ./mt5-small-greedy \
  --num_train_epochs 20 \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --per_device_eval_batch_size=4 \
  --eval_accumulation_steps=4 \
  --predict_with_generate \
  --text_column maintext \
  --summary_column title \
  --adafactor \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1 \
  --overwrite_output_dir \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
```
- output_dir: model save path (default: ./mt5-small-greedy)

# Testing
```
python run_summarization.py \
  --do_predict \
  --model_name_or_path ./mt5-small-20 \
  --test_file ./data/public.jsonl \
  --output_file ./pred_output/mt5_small.jsonl \
  --output_dir ./pred_rouge \
  --predict_with_generate \
  --text_column maintext \
  --summary_column title \
  --per_device_eval_batch_size 4 \
  [--num_beams 5 \]
  [--do_sample True \]
  [--top_k 10 \]
  [--top_p 0.5 \]
```
## generation strategies:
* --num_beams : Beam Search, ex. 5, 10
* --do_sample True : for following hyperparameters, Top-k Sampling / Top-p Sampling / Temperature
* --top_k : Top-k Sampling, ex. 10, 50
* --top_p : Top-p Sampling, ex. 0.5, 0.9
* --temperature : Temperature, ex. 0.6, 1.4

# Evaluation
use ROUGE score with chinese word segmentation([ckiptagger](https://github.com/ckiplab/ckiptagger)) to evaluate
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```

Example:
```
python eval.py -r public.jsonl -s submission.jsonl
```
Output:
```
{
  "rouge-1": {
    "f": 0.21999419163162043,
    "p": 0.2446195813913345,
    "r": 0.2137398792982201
  },
  "rouge-2": {
    "f": 0.0847583291303246,
    "p": 0.09419044877345074,
    "r": 0.08287844474014894
  },
  "rouge-l": {
    "f": 0.21017939117006337,
    "p": 0.25157090570020846,
    "r": 0.19404349000921203
  }
}
```