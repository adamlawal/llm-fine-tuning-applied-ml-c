# Applied ML C: Fine-tuning Chronos on solar data

CS197 MLC cohort · Mariam Yusuff, Adam Lawal, Tunde Egunjobi, Ayodeji Akande, Chima Precious Ikwen

We wanted to see whether a pretrained time-series model gets better at forecasting solar power after we fine-tune it on solar data. Grid operators plan tomorrow's supply around forecasts like this, so even small gains matter.

We used [Chronos](https://arxiv.org/abs/2403.07815) (`amazon/chronos-t5-mini`). It treats a time series like a sentence: it scales the values, puts each one into a bin, and each bin becomes a token that a normal T5 language model learns to predict.

![system design](docs/system-design.png)

## What we did

- **Data:** hourly solar output (`pv_kw`) from a [microgrid dataset on Hugging Face](https://huggingface.co/datasets/Tristanchou/Preference-Conditioned-Heterogeneous-MARL-Microgrid), from 2016 onwards. The dataset card says this column is rescaled from real OPSD data, so it isn't raw readings from one plant.
- **Split by time:** everything up to the last 60 days for training, then 30 days for validation and 30 days for test.
- **Windows:** 168 hours of history in, the next 24 hours out.
- **Training:** Hugging Face `Trainer` (the same approach as Lecture 4), 1 epoch, trying learning rates 1e-5, 1e-4 and 1e-3, and picking the best one on validation.
- **Scoring:** mean absolute error (MAE) on the test month, compared with Chronos zero-shot and with "same as yesterday".

We planned to track runs with Weights & Biases (Lectures 8 & 9), but W&B had paused new sign-ups when we tried. Instead, every run is logged to `outputs/tuning_log.csv`.

## Results

| | test MAE (kW) |
|---|---|
| Same as yesterday | |
| Chronos zero-shot | |
| Chronos fine-tuned | |

## Running it

```bash
pip install chronos-forecasting pandas matplotlib jupyter
cd notebooks && jupyter lab chronos_solar_finetune.ipynb
```

It runs on a Mac (Apple GPU), on Kaggle, or on CPU (slowly). The dataset downloads into `data/` the first time you run it.

## Repo

- `notebooks/chronos_solar_finetune.ipynb`: the main notebook
- `notebooks/team-drafts/`: our earlier attempts
- `GPT-2_fine_tune/`: the Lecture 4 exercise we did before this project

## Who did what

- **Mariam Yusuff:** built the main pipeline the final notebook is based on: data loading and checks, the time split, sliding windows, tokenization, evaluation against the naive baseline, and the grid search with the CSV log. She also tried fine-tuning with Amazon's official Chronos training script (`team-drafts/chronos-fine-tune.ipynb`).
- **Chima Precious Ikwen:** built our first end-to-end fine-tuning run with the Hugging Face `Trainer`: a custom PyTorch `Dataset` for Chronos tokens, a tokenization check, and saving the model (`team-drafts/chronos-fine-tuning-task.ipynb`).
- **Adam Lawal:** brought the drafts together into the final notebook, got it running locally, set up the repo, the system design and the slides, and proofread everything.
- **Tunde Egunjobi:** _to confirm_. Presents evaluation and tuning.
- **Ayodeji Akande:** _to confirm_. Presents the results.


## What's missing / next

There are no weather inputs yet, and weather is the biggest factor in solar output. We only tested on one month with one seed. Next we'd like to try Nigerian solar data and add weather (Chronos-2 supports extra inputs).
