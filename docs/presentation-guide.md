# Presentation guide: 7 minutes, 5 of us

Each person owns one part. Read your part and the notebook cells it points to, then practise saying it in your own words. Don't read from the slide.

| Time | Who | Part | Notebook cells to know |
|---|---|---|---|
| 0:00–1:00 | Adam | Problem + what we built (show the system design) | (the README diagram) |
| 1:00–2:30 | Mariam | Data, time split, sliding windows | Steps 1–3 |
| 2:30–3:45 | Precious | How Chronos reads numbers + fine-tuning with the HF Trainer | `tokenize_windows`, `train_model` |
| 3:45–5:00 | Tunde | How we evaluate + the grid search / CSV log | `make_eval_data`, `get_scores`, grid search |
| 5:00–6:15 | Ayodeji | Results + the forecast plot | Final test, plot |
| 6:15–7:00 | Adam | Limitations, next steps, who did what | (README) |

---

## Adam: problem and pipeline (1 min)
- The grid has to plan tomorrow's supply. Solar output changes every hour, so bad forecasts mean wasted energy or shortfalls.
- Our question: does fine-tuning a pretrained forecasting model, Chronos, on solar data beat (a) using it as-is and (b) just saying "tomorrow will look like today"?
- Walk the diagram left to right: dataset → split by time → windows → tokenizer → training → fine-tuned model → tested on a month it never saw.

## Mariam: data and windows (1.5 min)
- Dataset: hourly solar output (`pv_kw`) from a microgrid dataset on Hugging Face, from 2016. There are no missing hours (we checked that the gaps are all 1 hour).
- We split **by time, never randomly**: train, then 30 days for validation, then 30 days for test. Shuffling would let the model see the future.
- `make_windows` cuts one long series into thousands of examples: 168 hours of history → the next 24 hours. It slides forward 6 hours each time.
- Likely question: *why not shuffle?* Answer: because then the model would train on days that come after the days we test it on.

## Precious: how Chronos reads numbers and how we fine-tune (1.25 min)
- Language models only understand tokens. Chronos turns numbers into tokens in two steps:
  1. **Mean scaling:** divide by the average size of the history. Every series ends up on a similar scale, and night-time zeros stay zero.
  2. **Binning:** split the range into ~4,000 bins. Each value becomes its bin number, and that's the token.
- `tokenize_windows` does this for every window, using the Chronos tokenizer.
- We train with the Hugging Face `Trainer`, the same as Lecture 4. It runs the PyTorch loop for us: forward pass → cross-entropy loss → backward pass → optimizer step.
- Likely question: *why cross-entropy for numbers?* Answer: because after binning, predicting the next value is a classification problem, picking 1 of ~4,000 bins.

## Tunde: evaluation and tuning (1.25 min)
- For each day in the val or test month, we forecast the next 24 hours from the history before it.
- Chronos samples 20 possible futures, and we take the median. That's a probabilistic forecast: a range, not one guess.
- **Baseline:** "same as yesterday". Solar repeats every day, so this is hard to beat.
- **Metric:** MAE, the average error in kW. Lower is better.
- **Tuning:** we tried learning rates 1e-5, 1e-4 and 1e-3. We meant to track runs in Weights & Biases (Lectures 8–9), but W&B sign-ups were paused, so `log_result` writes each run to `tuning_log.csv` instead.
- We pick the learning rate on **validation** and touch the test month only once.

## Ayodeji: results (1.25 min)
- Read the three test numbers: same as yesterday / zero-shot / fine-tuned. Say the % change.
- Show the plot: black is what actually happened, orange is our forecast, and the shaded band is the model's uncertainty.
- Be honest: if the gain is small, say so. Pretraining already does a lot, and one test month with one seed is a small test.

## Adam: limitations and next steps (45 sec)
- No weather inputs, which are the biggest driver of solar output. One test month, one seed. The data is rescaled, not raw plant readings.
- Next: add weather (Chronos-2 supports extra inputs), test more months, try Nigerian solar data.
- Credit everyone (see the README).
