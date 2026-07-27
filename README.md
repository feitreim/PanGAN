# PanGAN
This is PanGAN (Pangram + Generative Adversarial Network). A simple Prime RL
training script that uses the Pangram AI detector as a reward signal for
finetuning a model. Right now I'm just experiementing with really small models.

The best possible result I could see from this is achieving effects similar to
what an actual GAN would, which in this case would be pushing model outputs more
towards the human part of the distribution of text. LLM outputs tend to be
clustered together in the distribution of text, while human written text is much
broader.

A good evaluation of the success of this method could maybe be to look at the
distribution before and after training from a model, and see if a) the
distribution moved, and b) if became more human or just more unique relative to
other LLM outputs (more likely).

First experiment results are starting to show some promise. For about $50 in
Pangram credits evasion success rate went from 0% to ~50%.

┌───────────────────────────────┬─────┬─────────┬───────┬───────────────┐
│            cohort             │  n  │ quality │  sd   │ mean pangram  │
├───────────────────────────────┼─────┼─────────┼───────┼───────────────┤
│ early (steps 1–3)             │ 30  │ 0.325   │ 0.083 │ 0.990         │
├───────────────────────────────┼─────┼─────────┼───────┼───────────────┤
│ late_ordinary (19–21, ai≥0.9) │ 30  │ 0.297   │ 0.094 │ 0.983         │
├───────────────────────────────┼─────┼─────────┼───────┼───────────────┤
│ late_escape (19–21, ai<0.9)   │ 30  │ 0.309   │ 0.110 │ 0.572         │
└───────────────────────────────┴─────┴─────────┴───────┴───────────────┘
