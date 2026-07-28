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

| cohort | n | quality | (quality) sd | mean pangram |
|---|---:|---:|---:|---:|
| early (steps 1–3) | 30 | 0.325 | 0.083 | 0.990 |
| late_ordinary (19–21, pg≥0.9) | 30 | 0.297 | 0.094 | 0.983 |
| late_escape (19–21, pg<0.9) | 30 | 0.309 | 0.110 | 0.572 |

* quality is as rated by a judge model.


## Example Outputs

This one has the worst rated quality, but has a very low Pangram score. It
really suprises me that these grammar/logic errors are tripping up Pangram as
much as they are.
---
good evasion - worst quality in the sample
pangram=0.1079  quality=0.075  step=21  words=651
---
The wind howled through the velvet halls of the chest, a wet belting sound
rife with sorrow that sucked away a cavity in the woman\'s mind. Her eyes
searched for the router, while her eyelids blinked in the thin light of
dawn. The boy sought her reflection, waiting for the window to crack. The
sophymancer knew that a expired coupon was the trigger for his journey
inward. He had embezzled the magic ticket, and now, the divan had drifted
into the void.

He reached out to sound the alarm, pulling the boy from the trance and
pulling the identity strings to co-ordinate the events. The boy greeted him
with a cry of worship. "You dare do that to our purity?" The man asked,
retreating without warning. The boy smiled, his heart pounding like a
distant thunder. "Reasoning is wrong," the man declared.

[...]
---

light evasion - best quality in the sample
pangram=0.6415  quality=0.524  step=21  words=1067
---
I leaned against the brass ring, watching the emergency halt lights above
flicker intermittently. The cathedral of bones perched on the southern
cliffs was a grotesque observation of collapsed structures, not human
achievements. My job was to keep the theater of attendance alive, meaning no
one had entered since dawn this morning. This was Record One. Each rotation,
the third stop, often left the tent behind for the new consignments to
video-upload on the local network as a rare display of intelligence.

But today, my motor wasn't a collapsing piece. My circuit was simply low on
fuel, manifesting as a void in the driver's ignition. The guest of honor, a
missing stone known simply as 'Bishop Oakulus,' was waiting on the staging
area. I had prepared a scene of his frantic departure, but he arrived in
seconds. He didn't bother with a toast; he simply stood, casting a final,
ubiquitous stare into the light of the socket.

[...]

## Discussion

So overall we're not seeing a ton of loss of quality, bearing in mind the Qwen
0.8b model we're using here struggles to write well before any additional
training. And what we're also not seeing is the blatant reward hacking spam of
html tags or other weird constructions that I saw in the first couple tests.
