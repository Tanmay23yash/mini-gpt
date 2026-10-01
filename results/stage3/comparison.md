# Stage 3: final comparison on the SentiMix test set (3,000 tweets)

Mean ± std over seeds (training seeds, or few-shot example draws). The 95% CI is a paired bootstrap
(2,000 resamples of the test tweets) of one representative run per model (the median seed).

| Model | Stage | Params | Trained params | Test macro F1 | 95% CI (rep. run) | Test acc |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B + LoRA | 3 | 0.75B | 10.8M | 0.714 ± 0.007 | 0.714 [0.699, 0.730] | 0.710 |
| TF-IDF + LogReg | 1 | 0.3M | 0.3M | 0.696 | 0.696 [0.680, 0.712] | 0.691 |
| mini-GPT (pre-trained, fine-tuned) | 2 | 5.3M | 5.3M | 0.679 ± 0.002 | 0.680 [0.663, 0.696] | 0.676 |
| BiGRU (word vocab) | 1 | 1.8M | 1.8M | 0.653 ± 0.002 | - | 0.651 |
| mini-GPT (random init, fine-tuned) | 2 | 5.3M | 5.3M | 0.640 ± 0.006 | - | 0.636 |
| BiLSTM (word vocab) | 1 | 1.9M | 1.9M | 0.639 ± 0.017 | - | 0.638 |
| Qwen3.5-0.8B 3-shot | 3 | 0.75B | none | 0.508 ± 0.026 | 0.506 [0.490, 0.522] | 0.565 |
| Qwen3.5-0.8B zero-shot | 3 | 0.75B | none | 0.415 | - | 0.511 |
| Majority class | 1 | - | - | 0.179 | - | 0.367 |

## Is the difference real? Paired bootstrap of the macro F1 difference

| Comparison | Difference | 95% CI | Share of resamples where the first is better |
|---|---|---|---|
| Qwen3.5-0.8B + LoRA vs TF-IDF + LogReg | +0.018 | [+0.002, +0.035] | 98.6% |
| Qwen3.5-0.8B + LoRA vs mini-GPT (pre-trained, fine-tuned) | +0.034 | [+0.017, +0.051] | 100.0% |
| TF-IDF + LogReg vs mini-GPT (pre-trained, fine-tuned) | +0.015 | [-0.001, +0.032] | 96.8% |
| Qwen3.5-0.8B + LoRA vs Qwen3.5-0.8B 3-shot | +0.208 | [+0.189, +0.229] | 100.0% |
| mini-GPT (pre-trained, fine-tuned) vs Qwen3.5-0.8B 3-shot | +0.174 | [+0.153, +0.195] | 100.0% |

## Per-class F1 (representative runs)

| Model | negative | neutral | positive |
|---|---|---|---|
| TF-IDF + LogReg | 0.718 | 0.625 | 0.744 |
| mini-GPT (pre-trained, fine-tuned) | 0.712 | 0.587 | 0.742 |
| Qwen3.5-0.8B 3-shot | 0.634 | 0.183 | 0.701 |
| Qwen3.5-0.8B + LoRA | 0.737 | 0.647 | 0.759 |

## Example test tweets where the models disagree

Picked at random (fixed seed), 4 per gold label, among tweets where at least two models disagree;
tweets with strong profanity are skipped for display (the metrics above use every tweet).

| Tweet | Gold | TF-IDF + LogReg | mini-GPT (pre-trained, fine-tuned) | Qwen3.5-0.8B 3-shot | Qwen3.5-0.8B + LoRA |
|---|---|---|---|---|---|
| BJP ko public ka vote nahi chahiye. Yeh too EVM se jeet jayenge | negative | ✓ negative | ✓ negative | ✓ negative | ✗ neutral |
| Congress ko dhakka lagna swabhavik hai... jo sarei media ko apne ungliyo pe nachata tha waha ab uski mar | negative | ✓ negative | ✓ negative | ✓ negative | ✗ neutral |
| #ModiSarkaar2 ka abhi se satya naas?? India no longer world's fastest-growing economy | negative | ✗ neutral | ✗ positive | ✓ negative | ✗ neutral |
| Or sath sath paisy b chori kiye is 1% tax mai say itny flates b bnaye well done | negative | ✗ neutral | ✓ negative | ✓ negative | ✗ neutral |
| Utrakhand mae har post pae stay order kyu lagtae hai sir??? utrahand ka youth barbadi ki | neutral | ✓ neutral | ✓ neutral | ✗ negative | ✗ negative |
| Matlab ab satta paksh par koi sawal nahi. tum vipaksh me bhi baithne k layak nahi | neutral | ✓ neutral | ✗ negative | ✗ negative | ✓ neutral |
| modi ko jeetana aur harana tumhari kaam hai yeah kaam jonnota ke nahi hai kyunki tumh jesa exi | neutral | ✗ negative | ✗ negative | ✗ negative | ✓ neutral |
| Is he wearing a seat belt? Shayad dono ne nahi pehna hai | neutral | ✓ neutral | ✗ negative | ✗ negative | ✓ neutral |
| Ek dam mast video bnai ry ladky...👌😂😋 | positive | ✗ neutral | ✗ neutral | ✓ positive | ✓ positive |
| Eid Mubarak Handmade Greeting Card - Paper Crafts - Eid Ul Fitr - Ramadan #EidMubarakHandmadeGreetingCard | positive | ✓ positive | ✓ positive | ✓ positive | ✗ neutral |
| que s í my love muah | positive | ✓ positive | ✗ neutral | ✓ positive | ✓ positive |
| We have choosen Modi Ji NOT for just sitting purpose Balki Desh ke Namak ha | positive | ✗ neutral | ✗ neutral | ✗ negative | ✓ positive |

## Text generation on the same prompts (T=0.7, top-k 50, 30 new tokens)

| Prompt | mini-GPT (5.3M, Hinglish tweets only) | Qwen3.5-0.8B (no fine-tuning) |
|---|---|---|
| `bhai aaj ka match` | bhai aaj ka match kab hoga? | bhai aaj ka match kyu hua hai? Match hai kya hai, Match hai kya hai? Match kya hai hai? Match hai |
| `yaar mood` | yaar mood hota toh mai bhi hota. | yaar mood  <think>  </think>  Salam! 🙌 Kya aap khud aapki ya koi bhi baat ke taq |
| `modi ji ne` | modi ji ne kab tak chupa diya. | modi ji nee === Navdêr === Modi ji nee |
| `movie dekhi, ekdum` | movie dekhi, ekdum sahi bola. But I'm sure we are with you. | movie dekhi, ekdum  <think>  </think>  **Dekhi ekdum** (Sanskrit: देखा उड़ें) translation: **"Look, fly |
| `I love` | I love you too much.. | I love this story! It feels like 5 years ago, when I was a child, and my dad was telling the story to me.  What |
