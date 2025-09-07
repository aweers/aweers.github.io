+++
date = '2025-09-07T00:00:46+02:00'
draft = false
title = 'DPO is SFT'
showTableOfContents = true
+++
{{< katex >}}

# DPO is SFT (with relative sequence weighting)
Training stages of LLMs almost certainly include a supervised fine-tuning (SFT) stage in which the model is trained with cross-entropy loss on many data samples.
Current state-of-the-art LLMs usually also employ a human preference alignment by training the model based on preference-ranked completion sequences using direct preference optimization (DPO).

On first glance, those training stages and their respective loss functions might be very different, but they have a lot more in common than I thought before analyzing their exact behavior.
But let's start with the higher-level overview:

> **SFT with cross-entropy loss** (_the classic, the powerhouse_)
    #linebreak()
    What would deep learning be without cross-entropy loss? It's not only a multi-time-proven, well-working loss function that can be efficiently and stably implemented, but it is also deeply rooted in a probabilistic theoretical framework.
    I won't start from the concept of entropy here, even though it is quite interesting (see [this article](https://machinelearningmastery.com/cross-entropy-for-machine-learning/) for more information). \
    In LLM training, SFT is usually employed after the pretraining (which _also_ uses cross-entropy loss) to _teach_ the model chat template, instruction following, and some general concepts and behaviors. In this article we simplify the dataset to just contain single-turn prompt-answer pairs ($\mathcal{D}_\text{CE} = \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(n)}, y^{(n)})\}$), where $x$ is a prompt (e.g. question or instruction) and $y$ is the expected answer. During training, we let the model predict every token of the answer based on the prompt and all previous answer tokens: $\hat{y}_{t + 1} = \text{LLM}(x; y_{< t})$.\
    The cross-entropy loss together with the optimizer then _pushes_ the predicted probability distribution over all tokens of the vocabulary towards the one-hot encoded ground-truth.


> **Alignment with Direct Preference Optimization (DPO)** (_the new, the pure_)\
    After _teaching_ the model basic templates, instruction following, and some general behaviors, an additional alignment stage is necessary to further improve the quality of predictions. DPO (as well as other alignment methods) enables the _teaching_ of hard-to-describe behaviors, output formats, language usage, and structure. I definitely recommend [the paper](https://arxiv.org/abs/2305.18290), since it is well-written and does a good job motivating the method and explaining it in a simple way.\
    DPO requires an already pre-trained and fine-tuned model, since the first step is to sample multiple completions for a set of prompts. In this article we assume just two completions, which are ranked and labelled as _preferred_ (commonly referred to as _winning_) and _dispreferred_ (_losing_): $\mathcal{D}_\text{DPO} = \{(x^{(1)}, y_w^{(1)}, y_l^{(1)}), (x^{(2)}, y_w^{(2)}, y_l^{(2)}), \dots, (x^{(n)}, y_w^{(n)}, y_l^{(n)})\}$. During training, both _sequence probabilities_ are computed, and the objective of the loss is to increase the likelihood of the preferred completion while simultaneously decreasing the probability of the dispreferred one.

> **Sequence probabilities** is a term frequently used in this summary, so lets quickly define it: A sequence probability  is the product of the token probabilities, i.e., $\pi(y |x) = \prod_i p(y_i |x, y_{< i})$.

Both loss functions have different objectives, different settings in which they are employed, and require different datasets, so why should they be further compared, and how are they connected?
Well, on closer look, they share also quite a few things, and it becomes hard to really tell the difference. Let me slightly reformulate their descriptions:
- **Cross-entropy** lets the model predict next tokens based on some context (prompt + previous completion tokens) and tries to move the predicted probability distribution towards the ground-truth completion.
- **DPO** lets the model estimate probabilities of possible completions and tries to increase the probability of the preferred completion and reduce the probability of the dispreferred completion.

When I started to dive deeper into the details of DPO, I had an image of both methods close to those descriptions, and it was hard for me to pinpoint the exact difference between them.

So here is a summary of the deep dive with calculated gradients and a toy example to empirically show the concrete difference.

## Cross-entropy
Well, cross-entropy is probably known, but here is the formula again (for a single data sample $(x, y) \in \mathcal{D}$):
$$
 \begin{aligned}
 \mathcal{L}_\text{ce} & = - \frac{1}{N} \sum_{i = 1}^N \sum_{j = 0}^V y_{i, j} \log(\hat{y}_{i, j}) \\
              & = - \frac{1}{N} \sum_{i = 1}^N \log(\hat{y}_{i, t_i})
\end{aligned}
$$
where
- $N$ is the sequence length (number of token)
- $V$ is the vocabulary size
- $y_{i,j}$ is the target probability token at position $i$ being token $j$ (usually either $0$ or $1$)
- $\hat{y}_{i, j}$ is the predicted probability that token at position $i$ is token $j$
- and $t_i$ is the true token at position $i$

### Gradient of cross-entropy loss wrt logits
To get a better understanding of how both loss functions work, what they optimize and how they differ, let's have a look at the gradients with respect to the logits. The gradients will let us understand how a gradient descent algorithm would act based on the loss.

So let's derive the loss function with respect to the logits $z$ with $\hat{y}_k = \pi_\theta (k) = \text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{i = 1}^{|\mathcal{V} |} e^{z_i}}$:
$$
\begin{aligned}
  \frac{\partial \mathcal{L}}{\partial z_k} & = \hat{y}_k - y_k \\
    \frac{\partial \mathcal{L}}{\partial z} & = \hat{y} - \boldsymbol{e}_y
\end{aligned}
$$

The gradient is very simple, but that does not mean it is less powerful. For the ground-truth token the gradient becomes $\hat{y}_k - 1$, so it will be negative.
The update step of a gradient descent optimizer (something like $\eta_{t + 1} = \eta_t - \alpha \frac{\partial \mathcal{L}}{\partial \eta}$) would then increase the score of this logit, which further reinforces the correct choice.
Similarly, for the incorrect tokens, we get a gradient of just $\hat{y}_k$, so a gradient descent optimizer would decrease those logits.
Note that the steepness of the gradient correlates with the predicted scores, so correct tokens with an already high score receive a smaller update than completely incorrect predicted tokens.

## DPO
Direct preference optimization (DPO) requires for each data sample $x$ a preferred answer $y_w$ and a dispreferred answer $y_l$. Usually, they are generated using the model currently in training and ranked by humans or stronger models.
The loss is calculated with this formula:
$$
  \mathcal{L}_\text{DPO} = - \mathbb{E}_{x, y_w, y_l \in \mathcal{D}} \left[ \log \sigma \left(\beta \log \frac{\pi_\theta (y_w | x)}{\pi_\text{ref} (y_w | x)} - \beta \log \frac{\pi_\theta (y_l | x)}{\pi_\text{ref} (y_l | x)} \right) \right]
$$
We can define some functions to make it better understandable and simplify the next steps:
$$
\begin{align}
  \hat{r} (y | x) = \beta \log \frac{\pi_\theta (y | x)}{\pi_\text{ref} (y | x)} \\
  \Delta = \beta \log \frac{\pi_\theta (y_w | x)}{\pi_\text{ref} (y_w | x)} - \beta \log \frac{\pi_\theta (y_l | x)}{\pi_\text{ref} (y_l | x)} = \hat{r} (y_w | x) - \hat{r} (y_l | x)
\end{align}
$$
With that the loss function for a single data sample $(x, y_w, y_l) \in \mathcal{D}$ simplifies to
$$\mathcal{L}_{DPO} = - \log \sigma (\Delta)$$

### Gradient of DPO
We can now derive the DPO loss for one sample wrt the logits:

$$
  \begin{aligned}
\nabla_\theta \mathcal{L}_\text{DPO} &= - \frac{1}{\sigma (\Delta)} \sigma(\Delta) (1 - \sigma(\Delta)) \nabla_\theta \Delta \\
  &= \sigma(-\Delta) (\nabla_\theta \hat{r}(y_w | x) - \nabla_\theta \hat{r}(y_l | x)) \\
  & = \sigma(-\Delta) \beta (\nabla_\theta \log \pi_\theta (y_w | x) - \nabla_\theta \log \pi_\theta (y_l | x))
\end{aligned}
$$
We can replace $\pi_{\theta}(y | x)$ with $\hat{y}$ and derive the gradient wrt to the logits. By now we can already notice the similarity to cross-entropy loss:
$$
\begin{aligned}
\frac{\partial \mathcal{L}_\text{DPO}}{\partial z} &= \beta \sigma(-\Delta) (\hat{y}_w - e_{y_w} - (\hat{y}_l - e_{y_l}))
\end{aligned}
$$
Here we see the reason for the title of this summary: The gradient of DPO for the preferred sample is the same as for cross-entropy up to a scaling factor. For the dispreferred sample we just have an inverted gradient. 

So basically the only, but crucial difference between SFT with cross-entropy and DPO is the relative difference between the normalized sequence probabilities of the preferred and the dispreferred sample. Importantly, not token probabilities are used here, but whole normalized sequence probabilities, so the factor is the same for all token in a sequence. Normalized, since we look at $\log \frac{\pi_\theta (y | x)}{\pi_\text{ref} (y | x)}$ instead of $\log \pi_\theta (y | x)$, but the general intuition is similar.

Lets have a look at some examples for the weighting factor before we continue with a full example:
- _The preferred answer is already strongly preferred by the model_: In cases where the normalized probability of the preferred answer is already higher than that of the dispreferred answer $\hat{r} (y_w | x) \gg \hat{r} (y_l | x)$ we end up with a very small weighting: The large, positive value of $\Delta$ leads to a value close to $0$ because of the sigmoid in the weighting. So we receive only a **small gradient** for both the preferred and dispreferred sequences.
- _Both answers have a similar probability_: If the model has no clear preference we get roughly the same sequence probabilities $\hat{r} (y_w | x) \approx \hat{r} (y_l | x)$ and the scaling factor becomes roughly $\beta/2$.
- _The dispreferred answer is strongly preferred by the model_: In cases where the preference is inverted to our expectation with $\hat{r} (y_w | x) \ll \hat{r} (y_l | x)$, we get a high scaling factor close to $beta$.

Another interesting observation from the gradient is that due to the symmetry for preferred and dispreferred samples these parts cancel for shared prefixes. This means that the model's parameters are only updated based on the parts of the sequences that actually differ.

## Simple example
Assumptions:
- Vocabulary $\mathcal{V} = \{A, B, C\}$
- Dataset $\mathcal{D} = [(x, A, B), (x, C, B)]$ where $y_w = (A, B)$ was a preferred answer compared to $y_l = (C, B)$
- Model conditional probabilities under current policy $\pi_\theta$:
  - Predicting first token: $\pi_\theta (\hat{y}_1 | x) = \begin{pmatrix}0.5 \\ 0.1 \\ 0.4\end{pmatrix}$ for $\begin{pmatrix}A \\ B \\ C\end{pmatrix}$
  - Predicting second token after $\hat{y}_1=A$: $\pi_\theta (\hat{y}_2 | [x, A]) = \begin{pmatrix}0.3 \\ 0.6 \\ 0.1\end{pmatrix}$
  - Predicting second token after $\hat{y}_1=C$: $\pi_\theta (\hat{y}_2 | [x, C]) = \begin{pmatrix}0.2 \\ 0.3 \\ 0.5\end{pmatrix}$
  - Note that especially when generating responses for DPO not the token with the highest probability is chosen. Instead the next token is sampled according to the output distribution.

### Supervised fine-tuning with cross-entropy
Let's start with the easier part, cross-entropy loss on the preferred answer. Therefore we simplify the dataset to $\mathcal{D}_\text{SFT} = [(x, A, B)]$ (since that is the preferred answer).
Given the cross-entropy loss $$\mathcal{L}_\text{CE} = - \frac{1}{N} \sum_{i = 1}^N \log(\hat{y}_{i, t_i})$$ we can compute the loss for our small toy example:
$$\begin{aligned}
\mathcal{L}_\text{CE} &= - \frac{1}{2} \sum_{i = 1}^2 \log(\hat{y}_{i, t_i}) \\
\mathcal{L}_\text{CE} &= - (\log (0.5) + \log (0.6)) \approx - (-0.69 + - 0.51) = 1.2
\end{aligned}$$

So our loss is $1.2$, but way more interesting are the gradients. So let's derive the loss function with respect to the logits $z$ with $\pi_\theta (i) = \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j = 1}^{|\mathcal{V} |} e^{z_j}}$:
$$\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k$$
The full derivation is left as an exercise to the reader ;)
That let us calculate the per token gradient of the short sequence:
- for the first predicted token: $\frac{\partial \mathcal{L}}{\partial z} =\begin{pmatrix}0.5 - 1 \\ 0.1 - 0 \\ 0.4 - 0\end{pmatrix} = \begin{pmatrix} - 0.5 \\ 0.1 \\ 0.4 \end{pmatrix}$
- for the second predicted token: $\frac{\partial \mathcal{L}}{\partial z} = \begin{pmatrix}0.3 - 0 \\ 0.6 - 1 \\ 0.1 - 0 \end{pmatrix} = \begin{pmatrix} 0.3 \\ - 0.4 \\ 0.1 \end{pmatrix}$

Using a gradient descent optimizer negative gradients lead to an increase, while for positive gradients the logits of the tokens will be decreased.
Concretely for our example that leads to the expected behavior: The logits of the ground truth token at each step are increased, proportional to their current score (steeper gradient for worse predictions). Similarly, due to the softmax function, we have implicit negative samples, so all logits of not top-1 tokens are decreased, again with a steepness based on their current score.

### DPO
DPO has two additional things we need to consider:
- Hyperparameter $\beta$: We set $\beta=1$, since our predictions are not overconfident (we will see the effect of $\beta$ in a second).
- Reference policy $\pi_\text{ref}$: Usually that is the model before the DPO training starts, but in our toy example we assume it to be the uniform distribution: $\pi_\text{ref} = \begin{pmatrix} \frac{1}{3} \\ \frac{1}{3} \\ \frac{1}{3} \end{pmatrix}$

Now we can calculate the DPO loss for our example:
$$
  \begin{aligned}
\mathcal{L}_\text{DPO} &= - \log \sigma \left(\beta \log \frac{\pi_\theta (y_{w_i} | x)}{\pi_\text{ref} (y_{w_i} | x)} - \beta \log \frac{\pi_\theta (y_{l_i} | x)}{\pi_\text{ref} (y_{l_i} | x)} \right) \\
  &= - \log \sigma  (\sum_{i = 1}^2 \log \pi_\theta (y_{w_i} | x, y_{w_{< i}}) - \sum_{i = 1}^2 \log \pi_\text{ref} (y_{w_i} | x, y_{w_{< i}})   \\ & \quad - \sum_{i = 1}^2 \log \pi_\theta (y_{l_i} | x, y_{l_{< i}}) - \sum_{i = 1}^2 \log \pi_\text{ref} (y_{l_i} | x, y_{l_{< i}}) ) \\
  &= - \log \sigma  ((-0.693 - 0.511) - 2 \dot (-1.099)   - (-0.916 - 1.204) - 2 \dot (-1.099)) \\
  &= - \log \sigma  (0.994   - 0.078 ) \\
  &= - \log \sigma  (0.916) \\
  &= - \log 0.714 \\
  &= - 0.337
\end{aligned}
$$
Here we can see the effect of $\beta$. If its a small value we always stay close to values of $0.5$ from the sigmoid functions. 
Note that in practice LLMs are likely to be overconfident, so their output values tend to be either close to zero or close to one. With a small value for $\beta$ we reduce those extreme values somewhat.

Next, lets have a look at the gradient. Recall the formula of the gradient of DPO as:
$$
  \begin{aligned}
\frac{\partial \mathcal{L}_\text{DPO}}{\partial z} &= \beta \sigma(-\Delta) (\hat{y}_w - e_{y_w} - (\hat{y}_l - e_{y_l}))
\end{aligned}
$$

Continuing with the toy example:
- that brings us to the sequence reward for the preferred sequence:
$$
  \begin{aligned}
r_\theta (y_w | x) &= \beta \log \frac{\pi_\theta (y_w | x)}{\pi_\text{ref} (y_w | x)} = \beta \sum_{t = 1}^T \log \frac{\pi_\theta (y_{w, t} | x, y_{w, < t})}{\pi_\text{ref} (y_{w, t} | x, y_{w, < t})} \\
  &=  \log(0.5) + \log(0.6) - 2 \log \left(\frac{1}{3} \right) \approx 0.99
\end{aligned}
$$
- and dispreferred sequence:
$$
  \begin{aligned}
r_\theta (y_l | x) &= \beta \log \frac{\pi_\theta (y_l | x)}{\pi_\text{ref} (y_l | x)} = \beta \sum_{t = 1}^T \log \frac{\pi_\theta (y_{l, t} | x, y_{l, < t})}{\pi_\text{ref} (y_{l, t} | x, y_{l, < t})} \\
  &= \log(0.4) + \log(0.3) - 2 \log \left(\frac{1}{3} \right)  \approx 0.08
\end{aligned}
$$
- Our policy already gives a higher probability to the preferred answer compared to the dispreferred answer, so the dynamic weighting factor becomes:
$$ \sigma (r_\theta (y_l) - r_\theta (y_w)) = \sigma(0.08 - 0.99) \approx 0.287 $$
- for first predicted token (can be combined, since both sequences share the same prefix):
$$
\frac{\partial \mathcal{L}_\text{DPO}}{\partial z_1} = 0.287 \left(\begin{pmatrix}
0.5 - 1 \\ 0.1 - 0 \\ 0.4 - 0
\end{pmatrix} - \begin{pmatrix}
0.5 - 0 \\ 0.1 - 0 \\ 0.4 - 1
\end{pmatrix}\right) = \begin{pmatrix} - 0.287 \\ 0 \\ 0.287
\end{pmatrix}
$$
- for the second predicted token (given $A$, i.e. preferred answer):
$$\frac{\partial \mathcal{L}_\text{DPO}}{\partial z_2^{(w)}} = 0.287 \begin{pmatrix}
0.3 - 0 \\ 0.6 - 1 \\ 0.1 - 0
\end{pmatrix} = \begin{pmatrix}
0.086 \\ - 0.115 \\ 0.029
\end{pmatrix}$$
- for the second predicted token (given $C$, i.e. dispreferred answer):
$$\frac{\partial \mathcal{L}_\text{DPO}}{\partial z_2^{(l)}} = 0.287 \left(-\begin{pmatrix}
0.2 - 0 \\ 0.3 - 1 \\ 0.5 - 0
\end{pmatrix}\right) = \begin{pmatrix} - 0.057 \\ 0.2 \\ - 0.144
\end{pmatrix}$$

If we compare those gradients to the gradients of the cross-entropy loss above, we see some similarities, but also some differences:
- For the first token we have a different gradient distribution. Cross-entropy targets all logits, increasing the likelihood of the ground-truth logit and decreasing all others. DPO only targets the tokens involved in the preferred and dispreferred sequence, due to the same prefix and symmetry mentioned earlier. We can rewrite the gradient calculation as:
$$\frac{\partial \mathcal{L}_\text{DPO}}{\partial z_1} = \sigma (\Delta) \hat{y}_{w_1} - e_{w_1} - (\hat{y}_{l_1} - e_{l_1})$$
- Since $\hat{y}_{w_1} - \hat{y}_{l_1}$ we get $(\partial \mathcal{L}_\text{DPO} / (\partial z_1) =  \sigma (\Delta) e_(l_1) - e_(w_1)$. This shows that for shared prefixes, the gradient only depends on the difference between the one-hot vectors of the first diverging tokens.
- The gradients for the second token of cross-entropy loss and the preferred sequence in DPO are the same, up to the scaling.
- The gradient second token of the dispreferred sequence in DPO is effectively an _anti-SFT_ update. Since the sequence was dispreferred, the model is penalized for choosing token 'B'. The gradient pushes the logit for 'B' down, reducing the probability of this choice in the future when coming from the prefix $[x, C]$

## Summary
To summarize, we saw that the gradients of cross-entropy and DPO have the same core part $\hat{y} - e_y$, but DPO has a crucial **sequence-level weighting factor**, that reduces the gradient for sequences where the preference of the model is already aligned to the expected one. 

> This summary is contains my learning notes. They may contain errors or be updated as my understanding evolves. Feel free to report errors or provide feedback to blog\@aweers.de


