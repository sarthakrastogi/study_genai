# Notes on Generative AI

## From [Chip Huyen’s book “*AI Engineering”*](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)

## Notes taken by [Sarthak Rastogi](https://www.linkedin.com/in/sarthakrastogi/)
Any mistakes are mine. Please [raise an issue](https://github.com/sarthakrastogi/study_genai/issues/new) for correction.

# 1. Understanding LLMs

## Language Models

LMs are statistical models of text. They are autoregressive models which means that they predict the next token based on their own prediction of the previous token.

- Autoregressive models, like GPT, generate text sequentially by predicting the next word based on previous tokens (e.g., "The cat sat on the \_\_\_"). They use causal masking during training so that each token only sees previous ones. Autoregressive training uses left-to-right context and teacher forcing.

- Masked language models, like BERT, predict missing words in a sentence by randomly masking some (e.g., "The cat \[MASK\] on the mat"). BERT is trained using bidirectional context, seeing words before and after the mask – this is more ideal for capturing the full context of a sentence. Masked models use bidirectional context with a masked loss function.

## How LMs Scale

Text documents don’t need manual labelling because a text sequence itself can be viewed as labelled data – eg, a sentence can be split at any point to give an input and output training example. So, LMs are trained using self-supervision instead of supervision which would require manual labelling, and therefore we are able to train LMs on such huge datasets.

An LM with over 100B params is called an LLM. And a foundation model is one which has been trained on broad data and can be fine-tuned for a specific downstream task.

Because of the popularity of foundation models, while ML Engineering primarily involves training model development, within AI Engineering we’re more concerned with adapting and evaluating models.

## LLM Size

1. The dataset an LLM is trained on is measured in tokens. If it’s trained for N epochs on T tokens, the no. of training epochs is N x T.

2. The number of params in an LLM determines its size, and together with the bit precision used for it, it determines the compute required for inference.

Assuming a 20% overhead, the memory required for inference in GB \=  
(no. of params in billions x bit precision / 8\) x 1.2

Training typically needs 3-4x the memory needed for inference.

This model size calculation does not apply to sparse models, which have most of their params as zero. Mixture of Experts is an example – the model is grouped into different groups of params, each called an expert, with some params being shared among experts. Only a subset of the experts is used to process each token, and the compute required for inference depends on how many experts are being used.

### Scaling law

Together, the model size and dataset size determine the model performance.

- [DeepMind has found that](https://arxiv.org/abs/2203.15556) for compute-optimal training:

no. of training tokens required \= 20 x model size

- [Meta has found that](https://ai.meta.com/research/publications/beyond-neural-scaling-laws-beating-power-law-scaling-via-data-pruning/) it gets progressively more expensive to reduce a model’s error rate further and further.

**Hyperparameter transferring:** Since it’s not possible to try out multiple combinations of hyperparameters while training large models, the current approach is to tune hyperparameters on smaller models and then extrapolate them to larger ones.

## Post-training

While pre-training a model is self-supervised, post-training is usually supervised. We can adapt models by updating weights (fine-tuning and inference optimisation) or without updating weights (prompt engineering).

Post-training takes only a small fraction (eg, 2%) of the compute resources.

A pre-trained LLM needs to be post-trained because pre-trained LLMs need to be

1. trained for conversation instead of completion and  
2. moderated.

Steps to post-training:

1. Start with a raw pre-trained LLM trained on internet data including unsafe content.  
2. **Supervised/Instruction Fine-Tuning (SFT):**  
* Fine-tune the model on high quality data consisting of (prompt, response) pairs for a variety of tasks.  
* This optimises the model for conversations and makes it somewhat acceptable.  
3. Alignment: Fine-tune the model so that the responses are better aligned with human preference and the model is customer-appropriate. This is done using RLHF, DPO, RLAIF, or other techniques. RLHF is the most popular and works in 2 steps:  
   1. Training a reward model that scores the LLM’s response. The model is trained on response ranking data from human labellers. During training it learns to maximise the difference between the scores of the winning and losing responses.  
   2. Optimise the LLM to give responses for which the reward model will give the max score.

## LLM Inference

### Sampling

When we write an input prompt, in the LLM’s output we are more likely to get a response from those documents whose prefixes are more similar to our input prompt. Hence, the output we get is from a subset of the large amount of data present in the LLM.

#### Temperature

Being probabilistic models, it’s possible to sample different responses each time you run the same input through an LLM. Each possible response has a probability p associated with it, and will occur that p% of the time if not using temperature (i.e., temperature t \= 1).  
t \= 0.7 is recommended for creative tasks.  
However, if t is lowered, the probability of the most likely response increases, and at t approaching 0, only the most likely (and obvious) response will be returned every time and hence the LLM will respond “less creatively”.

#### Top-k sampling

Instead of softmaxing over the probabilities of all tokens in the vocab, which is computationally expensive, it’s better to only do it over the top k tokens where k is between 50 \- 500\.  
A smaller k value means the set of possible words is smaller, which makes the output more predictable and less interesting.

#### Top-p / nucleus sampling

Instead of using the same k value for all inputs, top-p sampling effectively uses only those top values such that their cumulative probability becomes p, where p ranges from 0.9 \- 0.95.

### Test-time sampling

Out of the multiple possible responses, usually the one with the highest logprobs is picked as the final response. But a reward model can also be used to score the responses and pick the best one – this boosts the model performance significantly, but the costs increases approximately by a multiple of the number of responses sampled.  
For objective responses, test-time sampling can also be used to pick the most frequently given answer as the correct one.

## Hallucinations

### Why Hallucinations happen

**Hypothesis 1:** LLMs hallucinate because they can’t differentiate between external data (including training data and user prompt context) vs the data already generated by the model so far.  
**Mitigation strategies:**

1. Teaching the model to differentiate between the two using RL.  
2. Including both correct and incorrect information in training data and using SL to teach the difference.

**Hypothesis 2:** Hallucinations are caused by the mismatch between the knowledge in the model’s training data vs the SFT labeller’s knowledge used when writing a response.  
**Mitigation strategies:**

1. Ask the model to provide sources used for the response.  
2. The RLHF reward model should punish the LLM for hallucinating.

