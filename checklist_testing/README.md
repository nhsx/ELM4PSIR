# CheckList Process

The [CheckList](https://github.com/macrotcr/checklist) framework originates from the work of Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin and Sameer Singh - we encourage users to visit the original repository for an in-depth set of tutorials and discussion which links directly to the original paper - [Beyond Accuracy: Behavioral Testing of NLP models with CheckList](https://arxiv.org/abs/2005.04118).

CheckList is an approach to undertake behavioural or unit testing of any model, although the original implementation was geared towards language models. Whilst the original work focussed on open datasets for its examples, here we investigated CheckList's use for models trained specifically on incident report data. Thus, this section of the repositorty is focussed on testing a langauge model trained on this data, but should work using any pre-trained language model, but results may suffer.

## Types of test
### Minimal Functionality Test (MFTs)

These are simple tests to target a specific behaviour of the model. An easy example being *negation* in a sentiment classification task. A model trained to predict the sentiment of a piece of text should be able to recognise and understand that negation terms can somewhat flip the sentiment

> e.g. "The movie was very enjoyable" -> "The movie was **not** enjoyable" should flip the sentiment from *positive -> negative*.

In the case of incident reports we could re-imagine this as a *severity* of incident rather than *sentiment*.


### Invariance Test (INVs)
These are tests that perturb the inputs and we expect the predicted labels to remain the same. A simple example would be adding or removing punctuation from the text, or introducing or removing redundant information whilst preserving the general meaning of the text.

> e.g. "The movie was amazing!" -> "The movie was amazing." should not result in a different sentiment prediction.

In the case of incident reports this should work similarly.

### Directional Tests (DIRs)
These are similar to INVs, however instead of expecting invariance, we expect the model to behave in a specified or directional way. Typically these tests utilise an expectation function, such as does the probability of a prediction increase by a certain amount.

> e.g. "I liked the food" -> "I really liked the food" we would expect the latter to have an increased probability of *positive* sentiment

Again, in the case of incident reports we could re-imagine this as a *severity* of incident rather than *sentiment*.

### Application to Incident Reports

The CheckList process is very task and user specific, with the more useful tests entirely dependent on the task the model was trained to do. Classification tasks appear to be the most intuitive and with that in mind we investigated incident report models trained to classify reports as *low* or *high* severity.

An initial run through of CheckListing of incident report models is provided in the `incident_severity.ipynb`, but this is by no means a finished production-ready pipeline, or suite of tests. It should be considered more as a jumping-off point for further experimentation.
