## Topic Modelling with Pre-trained Language models and beyond
This will serve as a guide and set of suggestions for reading around the subject of topic modelling and in particular topic modelling approaches that can be used with transformed based pre-trained language models.

### BERTopic
[BERTopic](https://github.com/MaartenGr/BERTopic) is a an amazing piece of work with a very well thought out repo, [paper](https://arxiv.org/pdf/2203.05794.pdf) and [documentation](https://maartengr.github.io/BERTopic/). There is no real benefit to outlining the BERTopic process in any great depth here, as it is done so well by the original author(s). Suffice to say, the method operates by applying dimensionality reduction and clustering techniques to sentence or document embeddings produced by PLMs like `BERT`, `RoBERTa`, and `Sentence-Transformers`. This aims to cluster **similar** documents together, and a class based tf-idf algorithm is used to derive words that represent **topics**. This is a modular approach and is open to adapting each of the steps involved, and all that is really required is a set of documents, and a encoder model to produce embeddings.

During this project we did not find sufficient time to invest any code that added any real functionality to the original authors work. Thus our recommendation is simply to follow the tutorials and various articles provided at the following [BERTopic repo](https://github.com/MaartenGr/BERTopic). They are quite clear and its just a case of shaping it to the new datasets and providing your own trained embedding model.

### Contextualised Topic Modelling (CTM)
Again a great package which was not explored in any great depth in this work, but the [repo](https://github.com/MilaNLProc/contextualized-topic-models) along with accompanying tutorials will provide readers with the tools to get going.

### OCTIS
A great repository for evaluating differnet topic models and is worth a deep dive and run through their tutorials provided with their [repo](https://github.com/MIND-Lab/OCTIS).
