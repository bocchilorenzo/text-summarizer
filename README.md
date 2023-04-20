# text-summarizer

Text summarizer based on the code at https://github.com/holydrinker/text-summarizer/ and the paper [Centroid-based Text Summarization through Compositionality of Word Embeddings](www.aclweb.org/anthology/W/W17/W17-1003.pdf) by Gaetano Rossiello, Pierpaolo Basile and Giovanni Semeraro.

The code was adapted to allow for the use of either a Word2Vec model or a fastText model. It also has the ability to work with compressed fastText models in order to be usable in an environment with limited resources.

## How to use
Clone the repository, and install the requirements:
```bash
pip install -r requirements.txt
```

Download a model and place it in the root directory. Then, import the Summarizer:
```python
from summarizer import Summarizer
```

Then, create a Summarizer object:
```python
model = Summarizer("model_path")
``` 

Finally, summarize a text:
```python
model.summarize(text)
```

This will return a list of sentences, and a list of the corresponding scores. From this, you can order the scores and sentences, and return the top n sentences.

## Where to get the model
To get the compressed fastText models, you can check https://github.com/avidale/compress-fasttext/releases/tag/gensim-4-draft and https://zenodo.org/record/4905385. The standard fastText models can be found on https://fasttext.cc/docs/en/crawl-vectors.html.
