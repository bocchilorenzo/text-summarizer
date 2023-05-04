# text-summarizer

Text summarizer based on the code at https://github.com/holydrinker/text-summarizer/ and the paper [Centroid-based Text Summarization through Compositionality of Word Embeddings](www.aclweb.org/anthology/W/W17/W17-1003.pdf) by Gaetano Rossiello, Pierpaolo Basile and Giovanni Semeraro.

The code was adapted to allow for the use of either a Word2Vec model or a fastText model. It also has the ability to work with compressed fastText models in order to be usable in an environment with limited resources.

## How to use
NOTE: You can skip steps 2, 3 and 4 if you already have UDpipe 1 and the models installed or if you want to use the NLTK sentence tokenizer instead of UDpipe's.

1. Clone the repository, and install the requirements:
```bash
pip install -r requirements.txt
```

2. Install UDpipe 1. You can find installation instructions on https://ufal.mff.cuni.cz/udpipe/1/install. In short, download the release from Github and install the binary (on Windows, copy the folder for either the 32bit or 64bit binary wherever you want and add its path to the PATH environment variable).

3. Download the zip with all the UDpipe models from http://hdl.handle.net/11234/1-3131

4. Create a folder named 'models' in the root directory of this repository, and extract the models from the zip in it.

5. Download a word embeddings model and place it in the root directory. We recommend using fastText. Then, import the Summarizer:
```python
from summarizer import Summarizer
```

6. Then, create a Summarizer object such as:
```python
summ = Summarizer(
    model_path="./fasttext.bin",
    model_type="fasttext",
    compressed=True,
    language="italian",
    tokenizer="udpipe"
    )
``` 

7. Finally, summarize a text:
```python
summ.summarize(text)
```

This will return a list of sentences, and a list of the corresponding scores. From this, you can order the scores and sentences, and return the top n sentences.

## Compatible languages
If you use the NLTK tokenizer, the currently supported languages correspond with those for the PunktSentenceTokenizer, which are: czech, danish, dutch, english, estonian, finnish, french, german, greek, italian, malayalam, norwegian, polish, portuguese, russian, slovene, spanish, swedish, turkish.

For the UDPipe tokenizer, the list is vastly larger and can be found at https://ufal.mff.cuni.cz/udpipe/1/models.

## Where to get the word embedding model
To get the compressed fastText models, you can check https://github.com/avidale/compress-fasttext/releases/tag/gensim-4-draft and https://zenodo.org/record/4905385. The standard fastText models can be found on https://fasttext.cc/docs/en/crawl-vectors.html.
