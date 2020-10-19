# Stack Overflow Tags

The repository contains a project for automatic tagging of Stack Overflow questions. To help StackOverflow, we can create a model that will automatically assign tags to the question, based on its content. This problem is known as multi-label classification - for each observation, we can assign multiple labels (tags). This repository contains a simple walkthrough how to deal with such data, how to build a model that will create tags for each question. Soon, some experiment benchmark for testing different model architectures and model algorithms will be added.

## Data 

You can download a dataset used in this project [here](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data).


## Methods
The solution was implemented mainly by using scikit-learn and Tensorflow libraries. For tag prediction, a feedforward network was used initially. Soon, more models will be added.

To ech the workflow, see 'notebook.ipynb'. For data sample generation, use 'prepare_data.ipynb'. If you generate all the necessary objects from the notebook (Tokenizer, CountVectorizer, and model), you can launch a REST API server written in FastAPI by using:

`python -m server.main`

The server will be exposed on port 8000.

This app can be launched in Docker. To set it up, simply run a following command from the root directory of the repo:
```
docker build -t stacktags -f docker\\development.dockerfile .
```
To launch the FastAPI server on port 8000, run:
```
docker run -p 8000:8000  stacktags python -m server.main
```


## Next steps
The solution could contain analysis of different models - this is planned and will be added as soon as possible.

### Endnote
Feel free to check my other projects on my [Github Pages Site](https://wprazuch.github.io/).
