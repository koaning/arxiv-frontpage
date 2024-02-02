# Arxiv Frontpage 

Today's frontpage can be viewed here:

https://koaning.github.io/arxiv-frontpage/

## What's this? 

This project is an attempt at making my own frontpage of Arxiv. Every day this project does [git-scraping](https://simonwillison.net/2020/Oct/9/git-scraping/) on new Arxiv articles via [this Python API](https://pypi.org/project/arxiv/). Then, another cronjob runs a script that attemps to make recommendations based on annotations that reside in this repo. This is then comitted as a new `index.html` page which is hosted by Github pages.

This project is very much a personal one and may certainly see a bunch of changes in the future. But I figured it would be nice to host it publicly so that it may inspire other folks to make their own feed as well. 

## Contents 

- There is a `config.yml` file that contains definitions of the labels. All scripts will make assumptions based on the contents of this file. 
- There is a [taskfile](https://taskfile.dev/) that contains some common commands. 
- There is a `.github` folder that contains all the cronjobs.
- There is a `frontpage` Python module that contains all the logic to prepare data for annotation, to train sentence-models and to build the new site. 
- There are two `benchmark*.ipynb` files that contain some scripts that I've used to run benchmarks. Some attemps done with LLMs via `spacy-llm` while others were done with [pretrained embeddings](https://github.com/koaning/embetter).
- This project assumes a `.env` file, which you can use if you intend to use weights and biases to store custom sentence transformers or use external embedding providers.

## Notes 

This project also explores how to pragmatically bootstrap a predictive project. There are a few things in particular worth highlighting that feels somewhat unique. 

First off, instead of active learning this project assumes active teaching. There are multiple methods available to select a subset of interest which help the user steer the algorithm. 

![](/images/active-teaching.png)

If you want to explore the options, you can run:

```
python -m frontpage annotate
```

This will give a menu that you can use to select the subset selection method. You can annotate on a sentence-level or abstract-level and select from a number of tricks to find an interesting subset. 

In terms of modelling, this project employes a sentence-model that makes a prediction per sentence. It's possibly not _the_ most performany modelling approach, but it is easy to interpret. It also helps make the model more understandable in the UI.

![](/images/sentence-model.png)

This sentence-model can be trained by first finetuning the embedding by using a [setfit](https://github.com/huggingface/setfit)-like approach. We first use all the labels to finetune a new embedding layer, after which we train a classifier head for each label.

![](/images/multiheads.png)

I ended up writing custom implementations for a lot of this because my annotations don't really fit the "multi-label classification" setting. Some sentences may have two labels, but many will only have a single one. That means that my label matrix should have lots of `nan`-values, which goes against the assumptions of many libraries out there. 

![](/images/why-custom.png)
