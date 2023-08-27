# Arxiv Frontpage 

This project is an attempt at making my own frontpage of Arxiv. Every day this project does [git-scraping](https://simonwillison.net/2020/Oct/9/git-scraping/) on new Arxiv articles via [this Python API](https://pypi.org/project/arxiv/). Then, another cronjob runs a script that attemps to make recommendations based on annotations that reside in this repo. This is then comitted as a new `index.html` page which is hosted by Github pages.

This project is very much a personal one and may certainly see a bunch of changes in the future. But I figured it would be nice to host it publicly so that it may inspire other folks to make their own feed as well. 

The project assumes that you're using [Prodigy](https://prodi.gy) to annotate your data. You're still free to copy the code and change it to use [alternative labelling tools](https://github.com/agermanidis/pigeon) but you will have to make some code changes to get that to work.

## Contents 

- There is a `config.yml` file that contains definitions of the labels. All scripts will make assumptions based on the contents of this file. 
- There is a [taskfile](https://taskfile.dev/) that contains some common commands. 
- There is a `.github` folder that contains all the cronjobs.
- There is a `frontpage` Python module that contains all the logic to prepare data for annotation, to train sentence-models and to build the new site. 

