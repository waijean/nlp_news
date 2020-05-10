# nlp_news

The project's aim is to use Natural Language Processing (NLP) to perform sentiment analysis on news 
articles and pass that as a feature to predict exchange traded fund (ETF) prices.

News archive: https://components.one/datasets/all-the-news-2-news-articles-dataset/  
News website: https://inshorts.com/en/read/business  
Stock price: https://www.sharesmagazine.co.uk/shares/share/VEVE/historic-prices

## Data Pipeline
Aim:
- Extract only 4 columns (date, title, article and section)
- Drop null section
- Process date and convert to datetime dtype
- Convert csv into parquet file for faster loading and to preserve dtype

Problem: 

The original csv file (8.2GB) is too large to be loaded into RAM using pandas. Therefore, dask 
has been tried to process the csv file. However, there might be incorrect line endings within the 
columns and dask is not able to process those (https://github.com/dask/dask/issues/4145). If the 
blank lines are removed from the original csv file and the original csv file is split into smaller 
files (1GB each), dask is able to read the first csv partition using c engine. However, dask is still
not able to read the whole csv file. 

Current Solution:

Pass in error_bad_lines as False to drop the error lines. According to the warning logs, majority of
them are field larger than field limit (131072), with a couple of unexpected end of data and ',' 
expected after '"'. This might indicate that some columns are being read incorrectly as a result
of dropping erros lines. This might cause the date column to be read incorrectly and as a result 
there is a large spike of articles on Jan 1 2019 (18k compared to median 1.3k). The number of null 
values in section column is 905,493. 

Idea:
1. Increase csv field limit but this might not solve unexpected end of data error
2. Only process the first partition of csv file
3. Use bash script to remove lines that don't start with proper index. This is to remove articles
which have multiple lines which is suspected to be the root cause of the problem. 
 
## Data Cleaning
The data cleaning process is split into the following stages:
1. Tokenize the articles into words by using WordPunctTokenizer from nltk library
2. Remove standalone punctuations by using the regex library
3. Remove stopwords by using the nltk library
4. Lemmatize the words by using WordNetLemmatizer from nltk library (might switch to stemming due to
speed concern) 

## Data Preprocessing

 
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for 
development and testing purposes. See deployment for notes on how to deploy the project on a live 
system.

The Python version used is 3.7.1 and it is recommended to create a virtual environment and install
the required packages:  
```bash
#cd to root directory
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

Next, pre-commit is used to run black, mypy check and nbstripout before committing to Git. The 
.pre-commit-config.yaml file is included in the repo so only the following step is required after 
installing the pre-commit package (which is included in the requirements.txt):
```bash
pre-commit install 
```

To use the virtual environment in notebook, you have to install ipykernel package first (which is also
included in the requirements.txt). Then, you need to add your virtual environment to Jupyter:
```bash
python -m ipykernel install --user --name=venv
```

## Deployment

To deploy project on a live system
