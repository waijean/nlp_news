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
- Process columns to appropriate dtype
- Convert csv into parquet file for faster loading and to preserve dtype 

Problem: 

The original csv file (8.2GB) is too large to be loaded into RAM using pandas. Therefore, dask 
has been tried to process the csv file as dask automatically split the original csv file 
into several partitions and process them separately. However, each line in the csv file is not 
necessarily a new row. A row may span multiple lines if the article has multiple line breaks. This 
becomes a problem when the partition divisions happens in the middle of article. This will throw a 
"field larger than field limit (131072)" error because pandas can't find the correct line endings. 

Solution:

Therefore, bash script is used to remove the blank lines from the csv file and split the large csv 
file into smaller chunks (max 1GB each so 9 smaller csv files in total). Then, each file is checked 
to make sure that the partitions occur at the end of the article. It turns out that only the 
news_csv_03 and news_csv_04 have incorrect divisions so manual effort is required to fix it. 

Once this is fixed, pandas can process each of the 9 csv files separately and convert each of them to 
parquet file. Then, dask can be used to read the 9 parquet files collectively and process them in the
next steps of our pipeline. 

Result:
Dropping the null section reduces the size of the dataset from 2.7m rows to 1.7m rows with average 
1143 articles per day. 

The top 20 sections are listed below:
Market News                  108724
World News                   108651
Business News                 96395
Wires                         67352
Financials                    57845
politics                      53496
us                            51242
Intel                         39805
Bonds News                    39672
Politics                      33875
Healthcare                    30883
world                         28530
opinion                       27465
Consumer Goods and Retail     26766
Sports News                   26324
business                      25335
tv                            24783
sports                        23909
Tech                          21605
arts                          21230
 
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
