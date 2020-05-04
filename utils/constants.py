import os

import git

COL_HEADLINE = "news_headline"
COL_ARTICLE = "news_article"
COL_CATEGORY = "news_category"
COL_DATE = "date"

repo = git.Repo(".", search_parent_directories=True)
ROOT_DIR_PATH = repo.working_tree_dir

LOG_CONFIG_PATH = os.path.join(ROOT_DIR_PATH, "utils/logging.conf")
DATA_PATH = os.path.join(ROOT_DIR_PATH, "web_scraping/data")
