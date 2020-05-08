#!/bin/bash

DATA_DIR=/d/dev/Project/nlp_news/data_pipeline/data/
CSV_PARTITION_DIR=$DATA_DIR/csv_partition
ORIGINAL_CSV_FILE=$DATA_DIR/all-the-news-2-1.csv
OUTPUT_CSV_FILE=$DATA_DIR/processed-all-the-news-2-1.csv
PYTHON_FILE=/d/dev/Project/nlp_news/data_pipeline/partition_news.py

#echo "Splitting large csv file into smaller csv files"
#echo "Size of the large csv file: $(du -h $ORIGINAL_CSV_FILE | cut -f1)"
#cd CSV_PARTITION_DIR
#split -C 1G -d ORIGINAL_CSV_FILE news_csv_ && echo "Large csv file has been splitted to $CSV_PARTITION_DIR"
#echo -e "Size of the smaller csv files: \n$(du -h -a $CSV_PARTITION_DIR)"

echo "Removing blank lines in large csv file"
echo "Number of lines before removing blank lines: $(wc -l $ORIGINAL_CSV_FILE | cut -f1 -d' ')"
sed '/^$/d' $ORIGINAL_CSV_FILE > $OUTPUT_CSV_FILE && echo "Processed csv file without blank lines: $OUTPUT_CSV_FILE"
echo "Number of lines after removing blank lines $(wc -l $OUTPUT_CSV_FILE | cut -f1 -d' ')"