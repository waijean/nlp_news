#!/bin/bash

DATA_DIR=/d/dev/Project/nlp_news/data_pipeline/data/
CSV_PARTITION_DIR=$DATA_DIR/csv_partition
ORIGINAL_CSV_FILE=$DATA_DIR/all-the-news-2-1.csv
OUTPUT_CSV_FILE=$DATA_DIR/processed-all-the-news-2-1.csv
PYTHON_FILE=/d/dev/Project/nlp_news/data_pipeline/partition_news.py
PREFIX=news_csv_

echo "Removing blank lines in large csv file"
echo "Number of lines before removing blank lines: $(wc -l $ORIGINAL_CSV_FILE | cut -f1 -d' ')"
sed '/^$/d' $ORIGINAL_CSV_FILE > $OUTPUT_CSV_FILE && echo "Processed csv file without blank lines: $OUTPUT_CSV_FILE"
echo "Number of lines after removing blank lines $(wc -l $OUTPUT_CSV_FILE | cut -f1 -d' ')"

echo "Splitting large csv file into smaller csv files"
echo "Size of the large csv file: $(du -h $OUTPUT_CSV_FILE | cut -f1)"
cd $CSV_PARTITION_DIR
split -C 1G -d $ORIGINAL_CSV_FILE $PREFIX && echo "Large csv file has been splitted to $CSV_PARTITION_DIR"
echo -e "Size of the smaller csv files: \n$(du -h -a $CSV_PARTITION_DIR)"

echo "Inserting header row to each smaller csv file except for the first file"
cd $CSV_PARTITION_DIR
for file in $PREFIX*; do
  echo $file
  if [ $file = "news_csv_00" ]; then
      FIRST_LINE=$(sed '1q;d' $file)
      echo "Skipping $file"
      continue;
  fi
  sed '1q;d' $file
  sed -i "1 i $FIRST_LINE" $file
  sed '1q;d' $file
done && echo "Header row has been inserted"

echo "Adding the start of news_csv_04 to the end of news_csv_03"
sed -n '2,125p' data_pipeline/data/csv_partition/news_csv_04 >> data_pipeline/data/csv_partition/news_csv_03
echo "Verify the end of news_csv_03"
sed -n '712854,$p' data_pipeline/data/csv_partition/news_csv_03
echo "Delete the start of news_csv_04"
sed -i '2,125d' data_pipeline/data/csv_partition/news_csv_04