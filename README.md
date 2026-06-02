# Reflective-SQL-Debugging-Agent

## Dataset Builder
 
To generate the dataset, run:
 
```bash
python3 data/dataset-builder.py \
  --spider_dev data/spider/dev.json \
  --spider_tables data/spider/tables.json \
  --output data/dataset.json
```