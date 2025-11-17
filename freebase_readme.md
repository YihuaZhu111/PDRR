# Freebase Setup

## Requirements

- OpenLink Virtuoso 7.2.6 (download from this public [link](https://sourceforge.net/projects/virtuoso/files/virtuoso/))
- Python 3
- Freebase dump from this public [link](https://developers.google.com/freebase?hl=en)
- We use version 7.2.6 for the entity ID pattern match search

## Setup

### Data Preprocessing

We use this py script (public [link)](https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/code/FreebaseTool/FilterEnglishTriplets.py), to clean the data and remove non-English or non-digital triplets:

```shell
gunzip -c freebase-rdf-latest.gz > freebase # data size: 400G
nohup python -u FilterEnglishTriplets.py 0<freebase 1>FilterFreebase 2>log_err & # data size: 125G
```

## Import data

we import the cleaned data to virtuoso, 

```shell
tar xvpfz virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
cd virtuoso-opensource/database/
mv virtuoso.ini.sample virtuoso.ini
```

Now, you should check the content of your `virtuoso-opensource/database/virtuoso.ini` file. Around lines 116 and 117, you will find the following two lines:

```
NumberOfBuffers          = ?
MaxDirtyBuffers          = ?
```

Please modify these values according to your server’s CPU and memory capacity, because using the default settings may result in very slow performance.


```
# ../bin/virtuoso-t -df # start the service in the shell
../bin/virtuoso-t  # start the service in the backend.
../bin/isql 1111 dba dba # run the database

# 1、unzip the data and use rdf_loader to import
SQL>
ld_dir('.', 'FilterFreebase', 'http://freebase.com'); 
rdf_loader_run(); 
```

Wait for a long time and then ready to use.

## Mapping data to Wikidata

Due to the partial incompleteness of the data present in the freebase dump, we need to map some of the entities with missing partial relationships to wikidata. We download these rdf data via this public [link](https://developers.google.com/freebase?hl=en#freebase-wikidata-mappings)

we can use the above method to add it into virtuoso.

## Test example

run the code:
```
python test_freebase.py
```
to check whether Freebase can be used or not
## 
