# Please run this shell in NewsRecommendation root_dir
mkdir data && cd data

# Glove
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip

# Small
mkdir MINDsmall && cd MINDsmall
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d val
cp -r val test
rm MINDsmall_*.zip
cd ..

