mkdir laion2b_chinese_release
cd laion2b_chinese_release
for i in {00000..00012}; 
do wget https://huggingface.co/datasets/IDEA-CCNL/laion2B-multi-chinese-subset/resolve/main/data/train-$i-of-00013.parquet; 
done
cd ..

