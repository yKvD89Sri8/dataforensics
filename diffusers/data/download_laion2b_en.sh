mkdir laion2B-en
cd laion2B-en
for i in {00000..00003}; 
do wget https://huggingface.co/datasets/laion/laion2B-en/resolve/main/part-$i-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet; 
done
cd ..
