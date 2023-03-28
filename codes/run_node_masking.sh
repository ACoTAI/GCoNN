for i in 0.1 0.3 0.5 0.7 0.9;  
do 
echo "drop rate is:" $i
echo "run cora" 
python run_cora_node_masking.py $i 1
python run_cora_node_masking.py $i 0
echo "run citeseer"
python run_citeseer_node_masking.py $i 1
python run_citeseer_node_masking.py $i 0
echo "run pubmed"
python run_pubmed_node_masking.py $i 1
python run_pubmed_node_masking.py $i 0
done 