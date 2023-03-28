for i in 0.01 0.02 0.03 0.04 0.05;  
do 
echo "label rate is:" $i
echo "run cora" 
# python run_cora_label_scarce.py $i
echo "run citeseer"
# python run_citeseer_label_scarce.py $i
echo "run pubmed"
python run_pubmed_label_scarce.py $i
done 