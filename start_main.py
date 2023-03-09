import os




alpha=[2.3,2.5,2.7]
lr=0.001
l2=1e-3
gpu_id=3
dataset='amazon'
n_negs={"amazon":16,"ali":32,"yelp2018":64}
# 
for i_alpha in alpha:
    start_cmd=f"nohup python main.py --dataset {dataset} --lr {lr} --batch_size 2048 --gpu_id {gpu_id} --n_negs {n_negs[dataset]} --alpha {i_alpha}  >{dataset}_alpha_{i_alpha}_hop012_0309.log 2>&1 &"
    print(start_cmd)
    os.system(start_cmd)










