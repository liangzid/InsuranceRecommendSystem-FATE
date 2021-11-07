#!/bin/bash


export root_path=/home/szhang/liangzi_need_smile/fatee/isr/
export fate_path="your fate path."

export feature_dim=256
lr=2e-3
dataset_path="${root_path}/BackEnd/data.nlf_training_data.pk"
batch_size=32
save_path=${root_path}BackEnd/src/core/

echo "Begin to generate Config files, see ${save_path}"

python generateNCFJSONConfigs.py \
       --lr=$lr\
       --dataset_path=$dataset_path\
       --batch_size=$batch_size\
       --feature_dim=$feature_dim\
       --save_path=$save_path

echo "begin to upload."
python "${fate_path}/python/fate_flow/fate_flow_client.py" -f upload -c "${save_path}upload.json"


echo "begin to deploy and RUN."
python "${fate_path}/python/fate_flow/fate_flow_client.py" \
       -f submit_job \
       -c "${save_path}/ncf_conf.json" \
       -d "${save_path}/ncf_dsl.json"


echo "all things DONE."

echo "Please have a look in http://hostip:8080.\n username: admin\n password:admin\n"

echo ">>>DONE<<<"

