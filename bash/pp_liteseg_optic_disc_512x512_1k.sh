#!/bin/bash

# configs/runtime.yml中save_dir改为: output/faster_rcnn_r50_fpn_1x_coco
# configs/datasets/coco_detection.yml中dataset_dir改为: dataset/coco/cocomini
# configs/faster_rcnn/_base_/optimizer_1x.yml中learning_rate改为: 1.25e-3

selected_gpus="0"
configs_file=configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml
output_dir=output/pp_liteseg_optic_disc_512x512_1k

if [ ! -d ${output_dir} ]; then
    mkdir -p "$output_dir"
fi

nv_gpu=$(lspci | grep -i nvidia)
hygon_gpu=$(lspci | grep -i display | grep -i haiguang)

if [ -n "$nv_gpu" ]; then
    echo "Nvidia GPU is detected"
    export CUDA_VISIBLE_DEVICES=$selected_gpus
elif [ -n "$hygon_gpu" ]; then
    echo "Hygon GPU is detected"
    export HIP_VISIBLE_DEVICES=$selected_gpus
else
    echo "No GPU is detected"
    exit 1
fi

# dataset preparation
dataset=data/optic_disc_seg.zip
dataset_dir=data/optic_disc_seg
if [ -f "$dataset" ]; then
    echo "Dataset is already downloaded."
    if [ -d "$dataset_dir" ]; then
        echo "Dataset is already unzipped."
    else
        unzip data/optic_disc_seg.zip -d data
    fi
else
    wget -P data/ https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip
    unzip data/optic_disc_seg.zip -d data
fi

# training standalone
python tools/train.py --config ${configs_file} \
    --save_interval 500 --do_eval \
    --use_vdl --save_dir ${output_dir} >"${output_dir}/train.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Training failed"
    exit 1
else
    echo "Training finished, log saved in ${output_dir}/train.log"
fi

# evaluation
python tools/val.py --config ${configs_file} \
    --model_path ${output_dir}/best_model/model.pdparams >"${output_dir}/eval.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Evaluation failed"
    exit 1
else
    echo "Evaluation finished, log saved in ${output_dir}/eval.log"
fi

# inference
python tools/predict.py --config ${configs_file} \
    --model_path ${output_dir}/best_model/model.pdparams \
    --image_path data/optic_disc_seg/JPEGImages/H0002.jpg \
    --save_dir ${output_dir}/result >"${output_dir}/infer.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Inference failed"
    exit 1
else
    echo "Inference finished, log saved in ${output_dir}/infer.log"
fi
