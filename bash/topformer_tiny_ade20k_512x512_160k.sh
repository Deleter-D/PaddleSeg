#!/bin/bash

# configs/topformer/topformer_tiny_ade20k_512x512_160k.yml中
# iters改为: 16000
# batch_size改为: 16
# learning_rate改为: 0.0003

selected_gpus="0"
configs_file=configs/topformer/topformer_tiny_ade20k_512x512_160k.yml
output_dir=output/topformer_tiny_ade20k_512x512_160k

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

# training standalone
python tools/train.py --config ${configs_file} \
    --save_interval 1000 --do_eval \
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
    --image_path data/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg \
    --save_dir ${output_dir}/result >"${output_dir}/infer.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Inference failed"
    exit 1
else
    echo "Inference finished, log saved in ${output_dir}/infer.log"
fi
