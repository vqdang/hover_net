python run_infer.py \
--gpu='0,1' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=64 \
--model_mode=fast \
--model_path=../../pretrained/hovernet_fast_pannuke_pytorch.tar \
--nr_inference_workers=4 \
--nr_post_proc_workers=32 \
wsi \
--input_dir=exp_output/wsi/samples/full/ \
--output_dir=exp_output/wsi/pred/ \
--input_mask_dir=exp_output/wsi/samples/mask/ \
--save_thumb \
--save_mask \
--ambiguous_size=328 \
--tile_shape=2048