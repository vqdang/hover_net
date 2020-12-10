python run_infer.py \
--gpu '0,1' \
--nr_types 6 \
--type_info_path=type_info.json \
--batch_size 64 \
--model_mode 'pannuke' \
--model_path ../pretrained/pecan-hover-net-pytorch.tar \
--nr_inference_workers 8 \
--nr_post_proc_workers 16 \
tile \
--input_dir=dataset/sample_tiles/imgs/ \
--output_dir=dataset/sample_tiles/pred/ \

# "--gpu=0",
# "--nr_types=6",
# "--type_info_path=type_info.json",
# "--batch_size=8",
# "--model_mode=fast",
# "--model_path=../pretrained/pecan-hover-net-pytorch.tar",
# "--nr_inference_workers=4",
# "--nr_post_proc_workers=2",
# "tile",
# "--input_dir=dataset/sample_tiles/imgs/",
# "--output_dir=dataset/sample_tiles/pred/",
# "--draw_dot",
# "--save_qupath",