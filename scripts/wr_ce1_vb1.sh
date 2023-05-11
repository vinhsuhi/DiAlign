CUDA_VISIBLE_DEVICES=0 python src/align.py model.loss_type='hybrid' model.sample_mode='wr' model.ce_weight=1 model.vb_weight=0.0 &
CUDA_VISIBLE_DEVICES=1 python src/align.py model.loss_type='hybrid' model.sample_mode='wor' model.ce_weight=1 model.vb_weight=0.0 &
CUDA_VISIBLE_DEVICES=2 python src/align.py model.loss_type='hybrid' model.sample_mode='wr' model.ce_weight=1 model.vb_weight=1 &
CUDA_VISIBLE_DEVICES=3 python src/align.py model.loss_type='hybrid' model.sample_mode='wor' model.ce_weight=1 model.vb_weight=1 &
CUDA_VISIBLE_DEVICES=4 python src/align.py model.loss_type='hybrid' model.sample_mode='wr' model.ce_weight=1 model.vb_weight=0.1 &
CUDA_VISIBLE_DEVICES=5 python src/align.py model.loss_type='hybrid' model.sample_mode='wor' model.ce_weight=1 model.vb_weight=0.1 &



python src/align.py model.loss_type='hybrid' model.sample_mode='wr' model.ce_weight=1 model.vb_weight=1

python src/align.py model.loss_type='hybrid' model.sample_mode='wr' model.ce_weight=1 model.vb_weight=1

python src/align.py model.loss_type='hybrid' model.sample_mode='wr' model.ce_weight=1 model.vb_weight=1

python src/align.py model.loss_type='hybrid' model.sample_mode='wr' model.ce_weight=1 model.vb_weight=1
