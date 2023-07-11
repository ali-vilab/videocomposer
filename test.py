from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('damo/VideoComposer', cache_dir='model_weights/', revision='v1.0.0')


# use VideoComposer conda env: conda activate VideoComposer
# install everything in environment.yml, while in the environment: pip install -e .