export CUDA_VISIBLE_DEVICES=0
export COMMANDLINE_ARGS="--xformers --xformers-flash-attention --port 7860 --opt-channelslast --upcast-sampling --no-half-vae --medvram --listen --no-hashing --skip-version-check --api --enable-insecure-extension-access "

bash ./webui.sh