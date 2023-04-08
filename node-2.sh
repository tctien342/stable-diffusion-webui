export CUDA_VISIBLE_DEVICES=1
export COMMANDLINE_ARGS="--xformers --xformers-flash-attention --port 7861 --opt-channelslast --upcast-sampling --no-half-vae --medvram --listen --no-hashing --skip-version-check --api --enable-insecure-extension-access "

bash ./webui.sh