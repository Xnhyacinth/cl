pip install transformers==4.45.2
transformers_path=$(python -c "import os; import transformers; transformers_dir = os.path.dirname(transformers.__file__); print(transformers_dir)")
echo $transformers_path
python run_patch.py --package_path $transformers_path/models/llama --patch_files modeling_llama.py
python run_patch.py --package_path $transformers_path/models/t5 --patch_files modeling_t5.py
python run_patch.py --package_path $transformers_path --patch_files trainer.py