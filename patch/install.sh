pip install transformers==4.45.2
transformers_path=$(python -c "import os; import transformers; transformers_dir = os.path.dirname(transformers.__file__); print(transformers_dir)")
echo $transformers_path

cp patch/modeling_qwen2.py $transformers_path/models/qwen2
cp patch/modeling_llama.py $transformers_path/models/llama
cp patch/modeling_t5.py $transformers_path/models/t5
cp patch/trainer.py $transformers_path

# cp src/llamafactory/model/modeling_llama.py $transformers_path/models/llama
# cp src/llamafactory/model/modeling_t5.py $transformers_path/models/t5
# cp src/llamafactory/model/trainer.py $transformers_path
# python run_patch.py --package_path $transformers_path/models/llama --patch_files modeling_llama.py
# python run_patch.py --package_path $transformers_path/models/t5 --patch_files modeling_t5.py
# python run_patch.py --package_path $transformers_path --patch_files trainer.py
#  bash config/run2.sh 1 0 llama3.1-8b vida 1 constant 1e-4 0 all 30 8 -1 2 4 0 0 4 1 1 2 0 1