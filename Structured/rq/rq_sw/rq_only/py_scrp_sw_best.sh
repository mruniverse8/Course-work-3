#!/bin/bash
# Run the Python script and redirect both stdout and stderr to the log file
 python3 task2_simplified.py      --output_dir ../../../saved_models/AdvTest/      --model_name_or_path microsoft/unixcoder-base             --do_test      --eval_data_file ../../../dataset/AdvTest/valid.jsonl      --test_data_file ../../../dataset/AdvTest/test_2.jsonl      --codebase_file ../../../dataset/AdvTest/test_2.jsonl      --num_train_epochs 10      --code_length 256      --nl_length 128      --train_batch_size 64      --eval_batch_size 64      --learning_rate 2e-5      --seed 123456 --line_sw 15 &> "./res_py/output1.log"
echo "Python script execution completed. Output saved to log1"
 python3 task2_simplified.py      --output_dir ../../../saved_models/AdvTest/      --model_name_or_path microsoft/unixcoder-base             --do_test      --eval_data_file ../../../dataset/AdvTest/valid.jsonl      --test_data_file ../../../dataset/AdvTest/test_2.jsonl      --codebase_file ../../../dataset/AdvTest/test_2.jsonl      --num_train_epochs 10      --code_length 256      --nl_length 128      --train_batch_size 64      --eval_batch_size 64      --learning_rate 2e-5      --seed 123456 --line_sw 10 &> "./res_py/output2.log"
echo "Python script execution completed. Output saved to log2"
 python3 task2_simplified.py      --output_dir ../../../saved_models/AdvTest/      --model_name_or_path microsoft/unixcoder-base             --do_test      --eval_data_file ../../../dataset/AdvTest/valid.jsonl      --test_data_file ../../../dataset/AdvTest/test_2.jsonl      --codebase_file ../../../dataset/AdvTest/test_2.jsonl      --num_train_epochs 10      --code_length 256      --nl_length 128      --train_batch_size 64      --eval_batch_size 64      --learning_rate 2e-5      --seed 123456 --line_sw 25 &> "./res_py/output3.log"
echo "Python script execution completed. Output saved to log3"
 python3 task2_simplified.py      --output_dir ../../../saved_models/AdvTest/      --model_name_or_path microsoft/unixcoder-base             --do_test      --eval_data_file ../../../dataset/AdvTest/valid.jsonl      --test_data_file ../../../dataset/AdvTest/test_2.jsonl      --codebase_file ../../../dataset/AdvTest/test_2.jsonl      --num_train_epochs 10      --code_length 256      --nl_length 128      --train_batch_size 64      --eval_batch_size 64      --learning_rate 2e-5      --seed 123456 --line_sw 45 &> "./res_py/output4.log"
echo "Python script execution completed. Output saved to log0"
 python3 task2_simplified.py      --output_dir ../../../saved_models/AdvTest/      --model_name_or_path microsoft/unixcoder-base             --do_test      --eval_data_file ../../../dataset/AdvTest/valid.jsonl      --test_data_file ../../../dataset/AdvTest/test_2.jsonl      --codebase_file ../../../dataset/AdvTest/test_2.jsonl      --num_train_epochs 10      --code_length 256      --nl_length 128      --train_batch_size 64      --eval_batch_size 64      --learning_rate 2e-5      --seed 123456 --line_sw 88 &> "./res_py/output5.log"
echo "Python script execution completed. Output saved to log4"



python3 task2_merge.py      --output_dir ../../../saved_models/CSNjava/      --model_name_or_path microsoft/unixcoder-base       --do_zero_shot      --do_test      --eval_data_file ../../../dataset/AdvTest/valid.jsonl      --test_data_file ../../../dataset/AdvTest/test.jsonl      --codebase_file ../../../dataset/AdvTest/test.jsonl      --num_train_epochs 2      --code_length 256      --nl_length 128      --train_batch_size 64      --eval_batch_size 64      --learning_rate 2e-5      --seed 123456