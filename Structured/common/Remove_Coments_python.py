import json
from pathlib import Path
import  RemoveComents as rmcm
def process_file(file_path, data):
    with open(file_path) as f:
        if "jsonl" in file_path:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                if 'function_tokens' in js:
                    js['function'] = rmcm.remove_comments(js['function'])
                else:
                    js['code'] = rmcm.remove_comments(js['code'])
                data.append(js)
        elif "codebase" in file_path or "code_idx_map" in file_path:
            print("error here!")
            exit(0)
            js = json.load(f)
            print(js)
            for key in js:
                temp = {}
                temp['code'] = key
                temp['code_tokens'] = key.split()
                temp["retrieval_idx"] = js[key]
                temp['doc'] = ""
                temp['docstring_tokens'] = ""
                data.append(temp)
        elif "json" in file_path:
            print("error")
            exit(0)
            for js in json.load(f):
                data.append(js)

# Example usage
file_path = "../dataset/AdvTest/test.jsonl"
data = []
process_file(file_path, data)

# Save the modified data to a new JSONL file
new_file_path = str(Path(file_path).parent / (Path(file_path).stem + "_2.jsonl"))
print(new_file_path)
file_path = "../dataset/AdvTest/test2.jsonl"
with open(new_file_path, 'w') as f_out:
    for item in data:
        f_out.write(json.dumps(item) + '\n')