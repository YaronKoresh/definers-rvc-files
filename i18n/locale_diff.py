import json
import os
from collections import OrderedDict

                               
standard_file = "locale/zh_CN.json"

                                      
dir_path = "locale/"
languages = [
    os.path.join(dir_path, f)
    for f in os.listdir(dir_path)
    if f.endswith(".json") and f != standard_file
]

                        
with open(standard_file, "r", encoding="utf-8") as f:
    standard_data = json.load(f, object_pairs_hook=OrderedDict)

                                 
for lang_file in languages:
                            
    with open(lang_file, "r", encoding="utf-8") as f:
        lang_data = json.load(f, object_pairs_hook=OrderedDict)

                                                                         
    diff = set(standard_data.keys()) - set(lang_data.keys())

    miss = set(lang_data.keys()) - set(standard_data.keys())

                                               
    for key in diff:
        lang_data[key] = key

                                             
    for key in miss:
        del lang_data[key]

                                                                                
    lang_data = OrderedDict(
        sorted(lang_data.items(), key=lambda x: list(standard_data.keys()).index(x[0]))
    )

                                    
    with open(lang_file, "w", encoding="utf-8") as f:
        json.dump(lang_data, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")
