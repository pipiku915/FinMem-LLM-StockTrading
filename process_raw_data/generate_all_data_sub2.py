import polars as pl
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    with open("/workspaces/ArkGPT/data/06_input/new_data.pkl", "rb") as f:
        env_data_pkl = pickle.load(f)
    new_data = {}
    for key, val in tqdm(env_data_pkl.items()):
        cur_eco = val["eco"]
        if isinstance(cur_eco["t10yff"], str):
            if cur_eco["t10yff"] == ".":
                print(key)
            else:
                cur_eco["t10yff"] = float(cur_eco["t10yff"])
                val["eco"] = cur_eco
                new_data[key] = val
                cur_eco["awhaeman"] = cur_eco["AWHAEMAN"]
                cur_eco["icsa"] = cur_eco["ICSA"]
                del cur_eco["AWHAEMAN"]
                del cur_eco["ICSA"]
        else:
            raise TypeError("t10yff is not str")
    with open("/workspaces/ArkGPT/data/06_input/all_symbols.pkl", "wb") as f:
        pickle.dump(new_data, f)

# TODO: data removed
# 2021-11-11
# 2022-10-10
# 2022-11-11
