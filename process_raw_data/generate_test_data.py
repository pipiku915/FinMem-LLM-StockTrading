import pickle
from tqdm import tqdm

#### change the target symbol here
symbol_subset = ["TSLA"]

new_dict = {}

with open("data/06_input/all_symbols.pkl", "rb") as f:
    data = pickle.load(f)

for k, v in tqdm(data.items()):
    cur_price = v["price"]
    cur_eco = v["eco"]
    cur_filing_k = v["filing_k"]
    cur_filing_q = v["filing_q"]
    cur_news = v["news"]
    cur_ark_record = v["ark_record"]

    new_price = {}
    new_filing_k = {}
    new_filing_q = {}
    new_news = {}
    new_ark_record = {}
    for cur_symbol in symbol_subset:
        if cur_symbol in cur_price:
            new_price[cur_symbol] = cur_price[cur_symbol]
        if cur_symbol in cur_filing_k:
            new_filing_k[cur_symbol] = cur_filing_k[cur_symbol]
        if cur_symbol in cur_filing_q:
            new_filing_q[cur_symbol] = cur_filing_q[cur_symbol]
        if cur_symbol in cur_news:
            new_news[cur_symbol] = cur_news[cur_symbol]
        if cur_symbol in cur_ark_record:
            new_ark_record[cur_symbol] = cur_ark_record[cur_symbol]
        else:
            continue

    new_dict[k] = {
        "price": new_price,
        "eco": cur_eco,
        "filing_k": new_filing_k,
        "filing_q": new_filing_q,
        "news": new_news,
        "ark_record": new_ark_record,
    }

with open("data/06_input/subset_symbols.pkl", "wb") as f:
    pickle.dump(new_dict, f)
