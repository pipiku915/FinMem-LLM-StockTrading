import os
import faiss
import random
import pickle
import faiss
import logging
import shutil
import numpy as np
from sortedcontainers import SortedList
from itertools import repeat
from .embedding import get_embedding_func, EmbeddingFunction
from typing import List, Union, Dict, Any, Tuple, Callable
from .memory_functions import (
    ImportanceScoreInitialization,
    RecencyScoreInitialization,
    CompoundScoreCalculation,
    DecayFunctions,
    get_importance_score_initialization_func,
    get_recency_score_initialization_func,
    get_compound_score_calculation_func,
    get_decay_func,
    get_access_counter_change_func,
    ImportanceScoreChangeAccessCounter,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler("run.log", mode="a")
file_handler.setFormatter(logging_formatter)
logger.addHandler(file_handler)


class id_generator_func:
    def __init__(self):
        self.current_id = 0

    def __call__(self):
        self.current_id += 1
        return self.current_id - 1


class MemoryDB:
    def __init__(  # test pass
        self,
        db_name: str,
        id_generator: Callable,
        jump_threshold_upper: float,
        jump_threshold_lower: float,
        embedding_function: EmbeddingFunction,
        importance_score_initialization: ImportanceScoreInitialization,
        recency_score_initialization: RecencyScoreInitialization,
        compound_score_calculation: CompoundScoreCalculation,
        importance_score_change_access_counter: ImportanceScoreChangeAccessCounter,
        decay_function: DecayFunctions,
        clean_up_threshold_dict: Dict[
            str, float
        ],  # {"recency_threshold": x, "importance_threshold": y"}
        use_gpu: bool = True,
    ) -> None:
        # general attributes
        self.use_gpu = use_gpu
        self.db_name = db_name
        self.id_generator = id_generator
        self.jump_threshold_upper = jump_threshold_upper
        self.jump_threshold_lower = jump_threshold_lower
        self.emb_dim = embedding_function.get_embedding_dimension()
        self.emb_func = embedding_function
        self.importance_score_initialization_func = importance_score_initialization
        self.recency_score_initialization_func = recency_score_initialization
        self.compound_score_calculation_func = compound_score_calculation
        self.decay_function = decay_function
        self.importance_score_change_access_counter = (
            importance_score_change_access_counter
        )
        self.clean_up_threshold_dict = dict(clean_up_threshold_dict)
        if use_gpu:
            self.res = faiss.StandardGpuResources()

        # record for each equity
        self.universe = {}

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], agent_name: str, memory_layer: str
    ) -> "MemoryDB":
        return cls(
            db_name=f"{agent_name}_{memory_layer}",
            jump_threshold_upper=config[agent_name][memory_layer][
                "jump_threshold_upper"
            ],
            jump_threshold_lower=config[agent_name][memory_layer][
                "jump_threshold_lower"
            ],
            embedding_function=get_embedding_func(config[agent_name]["embedding"]),
            importance_score_initialization=get_importance_score_initialization_func(
                type=config[agent_name][memory_layer][
                    "importance_score_initialization_type"
                ],
                memory_layer=memory_layer,
            ),
            recency_score_initialization=get_recency_score_initialization_func(
                type=config[agent_name][memory_layer]["constant"],
                memory_layer=memory_layer,
            ),
            compound_score_calculation=get_compound_score_calculation_func(
                type="linear", memory_layer=memory_layer
            ),
            decay_function=get_decay_func(
                type=config[agent_name][memory_layer]["decay_type"],
                memory_layer=memory_layer,
                **config[agent_name][memory_layer]["decay_params"],
            ),
            clean_up_threshold_dict=config[agent_name][memory_layer][
                "clean_up_threshold_dict"
            ],
            importance_score_change_access_counter=get_access_counter_change_func(
                config[agent_name][memory_layer][
                    "importance_score_change_access_counter"
                ],
                memory_layer,
            ),
            use_gpu=config["general"]["use_gpu"],
        )

    def add_new_symbol(self, symbol: str) -> None:  # test pass
        cur_index = faiss.IndexFlatIP(
            self.emb_dim
        )  # normalized inner product is cosine similarity
        cur_index = faiss.IndexIDMap2(cur_index)
        if self.use_gpu:
            cur_index = faiss.index_cpu_to_gpu(
                self.res, 0, cur_index
            )  # this part will be very slow at the first time as cuda need to compile some files
        temp_record = {
            "score_memory": SortedList(
                key=lambda x: x["important_score_recency_compound_score"]
            ),
            "index": cur_index,
        }
        self.universe[symbol] = temp_record

    def add_memory(
        self, symbol: str, date: str, text: Union[List[str], str]
    ) -> None:  # test pass
        # add new symbol if not exist
        if symbol not in self.universe:
            self.add_new_symbol(symbol)

        if isinstance(text, str):
            text = [text]
        # get embedding
        emb = self.emb_func(text)
        faiss.normalize_L2(emb)
        ids = [self.id_generator() for _ in range(len(text))]
        # initialize importance score
        importance_scores = [
            self.importance_score_initialization_func() for _ in range(len(text))
        ]
        # recency
        recency_scores = [
            self.recency_score_initialization_func() for _ in range(len(text))
        ]
        # calculate partial score
        partial_scores = [
            self.compound_score_calculation_func.recency_and_importance_score(
                recency_score=cur_r, importance_score=cur_i
            )
            for cur_i, cur_r in zip(importance_scores, recency_scores)
        ]
        self.universe[symbol]["index"].add_with_ids(emb, np.array(ids))
        for i in range(len(text)):
            self.universe[symbol]["score_memory"].add(
                {
                    "text": text[i],
                    "id": ids[i],
                    "important_score": importance_scores[i],
                    "recency_score": recency_scores[i],
                    "delta": 0,
                    "important_score_recency_compound_score": partial_scores[i],
                    "access_counter": 0,
                    "date": date,
                }
            )
            # TODO: debug
            logger.info(
                {
                    "text": text[i],
                    "id": ids[i],
                    "important_score": importance_scores[i],
                    "recency_score": recency_scores[i],
                    "delta": 0,
                    "important_score_recency_compound_score": partial_scores[i],
                    "access_counter": 0,
                    "date": date,
                }
            )
            # TODO: debug

    def query(
        self, query_text: str, top_k: int, symbol: str
    ) -> Tuple[List[str], List[int]]:  # test pass
        if (
            (symbol not in self.universe)
            or (len(self.universe[symbol]["score_memory"]) == 0)
            or (top_k == 0)
        ):
            return [], []
        max_len = len(self.universe[symbol]["score_memory"])
        top_k = min(top_k, max_len)
        cur_index = self.universe[symbol]["index"]
        emb = self.emb_func(query_text)
        # temp dict ranking
        temp_text_list = []
        temp_score = []
        temp_date_list = []
        temp_ids = []
        # top 5 similar query: part 1 search
        p1_dists, p1_ids = cur_index.search(emb, top_k)
        p1_dists, p1_ids = p1_dists[0].tolist(), p1_ids[0].tolist()
        for cur_sim, cur_id in zip(p1_dists, p1_ids):
            cur_record = next(
                (
                    record
                    for record in self.universe[symbol]["score_memory"]
                    if record["id"] == cur_id
                ),
                None,
            )
            temp_text_list.append(cur_record["text"])
            temp_date_list.append(cur_record["date"])
            temp_ids.append(cur_record["id"])
            temp_score.append(
                self.compound_score_calculation_func.merge_score(
                    cur_sim, cur_record["important_score_recency_compound_score"]
                )
            )
        # top 5 partial compound score: part 2 search
        p2_ids = [self.universe[symbol]["score_memory"][i]["id"] for i in range(top_k)]
        temp_arrays = [cur_index.reconstruct(i) for i in p2_ids]
        p2_emb = np.vstack(temp_arrays)
        temp_index = faiss.IndexFlatIP(self.emb_dim)
        temp_index = faiss.IndexIDMap2(temp_index)
        temp_index.add_with_ids(p2_emb, np.array(p2_ids))
        p2_dist, p2_ids = temp_index.search(emb, top_k)
        p2_dist, p2_ids = p2_dist[0].tolist(), p2_ids[0].tolist()
        for cur_sim, cur_id in zip(p2_dist, p2_ids):
            cur_record = next(
                (
                    record
                    for record in self.universe[symbol]["score_memory"]
                    if record["id"] == cur_id
                ),
                None,
            )
            temp_text_list.append(cur_record["text"])
            temp_date_list.append(cur_record["date"])
            temp_ids.append(cur_record["id"])
            temp_score.append(
                self.compound_score_calculation_func.merge_score(
                    cur_sim, cur_record["important_score_recency_compound_score"]
                )
            )
        # rank sort
        score_rank = np.argsort(temp_score)[::-1][:top_k]
        # filter unique list
        temp_ret_text_list = [temp_text_list[i] for i in score_rank]
        temp_ret_date_list = [temp_date_list[i] for i in score_rank]
        temp_ret_ids = [temp_ids[i] for i in score_rank]
        ret_text_list = []
        ret_date_list = []
        ret_ids = []
        _, unique_index = np.unique(temp_ret_ids, return_index=True)
        unique_index = unique_index.tolist()
        for i in unique_index:
            ret_text_list.append(temp_ret_text_list[i])
            ret_date_list.append(temp_ret_date_list[i])
            ret_ids.append(temp_ret_ids[i])

        my_list = list(range(len(ret_text_list)))  # 这样 my_list 将是 [0, 1]
        random_index = random.sample(
            my_list, min(top_k, len(my_list))
        )  # 这将确保随机索引在有效范围内

        selected_text = [ret_text_list[i] for i in random_index]
        selected_ids = [ret_ids[i] for i in random_index]
        return selected_text, selected_ids

    def update_access_count_with_feed_back(  # test pass
        self, symbol: str, ids: List[int], feedback: List[int]
    ) -> List[int]:
        if symbol not in self.universe:
            return []
        success_ids = []
        cur_score_memory = self.universe[symbol]["score_memory"]
        for cur_id, cur_feedback in zip(ids, feedback):
            for cur_record in cur_score_memory:
                if cur_record["id"] == cur_id:
                    cur_record["access_counter"] += cur_feedback
                    cur_record[
                        "important_score"
                    ] = self.importance_score_change_access_counter(
                        access_counter=cur_record["access_counter"],
                        importance_score=cur_record["important_score"],
                    )
                    cur_record[
                        "important_score_recency_compound_score"
                    ] = self.compound_score_calculation_func.recency_and_importance_score(
                        recency_score=cur_record["recency_score"],
                        importance_score=cur_record["important_score"],
                    )
                    # TODO: debug
                    logger.info("update_access_count_with_feed_back")
                    logger.info(cur_record)
                    # TODO: debug
                    success_ids.append(cur_id)
                    break
        self.universe[symbol]["score_memory"] = cur_score_memory
        return success_ids

    def _decay(self) -> None:
        # 1. decay importance score
        # 2. decay recency score
        for cur_symbol in self.universe:
            cur_score_memory = self.universe[cur_symbol]["score_memory"]
            for i in range(len(cur_score_memory)):
                (
                    cur_score_memory[i]["recency_score"],
                    cur_score_memory[i]["important_score"],
                    cur_score_memory[i]["delta"],
                ) = self.decay_function(
                    important_score=cur_score_memory[i]["important_score"],
                    delta=cur_score_memory[i]["delta"],
                )
                cur_score_memory[i][
                    "important_score_recency_compound_score"
                ] = self.compound_score_calculation_func.recency_and_importance_score(
                    recency_score=cur_score_memory[i]["recency_score"],
                    importance_score=cur_score_memory[i]["important_score"],
                )
            self.universe[cur_symbol]["score_memory"] = cur_score_memory

    def _clean_up(self) -> List[int]:
        ret_removed_ids = []
        for cur_symbol in self.universe:
            cur_score_memory = self.universe[cur_symbol]["score_memory"]
            if remove_ids := [
                cur_score_memory[i]["id"]
                for i in range(len(cur_score_memory))
                if (
                    cur_score_memory[i]["recency_score"]
                    < self.clean_up_threshold_dict["recency_threshold"]
                )
                or (
                    cur_score_memory[i]["important_score"]
                    < self.clean_up_threshold_dict["importance_threshold"]
                )
            ]:
                new_list = SortedList(
                    [], key=lambda x: x["important_score_recency_compound_score"]
                )
                for cur_object in cur_score_memory:
                    if cur_object["id"] not in remove_ids:
                        new_list.add(cur_object)
                self.universe[cur_symbol]["score_memory"] = new_list
                self.universe[cur_symbol]["index"].remove_ids(np.array(remove_ids))
                ret_removed_ids.extend(remove_ids)
        return ret_removed_ids

    def step(self) -> List[int]:
        self._decay()
        return self._clean_up()

    def prepare_jump(self) -> Tuple[Dict[str, Dict[str, Any]]]:
        jump_dict_up = {}
        jump_dict_down = {}
        id_to_remove = []
        for cur_symbol in self.universe:
            temp_delete_ids_up = []
            temp_jump_object_list_up = []
            temp_emb_list_up = []
            temp_delete_ids_down = []
            temp_jump_object_list_down = []
            temp_emb_list_down = []
            cur_score_memory = self.universe[cur_symbol]["score_memory"]
            for i in range(len(cur_score_memory)):
                if cur_score_memory[i]["important_score"] >= self.jump_threshold_upper:
                    temp_delete_ids_up.append(cur_score_memory[i]["id"])
                    temp_jump_object_list_up.append(cur_score_memory[i])
                    temp_emb_list_up.append(
                        self.universe[cur_symbol]["index"].reconstruct(
                            cur_score_memory[i]["id"]
                        )
                    )
                if cur_score_memory[i]["important_score"] < self.jump_threshold_lower:
                    temp_delete_ids_down.append(cur_score_memory[i]["id"])
                    temp_jump_object_list_down.append(cur_score_memory[i])
                    temp_emb_list_down.append(
                        self.universe[cur_symbol]["index"].reconstruct(
                            cur_score_memory[i]["id"]
                        )
                    )
            temp_delete_ids = temp_delete_ids_up + temp_delete_ids_down
            id_to_remove.extend(temp_delete_ids)
            self.universe[cur_symbol]["index"].remove_ids(np.array(temp_delete_ids))
            new_memory = SortedList(
                [], key=lambda x: x["important_score_recency_compound_score"]
            )
            for record in cur_score_memory:
                if record["id"] not in temp_delete_ids:
                    new_memory.add(record)
            self.universe[cur_symbol]["score_memory"] = new_memory
            if temp_jump_object_list_up:
                temp_emb_list_up = np.vstack(temp_emb_list_up)
                jump_dict_up[cur_symbol] = {
                    "jump_object_list": temp_jump_object_list_up,
                    "emb_list": temp_emb_list_up,
                }
            if temp_jump_object_list_down:
                temp_emb_list_down = np.vstack(temp_emb_list_down)
                jump_dict_down[cur_symbol] = {
                    "jump_object_list": temp_jump_object_list_down,
                    "emb_list": temp_emb_list_down,
                }
        return jump_dict_up, jump_dict_down, id_to_remove

    def accept_jump(self, jump_dict: Dict[str, Dict[str, Any]], direction: str) -> None:
        if direction not in ["up", "down"]:
            raise ValueError("direction must be either [up] or [down]")

        jump_dict = jump_dict[0] if direction == "up" else jump_dict[1]
        for cur_symbol in jump_dict:
            if cur_symbol not in self.universe:
                self.add_new_symbol(cur_symbol)
            new_ids = []
            for cur_object in jump_dict[cur_symbol]["jump_object_list"]:
                # new_ids.append(self.id_generator())
                new_ids.append(cur_object["id"])
                # cur_object["id"] = new_ids[-1]
                if direction == "up":
                    cur_object[
                        "recency_score"
                    ] = self.recency_score_initialization_func()
                    cur_object["delta"] = 0
            self.universe[cur_symbol]["score_memory"].update(
                jump_dict[cur_symbol]["jump_object_list"]
            )
            self.universe[cur_symbol]["index"].add_with_ids(
                jump_dict[cur_symbol]["emb_list"], np.array(new_ids)
            )

    def save_checkpoint(self, name: str, path: str, force: bool = False) -> None:
        if os.path.exists(os.path.join(path, name)):
            if not force:
                raise FileExistsError(f"Memory db {name} already exists")
            shutil.rmtree(os.path.join(path, name))
        os.mkdir(os.path.join(path, name))
        # save config dict
        state_dict = {
            "use_gpu": self.use_gpu,
            "db_name": self.db_name,
            "id_generator": self.id_generator,
            "jump_threshold_upper": self.jump_threshold_upper,
            "jump_threshold_lower": self.jump_threshold_lower,
            "emb_dim": self.emb_dim,
            "emb_func": self.emb_func,
            "importance_score_initialization_func": self.importance_score_initialization_func,
            "recency_score_initialization_func": self.recency_score_initialization_func,
            "compound_score_calculation_func": self.compound_score_calculation_func,
            "decay_function": self.decay_function,
            "importance_score_change_access_counter": self.importance_score_change_access_counter,
            "clean_up_threshold_dict": self.clean_up_threshold_dict,
        }
        with open(os.path.join(path, name, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)
        # save universe
        save_universe = {}
        for cur_symbol in self.universe:
            cur_record = self.universe[cur_symbol]
            faiss.write_index(
                self.universe[cur_symbol]["index"],
                os.path.join(path, name, f"{cur_symbol}.index"),
            )
            save_universe[cur_symbol] = {
                "score_memory": list(cur_record["score_memory"]),
                "index_save_path": os.path.join(path, name, f"{cur_symbol}.index"),
            }
        with open(os.path.join(path, name, "universe_index.pkl"), "wb") as f:
            pickle.dump(save_universe, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MemoryDB":
        # load state dict
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
        # load universe
        with open(os.path.join(path, "universe_index.pkl"), "rb") as f:
            universe = pickle.load(f)
        for cur_symbol in universe:
            universe[cur_symbol]["index"] = faiss.read_index(
                universe[cur_symbol]["index_save_path"]
            )
            universe[cur_symbol]["score_memory"] = SortedList(
                universe[cur_symbol]["score_memory"],
                key=lambda x: x["important_score_recency_compound_score"],
            )
            del universe[cur_symbol]["index_save_path"]
        # create object
        obj = cls(
            db_name=state_dict["db_name"],
            id_generator=state_dict["id_generator"],
            jump_threshold_upper=state_dict["jump_threshold_upper"],
            jump_threshold_lower=state_dict["jump_threshold_lower"],
            embedding_function=state_dict["emb_func"],
            importance_score_initialization=state_dict[
                "importance_score_initialization_func"
            ],
            recency_score_initialization=state_dict[
                "recency_score_initialization_func"
            ],
            compound_score_calculation=state_dict["compound_score_calculation_func"],
            importance_score_change_access_counter=state_dict[
                "importance_score_change_access_counter"
            ],
            decay_function=state_dict["decay_function"],
            clean_up_threshold_dict=state_dict["clean_up_threshold_dict"],
            use_gpu=state_dict["use_gpu"],
        )
        obj.universe = universe.copy()
        return obj


class BrainDB:
    def __init__(
        self,
        agent_name: str,
        id_generator: id_generator_func,
        embedding_function: EmbeddingFunction,
        short_term_memory: MemoryDB,
        mid_term_memory: MemoryDB,
        long_term_memory: MemoryDB,
        reflection_memory: MemoryDB,
        use_gpu: bool = True,
    ):
        self.agent_name = agent_name
        self.embedding_function = embedding_function
        self.use_gpu = use_gpu
        self.id_generator = id_generator
        # memory layers
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.reflection_memory = reflection_memory
        # removed ids
        self.removed_ids = []

    @classmethod
    def from_config(cls, config: Dict[str, Any], agent_name: str) -> "BrainDB":
        # other states
        id_generator = id_generator_func()
        embedding_function = get_embedding_func(
            type=config[agent_name]["embedding"]["type"],
            config=config[agent_name]["embedding"]["detail"],
        )
        use_gpu = config[agent_name]["general"]["use_gpu"]
        # memory layers
        short_term_memory = MemoryDB(
            db_name=f"{agent_name}_short",
            id_generator=id_generator,
            jump_threshold_upper=config[agent_name]["short"]["jump_threshold_upper"],
            jump_threshold_lower=-999999999,
            embedding_function=embedding_function,
            importance_score_initialization=get_importance_score_initialization_func(
                type=config[agent_name]["short"]["importance_score_initialization"],
                memory_layer="short",
            ),
            recency_score_initialization=get_recency_score_initialization_func(
                type=config[agent_name]["short"]["recency_score_initialization"],
                memory_layer="short",
            ),
            compound_score_calculation=get_compound_score_calculation_func(
                type="linear", memory_layer="short"
            ),
            importance_score_change_access_counter=get_access_counter_change_func(
                config[agent_name]["short"]["importance_score_change_access_counter"],
                "short",
            ),
            decay_function=get_decay_func(
                type=config[agent_name]["short"]["decay_type"],
                memory_layer="short",
                **config[agent_name]["short"]["decay_params"],
            ),
            clean_up_threshold_dict=config[agent_name]["short"][
                "clean_up_threshold_dict"
            ],
            use_gpu=use_gpu,
        )
        mid_term_memory = MemoryDB(
            db_name=f"{agent_name}_mid",
            id_generator=id_generator,
            jump_threshold_upper=config[agent_name]["mid"]["jump_threshold_upper"],
            jump_threshold_lower=config[agent_name]["mid"]["jump_threshold_lower"],
            embedding_function=embedding_function,
            importance_score_initialization=get_importance_score_initialization_func(
                type=config[agent_name]["mid"]["importance_score_initialization"],
                memory_layer="mid",
            ),
            recency_score_initialization=get_recency_score_initialization_func(
                type=config[agent_name]["mid"]["recency_score_initialization"],
                memory_layer="mid",
            ),
            compound_score_calculation=get_compound_score_calculation_func(
                type="linear", memory_layer="mid"
            ),
            importance_score_change_access_counter=get_access_counter_change_func(
                config[agent_name]["mid"]["importance_score_change_access_counter"],
                "mid",
            ),
            decay_function=get_decay_func(
                type=config[agent_name]["mid"]["decay_type"],
                memory_layer="mid",
                **config[agent_name]["mid"]["decay_params"],
            ),
            clean_up_threshold_dict=config[agent_name]["mid"][
                "clean_up_threshold_dict"
            ],
            use_gpu=use_gpu,
        )
        long_term_memory = MemoryDB(
            db_name=f"{agent_name}_long",
            id_generator=id_generator,
            jump_threshold_upper=999999999,
            jump_threshold_lower=config[agent_name]["long"]["jump_threshold_lower"],
            embedding_function=embedding_function,
            importance_score_initialization=get_importance_score_initialization_func(
                type=config[agent_name]["long"]["importance_score_initialization"],
                memory_layer="long",
            ),
            recency_score_initialization=get_recency_score_initialization_func(
                type=config[agent_name]["long"]["recency_score_initialization"],
                memory_layer="long",
            ),
            compound_score_calculation=get_compound_score_calculation_func(
                type="linear", memory_layer="long"
            ),
            importance_score_change_access_counter=get_access_counter_change_func(
                config[agent_name]["long"]["importance_score_change_access_counter"],
                "long",
            ),
            decay_function=get_decay_func(
                type=config[agent_name]["long"]["decay_type"],
                memory_layer="long",
                **config[agent_name]["long"]["decay_params"],
            ),
            clean_up_threshold_dict=config[agent_name]["long"][
                "clean_up_threshold_dict"
            ],
            use_gpu=use_gpu,
        )
        reflection_memory = MemoryDB(
            db_name=f"{agent_name}_reflection",
            id_generator=id_generator,
            jump_threshold_upper=999999999,
            jump_threshold_lower=-999999999,
            embedding_function=embedding_function,
            importance_score_initialization=get_importance_score_initialization_func(
                type=config[agent_name]["reflection"][
                    "importance_score_initialization"
                ],
                memory_layer="reflection",
            ),
            recency_score_initialization=get_recency_score_initialization_func(
                type=config[agent_name]["reflection"]["recency_score_initialization"],
                memory_layer="reflection",
            ),
            compound_score_calculation=get_compound_score_calculation_func(
                type="linear", memory_layer="reflection"
            ),
            importance_score_change_access_counter=get_access_counter_change_func(
                config[agent_name]["reflection"][
                    "importance_score_change_access_counter"
                ],
                "reflection",
            ),
            decay_function=get_decay_func(
                type=config[agent_name]["reflection"]["decay_type"],
                memory_layer="reflection",
                **config[agent_name]["reflection"]["decay_params"],
            ),
            clean_up_threshold_dict=config[agent_name]["reflection"][
                "clean_up_threshold_dict"
            ],
            use_gpu=use_gpu,
        )
        return cls(
            agent_name=agent_name,
            id_generator=id_generator,
            embedding_function=embedding_function,
            short_term_memory=short_term_memory,
            mid_term_memory=mid_term_memory,
            long_term_memory=long_term_memory,
            reflection_memory=reflection_memory,
            use_gpu=use_gpu,
        )

    def add_memory_short(
        self, symbol: str, date: str, text: Union[List[str], str]
    ) -> None:
        self.short_term_memory.add_memory(symbol, date, text)

    def add_memory_mid(
        self, symbol: str, date: str, text: Union[List[str], str]
    ) -> None:
        self.mid_term_memory.add_memory(symbol, date, text)

    def add_memory_long(
        self, symbol: str, date: str, text: Union[List[str], str]
    ) -> None:
        self.long_term_memory.add_memory(symbol, date, text)

    def add_memory_reflection(
        self, symbol: str, date: str, text: Union[List[str], str]
    ) -> None:
        self.reflection_memory.add_memory(symbol, date, text)

    def query_short(
        self, query_text: str, top_k: int, symbol: str
    ) -> Tuple[List[str], List[int]]:
        return self.short_term_memory.query(query_text, top_k, symbol)

    def query_mid(
        self, query_text: str, top_k: int, symbol: str
    ) -> Tuple[List[str], List[int]]:
        return self.mid_term_memory.query(query_text, top_k, symbol)

    def query_long(
        self, query_text: str, top_k: int, symbol: str
    ) -> Tuple[List[str], List[int]]:
        return self.long_term_memory.query(query_text, top_k, symbol)

    def query_reflection(
        self, query_text: str, top_k: int, symbol: str
    ) -> Tuple[List[str], List[int]]:
        return self.reflection_memory.query(query_text, top_k, symbol)

    def update_access_count_with_feed_back(
        self, symbol: str, ids: Union[List[int], int], feedback: int
    ):
        if isinstance(ids, int):
            ids = [ids]
        ids = [i for i in ids if i not in self.removed_ids]
        feedback_list = list(repeat(feedback, len(ids)))  # match length
        success_ids = []
        # update short term memory
        success_ids.extend(
            self.short_term_memory.update_access_count_with_feed_back(
                symbol, ids, feedback_list
            )
        )
        ids = [i for i in ids if i not in success_ids]
        feedback_list = list(repeat(feedback, len(ids)))  # match length
        if not ids:
            return
        # update mid term memory
        success_ids.extend(
            self.mid_term_memory.update_access_count_with_feed_back(
                symbol, ids, feedback_list
            )
        )
        ids = [i for i in ids if i not in success_ids]
        feedback_list = list(repeat(feedback, len(ids)))  # match length
        if not ids:
            return
        # update long term memory
        success_ids.extend(
            self.long_term_memory.update_access_count_with_feed_back(
                symbol, ids, feedback_list
            )
        )
        ids = [i for i in ids if i not in success_ids]
        feedback_list = list(repeat(feedback, len(ids)))  # match length
        if not ids:
            return
        # reflection memory
        success_ids.extend(
            self.reflection_memory.update_access_count_with_feed_back(
                symbol, ids, feedback_list
            )
        )
        # if ids := [i for i in ids if i not in success_ids]:
        #     raise ValueError("ids conflict")
        # else:
        return

    def step(self):
        # first decay then clean up
        self.removed_ids.extend(self.short_term_memory.step())
        for cur_symbol in self.short_term_memory.universe:
            cur_memory = self.short_term_memory.universe[cur_symbol]["score_memory"]
            logger.info(f"short term memory {cur_symbol}")
            for i in range(len(cur_memory)):
                logger.info(f"memory: {cur_memory[i]}")
        self.removed_ids.extend(self.mid_term_memory.step())
        for cur_symbol in self.mid_term_memory.universe:
            cur_memory = self.mid_term_memory.universe[cur_symbol]["score_memory"]
            logger.info(f"mid term memory {cur_symbol}")
            for i in range(len(cur_memory)):
                logger.info(f"memory: {cur_memory[i]}")
        self.removed_ids.extend(self.long_term_memory.step())
        for cur_symbol in self.long_term_memory.universe:
            cur_memory = self.long_term_memory.universe[cur_symbol]["score_memory"]
            logger.info(f"long term memory {cur_symbol}")
            for i in range(len(cur_memory)):
                logger.info(f"memory: {cur_memory[i]}")
        self.removed_ids.extend(self.reflection_memory.step())
        for cur_symbol in self.reflection_memory.universe:
            cur_memory = self.reflection_memory.universe[cur_symbol]["score_memory"]
            logger.info(f"reflection term memory {cur_symbol}")
            for i in range(len(cur_memory)):
                logger.info(f"memory: {cur_memory[i]}")

        # then jump
        logger.info("Memory jump starts...")
        for _ in range(2):
            # short
            logger.info("Short term memory starts...")
            (
                jump_dict_up,
                jump_dict_down,
                deleted_ids,
            ) = self.short_term_memory.prepare_jump()
            jump_dict_short = (jump_dict_up, jump_dict_down)
            self.removed_ids.extend(deleted_ids)
            self.mid_term_memory.accept_jump(jump_dict_short, "up")
            for cur_symbol in jump_dict_up:
                logger.info(
                    f"up-{cur_symbol}: {jump_dict_up[cur_symbol]['jump_object_list']}"
                )
            for cur_symbol in jump_dict_down:
                logger.info(
                    f"down-{cur_symbol}: {jump_dict_down[cur_symbol]['jump_object_list']}"
                )
            logger.info("Short term memory ends...")
            # mid
            logger.info("Mid term memory starts...")
            (
                jump_dict_up,
                jump_dict_down,
                deleted_ids,
            ) = self.mid_term_memory.prepare_jump()
            self.removed_ids.extend(deleted_ids)
            jump_dict_mid = (jump_dict_up, jump_dict_down)
            self.long_term_memory.accept_jump(jump_dict_mid, "up")
            self.short_term_memory.accept_jump(jump_dict_mid, "down")
            for cur_symbol in jump_dict_up:
                logger.info(
                    f"up-{cur_symbol}: {jump_dict_up[cur_symbol]['jump_object_list']}"
                )
            for cur_symbol in jump_dict_down:
                logger.info(
                    f"down-{cur_symbol}: {jump_dict_down[cur_symbol]['jump_object_list']}"
                )
            logger.info("Mid term memory ends...")
            # long
            logger.info("Long term memory starts...")
            (
                log_jump_dict_up,
                log_jump_dict_down,
                deleted_ids,
            ) = self.long_term_memory.prepare_jump()
            self.removed_ids.extend(deleted_ids)
            jump_dict_long = (log_jump_dict_up, log_jump_dict_down)
            self.mid_term_memory.accept_jump(jump_dict_long, "down")
            for cur_symbol in jump_dict_up:
                logger.info(
                    f"up-{cur_symbol}: {jump_dict_up[cur_symbol]['jump_object_list']}"
                )
            for cur_symbol in jump_dict_down:
                logger.info(
                    f"down-{cur_symbol}: {jump_dict_down[cur_symbol]['jump_object_list']}"
                )
            logger.info("Long term memory ends...")
        logger.info("Memory jump ends...")

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        if os.path.exists(path):
            if not force:
                raise FileExistsError(f"Brain db {path} already exists")
            shutil.rmtree(path)
        os.mkdir(path)
        # save state dict
        state_dict = {
            "agent_name": self.agent_name,
            "use_gpu": self.use_gpu,
            "emb_func": self.embedding_function,
            "id_generator": self.id_generator,
            "removed_ids": self.removed_ids,
        }
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)
        # save memory layer
        self.short_term_memory.save_checkpoint(
            name="short_term_memory", path=path, force=force
        )
        self.mid_term_memory.save_checkpoint(
            name="mid_term_memory", path=path, force=force
        )
        self.long_term_memory.save_checkpoint(
            name="long_term_memory", path=path, force=force
        )
        self.reflection_memory.save_checkpoint(
            name="reflection_memory", path=path, force=force
        )

    @classmethod
    def load_checkpoint(cls, path: str):
        # load state dict
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
        # load memory
        short_term_memory = MemoryDB.load_checkpoint(
            os.path.join(path, "short_term_memory")
        )
        mid_term_memory = MemoryDB.load_checkpoint(
            os.path.join(path, "mid_term_memory")
        )
        long_term_memory = MemoryDB.load_checkpoint(
            os.path.join(path, "long_term_memory")
        )
        reflection_memory = MemoryDB.load_checkpoint(
            os.path.join(path, "reflection_memory")
        )
        return cls(
            agent_name=state_dict["agent_name"],
            id_generator=state_dict["id_generator"],
            embedding_function=state_dict["emb_func"],
            short_term_memory=short_term_memory,
            mid_term_memory=mid_term_memory,
            long_term_memory=long_term_memory,
            reflection_memory=reflection_memory,
            use_gpu=state_dict["use_gpu"],
        )
