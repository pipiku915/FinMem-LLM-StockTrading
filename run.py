import os
import toml
import openai
import typer
import logging
import pickle
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from puppy import MarketEnvironment, LLMAgent, RunMode


# set up
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = typer.Typer(name="puppy")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler("run.log", mode="a")
file_handler.setFormatter(logging_formatter)
logger.addHandler(file_handler)


@app.command("sim", help="Start Simulation", rich_help_panel="Simulation")
def sim_func(
    market_data_info_path: str = typer.Option(
        os.path.join("data", "06_input", "subset_symbols.pkl"),
        "-mdp",
        "--market-data-path",
        help="The environment data pickle path",
    ),
    start_time: str = typer.Option(
        "2022-04-04", "-st", "--start-time", help="The start time"
    ),
    end_time: str = typer.Option(
        "2022-06-15", "-et", "--end-time", help="The end time"
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="Run mode: train or test"
    ),
    config_path: str = typer.Option(
        os.path.join("config", "config.toml"),
        "-cp",
        "--config-path",
        help="config file path",
    ),
    checkpoint_path: str = typer.Option(
        os.path.join("data", "09_checkpoint"),
        "-ckp",
        "--checkpoint-path",
        help="The checkpoint path",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "11_train_result"),
        "-rp",
        "--result-path",
        help="The result save path",
    ),
) -> None:
    # verify run mode
    if run_mode in {"train", "test"}:
        run_mode_var = RunMode.Train if run_mode == "train" else RunMode.Test
    else:
        raise ValueError("Run mode must be train or test")
    # load config
    config = toml.load(config_path)
    # create environment
    with open(market_data_info_path, "rb") as f:
        env_data_pkl = pickle.load(f)
    environment = MarketEnvironment(
        symbol=config["general"]["trading_symbol"],
        env_data_pkl=env_data_pkl,
        start_date=datetime.strptime(start_time, "%Y-%m-%d").date(),
        end_date=datetime.strptime(end_time, "%Y-%m-%d").date(),
    )
    # create agent
    the_agent = LLMAgent.from_config(config)
    # start simulation
    pbar = tqdm(total=environment.simulation_length)
    while True:
        logger.info(f"Step {the_agent.counter}")
        the_agent.counter += 1
        market_info = environment.step()
        logger.info(f"Date {market_info[0]}")
        logger.info(f"Record {market_info[-2]}")
        if market_info[-1]:  # if done break
            break
        the_agent.step(market_info=market_info, run_mode=run_mode_var)  # type: ignore
        pbar.update(1)
        # save checkpoint every time, openai api is not stable
        the_agent.save_checkpoint(path=checkpoint_path, force=True)
        environment.save_checkpoint(path=checkpoint_path, force=True)
    # save result after finish
    the_agent.save_checkpoint(path=result_path, force=True)
    environment.save_checkpoint(path=result_path, force=True)


@app.command(
    "sim-checkpoint",
    help="Start Simulation from checkpoint",
    rich_help_panel="Simulation",
)
def sim_checkpoint(
    checkpoint_path: str = typer.Option(
        os.path.join("data", "09_checkpoint"),
        "-cp",
        "--checkpoint-path",
        help="The checkpoint path",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "11_train_result"),
        "-rp",
        "--result-path",
        help="The result save path",
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="Run mode: train or test"
    ),
) -> None:
    # verify run mode
    if run_mode in {"train", "test"}:
        run_mode_var = RunMode.Train if run_mode == "train" else RunMode.Test
    else:
        raise ValueError("Run mode must be train or test")
    # load env & agent from checkpoint
    environment = MarketEnvironment.load_checkpoint(
        path=os.path.join(checkpoint_path, "env")
    )
    the_agent = LLMAgent.load_checkpoint(path=os.path.join(checkpoint_path, "agent_1"))
    pbar = tqdm(total=environment.simulation_length)
    # run simulation
    while True:
        logger.info(f"Step {the_agent.counter}")
        the_agent.counter += 1
        market_info = environment.step()
        if market_info[-1]:
            break
        the_agent.step(market_info=market_info, run_mode=run_mode_var)  # type: ignore
        pbar.update(1)
        # save checkpoint every time, openai api is not stable
        the_agent.save_checkpoint(path=checkpoint_path, force=True)
        environment.save_checkpoint(path=checkpoint_path, force=True)
    # save result after finish
    the_agent.save_checkpoint(path=result_path, force=True)
    environment.save_checkpoint(path=result_path, force=True)


if __name__ == "__main__":
    app()
