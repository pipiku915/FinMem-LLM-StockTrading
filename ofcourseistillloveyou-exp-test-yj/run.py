import os
import toml
import typer
import logging
import pickle
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from typing import Union
from puppy import MarketEnvironment, LLMAgent, RunMode


# set up
load_dotenv()
app = typer.Typer(name="puppy")
warnings.filterwarnings("ignore")


@app.command("sim", help="Start Simulation", rich_help_panel="Simulation")
def sim_func(
    market_data_info_path: str = typer.Option(
        os.path.join("data", "03_model_input", "amzn.pkl"),
        "-mdp",
        "--market-data-path",
        help="The environment data pickle path",
    ),
    start_time: str = typer.Option(
        "2022-08-16", "-st", "--start-time", help="The start time"
    ),
    end_time: str = typer.Option(
        "2022-10-04", "-et", "--end-time", help="The end time"
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="Run mode: train or test"
    ),
    config_path: str = typer.Option(
        os.path.join("config", "amzn_tgi_config.toml"),
        "-cp",
        "--config-path",
        help="config file path",
    ),
    checkpoint_path: str = typer.Option(
        os.path.join("data", "06_train_checkpoint"),
        "-ckp",
        "--checkpoint-path",
        help="The checkpoint path",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "05_train_model_output"),
        "-rp",
        "--result-path",
        help="The result save path",
    ),
    trained_agent_path: Union[str, None] = typer.Option(
        None,
        "-tap",
        "--trained-agent-path",
        help="Only used in test mode, the path of trained agent",
    ),
) -> None:
    # load config
    config = toml.load(config_path)
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        os.path.join(
            "data",
            "04_model_output_log",
            f'{config["general"]["trading_symbol"]}_run.log',
        ),
        mode="a",
    )
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)
    # verify run mode
    if run_mode in {"train", "test"}:
        run_mode_var = RunMode.Train if run_mode == "train" else RunMode.Test
    else:
        raise ValueError("Run mode must be train or test")
    # create environment
    with open(market_data_info_path, "rb") as f:
        env_data_pkl = pickle.load(f)
    environment = MarketEnvironment(
        symbol=config["general"]["trading_symbol"],
        env_data_pkl=env_data_pkl,
        start_date=datetime.strptime(start_time, "%Y-%m-%d").date(),
        end_date=datetime.strptime(end_time, "%Y-%m-%d").date(),
    )
    if run_mode_var == RunMode.Train:
        the_agent = LLMAgent.from_config(config)
    else:
        the_agent = LLMAgent.load_checkpoint(path=os.path.join(trained_agent_path, "agent_1"))  # type: ignore
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
        os.path.join("data", "06_train_checkpoint"),
        "-ckp",
        "--checkpoint-path",
        help="The checkpoint path",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "05_train_model_output"),
        "-rp",
        "--result-path",
        help="The result save path",
    ),
    config_path: str = typer.Option(
        os.path.join("config", "aapl_tgi_config.toml"),
        "-cp",
        "--config-path",
        help="config file path",
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="Run mode: train or test"
    ),
) -> None:
    # load config
    config = toml.load(config_path)
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        os.path.join(
            "data",
            "04_model_output_log",
            f'{config["general"]["trading_symbol"]}_run.log',
        ),
        mode="a",
    )
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)
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
