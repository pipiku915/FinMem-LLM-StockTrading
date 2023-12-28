# FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) [![arXiv](https://img.shields.io/badge/arXiv-2311.13743-b31b1b.svg)](https://arxiv.org/abs/2311.13743)

```text
"So we beat on, boats against the current, borne back ceaselessly into the past."
                                        -- F. Scott Fitzgerald: The Great Gatsby
```

This repo provides the Python source code for the paper:
[FINMEM: A Performance-Enhanced Large Language Model Trading Agent with Layered Memory and Character Design](https://arxiv.org/abs/2311.13743)[[PDF]](https://arxiv.org/pdf/2311.13743.pdf)

```bibtex
@article{yu2023finme,
  title={FinMe: A Performance-Enhanced Large Language Model Trading Agent with Layered Memory and Character Design},
  author={Yu, Yangyang and Li, Haohang and Chen, Zhi and Jiang, Yuechen and Li, Yang and Zhang, Denghui and Liu, Rong and Suchow, Jordan W and Khashanah, Khaldoun},
  journal={arXiv preprint arXiv:2311.13743},
  year={2023}
}
```

## Abstract

Recent advancements in Large Language Models (LLMs) have exhibited notable efficacy in question-answering (QA) tasks across diverse domains. Their prowess in integrating extensive web knowledge has fueled interest in developing LLM-based autonomous agents. While LLMs are efficient in decoding human instructions and deriving solutions by holistically processing historical inputs, transitioning to purpose-driven agents requires a supplementary rational architecture to process multi-source information, establish reasoning chains, and prioritize critical tasks. Addressing this, we introduce FinMem, a novel LLM-based agent framework devised for financial decision-making, encompassing three core modules: Profiling, to outline the agent's characteristics; Memory, with layered processing, to aid the agent in assimilating realistic hierarchical financial data; and Decision-making, to convert insights gained from memories into investment decisions. Notably, FinMem's memory module aligns closely with the cognitive structure of human traders, offering robust interpretability and real-time tuning. Its adjustable cognitive span allows for the retention of critical information beyond human perceptual limits, thereby enhancing trading outcomes. This framework enables the agent to self-evolve its professional knowledge, react agilely to new investment cues, and continuously refine trading decisions in the volatile financial environment. We first compare FinMem with various algorithmic agents on a scalable real-world financial dataset, underscoring its leading trading performance in stocks and funds. We then fine-tuned the agent's perceptual spans to achieve a significant trading performance. Collectively, FinMem presents a cutting-edge LLM agent framework for automated trading, boosting cumulative investment returns.

![1](figures/memory_flow.png)
![2](figures/workflow.png)
![3](figures/character.png)

## Usage

### Docker Setup

We recommend using Docker to run the code. A development container set up with VSCode is available at [devcontainer.json](.devcontainer/devcontainer.json).

### Dependencies

The code is tested on Python 3.10. All dependencies can be installed via [poetry](https://python-poetry.org/) with the following command:

```bash
poetry config virtualenvs.in-project true  # optional: install virtualenv in project
poetry install
```

We recommend using [pipx](https://pypa.github.io/pipx/) to install poetry. After installing dependencies, you can activate the virtual environment with `poetry shell` or `source .venv/bin/activate` if virtualenv is installed in project folder.

### Run Code

The entry point of the code is `run.py`. Use

```bash
python run.py --help
```

to see the available options. All configurations are stored in `config/config.toml`.

### Run Simulation

```bash
python run.py sim
```

with the following default options:

```bash
 --market-data-path  -mdp      TEXT  The environment data pickle path [default: data/06_input/subset_symbols.pkl]                                          │
│ --start-time        -st       TEXT  The start time [default: 2022-04-04]                                                                                  │
│ --end-time          -et       TEXT  The end time [default: 2022-06-15]                                                                                    │
│ --run-model         -rm       TEXT  Run mode: train or test [default: train]                                                                              │
│ --config-path       -cp       TEXT  config file path [default: config/config.toml]                                                                        │
│ --checkpoint-path   -ckp      TEXT  The checkpoint path [default: data/09_checkpoint]                                                                     │
│ --result-path       -rp       TEXT  The result save path [default: data/11_train_result]                               Show this message and exit.  
```

As the OpenAI API is not always stable, the running process may be interrupted in many circumstances. The training process will automatically save the checkpoint at every step. You can resume the training process by running `python run.py sim-checkpoint` with the following default options:

```bash
--checkpoint-path  -cp      TEXT  The checkpoint path [default: data/09_checkpoint]                                                                                                                 │
│ --result-path      -rp      TEXT  The result save path [default: data/11_train_result]                                                                                                              │
│ --run-model        -rm      TEXT  Run mode: train or test [default: train] 
```

## Notes

### Data Sources

| Type | Source | Note | Download Method |
|-|-|-|-|
| Daily Stock Price | Yahoo Finance | OHLCV | [yfinance](https://pypi.org/project/yfinance/) |
| Daily Market News | Alpaca Market News API | Historical Daily News | [Alpaca News API](https://docs.alpaca.markets/docs/news-api) |
| Company 10K | SEC EDGAR | Item 7 | [SEC API Section Extractor API](https://sec-api.io/docs/sec-filings-item-extraction-api)  |
| Company 10Q | SEC EDGAR | Part 1 Item 2 | [SEC API Section Extractor API](https://sec-api.io/docs/sec-filings-item-extraction-api) |

### Data Schemas

After downloaded data from the above sources, each dataset need to be processed with following schemas to be able to convert to the environment data format.

#### Daily Stock Price

| Column | Type | Note |
|-|-|-|
| Date | datetime | - |
| Open | float | - |
| High | float | - |
| Low | float | - |
| Close | float | - |
| Adj Close | float | Adjusted Close Price |
| Volume | float | - |
|Symbol| str | Ticker Symbol |

#### Daily Market News

| Column | Type | Note |
|-|-|-|
| author | str | - |
| content | str | Almost empty when downloaded from Alpaca API|
| datetime | datetime | - |
| date | datetime | Align the news datetime with trading hours, i.e., the news after 4 PM is moved to the next day at 9 AM. |
| source | str | - |
| summary| str | - |
| title | str | - |
| url | str | - |
| equity | str | Ticker Symbol |
| text | str | Concatenate title and summary |

#### Company 10K & 10Q

| Column | Type | Note |
|-|-|-|
| document_url | str | EDGAR File Archive Link |
| content | str | - |
| ticker | str | Ticker Symbol |
| utc_timestamp | datetime | - |
| est_timestamp | datetime | - |
| type | str | "10-K" or "10-Q" |
