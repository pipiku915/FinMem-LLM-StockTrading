from puppy import LLMAgent
if __name__ == '__main__':
    agent = LLMAgent.load_checkpoint("./data/08_test_checkpoint/agent_1")
    df = agent.portfolio.get_action_df()
    df.write_csv("tsla_gpt3.5.csv")