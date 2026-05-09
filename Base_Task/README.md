The following contains terminal commands to run the python files for the base task

```!python env/dungeon_env.py``` --> to create the environment

```!python q_learning.py``` --> trains Q-table and saves models/q_table.pkl

```!python visualize_env.py``` --> to visualize training, with random actions

```!python watched_trained_agent.py``` --> loads q_table.pkl and visualizes learned behavior

```!python run_q_experiments.py``` --> runs different experiments with different configurations and saves graphs at results/.

```!python test_env.py & python watch_agent.py``` --> used only for environment testing and functioning purposes.
