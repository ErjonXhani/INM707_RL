# File Content

All the python files in this follow the following running pattern:

```
# to play the game in the model envirnoment
!python [filename].py manual

# To train the DQN algorithm, and enter the number of episodes to train for (here it will be for 100 episodes)
!python [filename].py train 1000

# To run the inference for the trained DQN algorithm
!python [filename].py inference

```

# Files

## Additional Files

Consists of all the testing python files used before formulating the final set of algorithms and model envirnoment for all the experiments

Consists of:

1) dqn_version2.py: The final model envirnoment setting used for all the experiments
2) q-learning_Game_DDQN.py: The initial implementation of game with DDQN techniques
3) q-learning_Game_DQN.py: The initial implementation of game with DQN techniques

## Algorithms

Consits of all the final python used for experimentation.

Consists of:

1) dqn_v2.py: The final implementation for DQN on our envirnoment.
2) ddqn_v2.py: The final implementation for DDQN on our envirnoment.
3) ddqn_v2_per.py: The final implementation for DDQN + PER on our envirnoment.

## Best Models

Consists of all the saved models that has come out of the Algorithms Folder when running the same.

Do note to match the models to their respective models/files. If there are issues in loading navigate to the following code block:

```
if mode == 'inference':
        try:
            policy_net.load_state_dict(torch.load('ddqn_v2_per_model.pth'))
            policy_net.eval()
            epsilon = 0
            print("Loaded weights from ddqn_v2_per_model.pth for inference.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
```

and change the file location.

## Experiments_Checkpoints

Contains all the saved models that we experimented with, these results were essential when we did quantitaive and qualitative analysis.
