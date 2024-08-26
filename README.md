# MIMo RL Experiments

This repository contains code for training and testing reinforcement learning agents in the MIMo environment using the PPO and SAC algorithms. Follow the steps below to set up and use the repository.

## Prerequisites

You must first have the official MIMo GitHub repository cloned and set up on your local machine.

### Steps:

1. **Clone the official MIMo repository**:

    ```bash
    git clone <MIMo-GitHub-URL>
    ```

2. **Replace `selfbody.py`**:

    Copy the `selfbody.py` file from this repository and replace the existing one in the MIMo environment directory.

    ```bash
    cp selfbody.py /path-to-MIMo-repo/mimoEnv/selfbody.py
    ```

3. **Install the MIMo environment**:

    ```bash
    cd /path-to-MIMo-repo
    pip install -e .
    ```

## Repository Contents

- `MIMoEnv_Colab.ipynb`: Jupyter notebook for setting up and running the MIMo environment on Google Colab.
- `inference_ppo.py`: Script for running inference using a pre-trained PPO model.
- `inference_sac.py`: Script for running inference using a pre-trained SAC model.
- `ppo_agent.zip`: Pre-trained PPO model used in my experiments.
- `sac_agent_env3.zip`: Pre-trained SAC model used in my experiments.
- `sec-and-ppo-test.py`: Script for comparing SAC and PPO models on the same environment.
- `selfbody.py`: Custom modifications to the MIMo environment. Replace the original `selfbody.py` with this one.
- `train_ppo.py`: Script for training a PPO model in the MIMo environment.
- `train_sac.py`: Script for training a SAC model in the MIMo environment.

## Running Experiments

### Training Models

To train new models:

- **PPO**: Run `train_ppo.py`
- **SAC**: Run `train_sac.py`

### Testing Pre-trained Models

To test the provided pre-trained models:

- Use the `sec-and-ppo-test.py` script to compare SAC and PPO performance.
- Alternatively, run `inference_ppo.py` or `inference_sac.py` to evaluate individual models.

## Conclusion

This repository is designed to seamlessly integrate with the official MIMo environment. Replace the necessary file, and you can use my pre-trained models or train new ones to replicate or extend my experiments.
