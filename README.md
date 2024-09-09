# Data forensics in diffusion models: A systematic analysis of membership privacy

In this project, we will explore the field of data forensics and learn how to analyze and detect potential membership leakage attacks(MIAs) on the training data. By following the steps outlined below, we will be able to train the diffusion models, extract the necessary data for MIAs, and run attacks in different scenarios. Let's get started!

Step 1: Train the diffusion models
```bash
bash guided-diffusion[improved-diffusion]/run_model_train.sh
```

Step 2: Extract the necessary data for attacks 
```bash
bash guided-diffusion[improved-diffusion]/run_save_loss_data.sh
```

Step 3: Run the attacks
```bash
bash guided-diffusion[improved-diffusion]/run_all_attacks.sh
```
