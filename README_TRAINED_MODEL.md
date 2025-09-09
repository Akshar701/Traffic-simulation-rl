# Running SUMO with Trained DQN Model

This guide explains how to run SUMO traffic simulation with your trained DQN model.

## Quick Start

### 1. Run with Default Configuration
```bash
python run_trained_model.py --model trained_models/dqn_final.pth
```

### 2. Run with Visualization (SUMO GUI)
```bash
python run_trained_model.py --model trained_models/dqn_final.pth --visualize
```

### 3. Run Multiple Episodes
```bash
python run_trained_model.py --model trained_models/dqn_final.pth --episodes 5 --verbose
```

## Available Models

- `trained_models/dqn_episode_100.pth` - Model saved at episode 100
- `trained_models/dqn_episode_200.pth` - Model saved at episode 200  
- `trained_models/dqn_final.pth` - Final trained model

## Available SUMO Configurations

- `Sumo_env/gpt_newint/intersection.sumocfg` - Default intersection
- `Sumo_env/gpt_newint/uniform_simulation.sumocfg` - Uniform traffic
- `Sumo_env/gpt_newint/congested_simulation.sumocfg` - Congested traffic
- `Sumo_env/gpt_newint/tidal_simulation.sumocfg` - Tidal traffic patterns
- `Sumo_env/gpt_newint/random_simulation.sumocfg` - Random traffic

## Command Line Options

```bash
python run_trained_model.py [OPTIONS]

Options:
  --model PATH           Path to trained model (.pth file) [REQUIRED]
  --config PATH          SUMO configuration file
  --episodes N           Number of episodes to run (default: 1)
  --max-steps N          Maximum steps per episode (default: 1000)
  --visualize            Run with SUMO GUI visualization
  --save-results         Save results to JSON file
  --verbose              Verbose output
```

## Examples

### Test Different Configurations
```bash
# Test with uniform traffic
python run_trained_model.py --model trained_models/dqn_final.pth --config Sumo_env/gpt_newint/uniform_simulation.sumocfg

# Test with congested traffic
python run_trained_model.py --model trained_models/dqn_final.pth --config Sumo_env/gpt_newint/congested_simulation.sumocfg

# Test with tidal traffic
python run_trained_model.py --model trained_models/dqn_final.pth --config Sumo_env/gpt_newint/tidal_simulation.sumocfg
```

### Compare Different Models
```bash
# Test episode 100 model
python run_trained_model.py --model trained_models/dqn_episode_100.pth --episodes 3

# Test episode 200 model  
python run_trained_model.py --model trained_models/dqn_episode_200.pth --episodes 3

# Test final model
python run_trained_model.py --model trained_models/dqn_final.pth --episodes 3
```

### Run Comprehensive Tests
```bash
# Test all models with all configurations
python test_different_configs.py
```

## Understanding the Output

### Performance Levels
- 🟢 **EXCELLENT**: Reward > -20
- 🟡 **GOOD**: Reward -20 to -50
- 🟠 **FAIR**: Reward -50 to -100
- 🔴 **POOR**: Reward < -100

### Action Distribution
The model can take 4 different actions:
- **Action 0**: North-South Green
- **Action 1**: East-West Green  
- **Action 2**: North-South Left Turn
- **Action 3**: East-West Left Turn

### Key Metrics
- **Total Reward**: Cumulative reward for the episode
- **Mean Reward**: Average reward per step
- **Steps**: Number of simulation steps
- **Action Distribution**: Percentage of each action taken

## Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   ❌ Error: Model file not found: trained_models/dqn_final.pth
   ```
   - Check that the model file exists
   - Use the correct path to your model

2. **SUMO configuration not found**
   ```
   ❌ Error: Config file not found: Sumo_env/gpt_newint/intersection.sumocfg
   ```
   - Check that SUMO configuration files exist
   - Use the correct path to your config file

3. **SUMO GUI not opening**
   - Make sure SUMO is properly installed
   - Check that your system supports GUI applications
   - Try running without `--visualize` first

### Performance Tips

1. **For better performance**: Use the final trained model (`dqn_final.pth`)
2. **For visualization**: Use `--visualize` with shorter episodes (`--max-steps 500`)
3. **For testing**: Use multiple episodes (`--episodes 5`) to get average performance
4. **For analysis**: Use `--save-results` to save detailed results to JSON

## Results

Results are saved to the `training_results/` directory:
- `trained_model_results_TIMESTAMP.json` - Detailed episode results
- `config_test_results_TIMESTAMP.json` - Configuration comparison results

Each result file contains:
- Model information
- Configuration used
- Episode rewards and metrics
- Action distributions
- Performance assessments
