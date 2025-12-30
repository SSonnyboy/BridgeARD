# BridgeARD
BridgeARD: Bridging the Latent Gap for Dual-Teacher Adversarial Robust Distillation

## Instructions for Reproducing Results

1. **Environment Setup**  
   Ensure you are using **Python 3.8**. Install all required packages using:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download Teacher Models**  
   - Download the **clean teacher** model checkpoint and place it in:  
     `checkpoints/nat_teacher_checkpoint/`  
   - Download the **robust teacher** model and place it in:
     `checkpoints/adv_teacher_checkpoint`  
     The models we used can be found [here](https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_robustness).

3. **Configuration**  
   - You can modify the configuration in `config` to change the settings.

4. **Run BridgeARD**
   ```bash
   sh run.sh
   ```

4. **Eval BridgeARD**
   ```bash
   sh test.sh
   ```