# Simulation-based Business Process Robustness Assessment Repository

## Overview
This repository provides a simulation-based approach for assessing **business process robustness**. 

## Abstract
Business process robustness describes a process’s ability to meet performance targets despite variations in parameters such as arrival rates or resource availability. Assessing business process robustness quantitatively is challenging because dependencies between process parameters and performance are complex and stochastic. Business Process Simulation (BPS) avoids the need to specify those dependencies by simulating the process and observing the resulting behavior. However, the space of possible parameter combinations grows quickly, which makes it challenging to identify parameters that still meet the required performance targets. In this paper, we address this challenge by proposing a simulation-based approach that systematically explores parameter configurations to efficiently identify those that maintain acceptable performance. Our evaluation demonstrates that the proposed approach substantially reduces assessment complexity compared to a naive search while accurately identifying parameter combinations under which a process remains robust. The approach provides users with transparent insights into how parameter variations influence performance, enabling a step toward a quantitative foundation for assessing and improving business process robustness.

---

## Repository Structure
- **`pipeline.ipynb`**:
  - Main notebook for orchestrating the business process robustness approach.
  - Requires configuration of all inputs.
  - Discovers a BPS model using SIMOD if specified.
  - Conducts the specified assessment approaches (our approach as well as grid search)


- **`src/evaluation.py`**:
  - Script to evaluate the approach against ta grid search ground truth.

- **`data/`**:
  - Stores input simulation logs and results per process.

- **`requirements.txt`**:
  - Lists all the Python dependencies required to run the repository.

---

## Usage
1. **Install Dependencies**:
   - Install the required Python libraries using the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run Analysis**:
   - Use `pipeline.ipynb` to process simulation logs and evaluate robustness.

3. **Visualize Results**:
   - Use `visualization_multiObjective.ipynb` to generate visualizations for results from 2D and 3D parameter spaces.

---

## Evaluation Results
The evaluation results for the robustness analysis from the experiments in the paper can be found at this **[link](https://drive.google.com/drive/folders/1l76hXWNAju-ofBOac1vL0wVNOML6N8DD?usp=sharing)**

---

## Requirements
- Python 3.8+
- Install dependencies using `requirements.txt`.

## License

This project is licensed under the [MIT License](LICENSE). 

## Acknowledgments

This repository is part of ongoing research on Business Process Simulation (BPS) and is based on empirical examples from publicly available event logs.
