<h1 align="center"> VeriGUI: Verifiable Long-Chain GUI Dataset</h1>

<div align="center">
<a href=''><img src='https://img.shields.io/badge/arXiv-xxxx.xxxxx-%23B31B1B?logo=arxiv'></a> 
<a href='https://huggingface.co/datasets/2077AIDataFoundation/VeriGUI'><img src='https://img.shields.io/badge/Hugging%20Face-Datasets-blue?style=flat&logo=huggingface&logoColor=%23FFD21E'></a>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache_2.0-green.svg'></a>
</div>

> [!NOTE]
> This work is still in progress and additional data will be included in a future version.

---

## 🧭 Contents

- 🌟 [Updates](#-updates)
- 📖 [Overview](#-overview)
- ✨ [Features](#-features)
- 🚀 [Installation](#installation)
- 🤖 [Running Agents](#running-agents)
- 📊 [Evaluation](#evaluation)
- 🏆 [Benchmark Results](#-benchmark-results)
- 🗂️ [Project Structure](#️-project-structure)
- 👥 [Contributors](#-contributors)
- 📄 [License](#-license)
- 🎓 [Citation](#-citation)
- 📞 [Contact](#-contact)

## 🌟 Updates

<!-- Updates will be added here -->

## 📖 Overview

Recent studies have delved into constructing autonomous agents capable of performing complex Graphical User Interface (GUI)-based computer tasks, with the potential to revolutionize human-computer interaction. Despite encouraging results, existing efforts mainly focus on **short-term interactions** and rely on **outcome-only verification**, thereby limiting their scalability in real-world GUI applications that demand long-horizon task decomposition and execution.

In this work, we introduce **VeriGUI**, a novel verifiable long-chain GUI dataset designed to facilitate the development and evaluation of generalist GUI agents operating in realistic computer environments. Our dataset emphasizes two critical dimensions:

- (1) **🔗 Long-chain complexity**, with tasks decomposed into a sequence of interdependent subtasks spanning hundreds of steps, explicitly designed to allow any subtask serve as a valid starting point;
- (2) **✅ subtask-level verifiability**, which enables diverse exploration strategies within each subtask, while ensuring that each subtask-level goal remain verifiable and consistent.

The dataset consists of GUI task trajectories spanning both desktop and web, **annotated by human experts**. Extensive experiments on VeriGUI using various agents with different foundation models reveal significant performance gaps in handling long-horizon tasks, highlighting the need for more robust planning and decision-making capabilities in GUI agents.

<div align="center">
  <img src="images/intro_hd.png" alt="VeriGUI Dataset Overview" width="800">
  <p><em>An overview of the VeriGUI dataset.</em></p>
</div>

## ✨ Key Features


### 🔗 Long-Chain Complexity
- Tasks require 2-15 interdependent steps with hundreds of GUI actions
- Complex workflows spanning multiple applications and web pages
- Realistic task dependencies that require adaptive reasoning and planning
- Tasks mirror real-world computer usage patterns


### ✅ Subtask-Level Verifiability
- Fine-grained evaluation at each intermediate step, not just final outcomes
- Verifiable goals for each step while supporting diverse exploration strategies
- Open-ended interaction within steps - agents can choose different paths to achieve the same goal
- Detailed supervision signals for better error diagnosis and agent improvement


### 🌐 Multi-Environment Coverage
- Web environments: Various websites, online services, and web applications
- Desktop environments: Office software, operating systems, and professional tools (TODO)
- Cross-platform task transitions and interactions


### 👥 Human-Expert Annotation
- All trajectories are carefully created and annotated by human experts
- High-quality task instructions and step-level annotations
- Verified task feasibility and realistic workflow patterns




## 🚀 Installation

```bash
# Only for evaluating
pip install openai tqdm

# Run agents
pip install openai tqdm camel-ai[all] browser-use
```

## 🤖 Running Agents

We provide some examples of agents under the `agents` directory. You can run these agents by executing the following command:

```shell
python agents/some_agent.py
```

## 📊 Evaluation

The dataset of VeriGUI is located at [veriGUI.json](data/veriGUI.json). The format of the dataset is described in detail in the following sections.

```json
[
  {
    "id": "1", // index id
    "name": "V1_3", // name of the task
    "type": "global", // type of the task, global or causal
    "instruction": "xxxxx", // instruction for the task
    "answer": "xxxxx", // expected answer for the task, in JSON format
  },
  ......
]
```

The evaluation script `evaluate.py` can be used to evaluate the performance of agents using LLM-as-a-judge. The evaluation script expects a JSON format file with the following format:

```json
[
  {
    "id": "1", // index id
    "name": "V1_3", // name of the task
    "type": "global", // type of the task, global or causal
    "instruction": "xxxxx", // instruction for the task
    "answer": "xxxxx", // expected answer for the task, in JSON format
    "prediction": "xxxxx", // agent's predicted result
    "nsteps": 10, // number of steps taken by the agent
  },
  ......
]
```

With this file, you can run the evaluation script to get the performance of the agent:

```shell
python evaluate.py --input_file veriGUI_prediction.json --output_file output.json
```

Then, you can use `calc_avg.py` to calculate the average score of the evaluation results:

```shell
python calc_avg.py --input_file output.json
```

## 🗂️ Project Structure

```
agent-workflow-devkit/
├── agents/                 # Agent implementations
│   └── browseruse.py      # Browser-use agent example
├── data/                  # Dataset files
│   └── veriGUI.json      # Main dataset
├── evaluated/             # Evaluation results
├── predictions/           # Model predictions
├── evaluate.py           # Evaluation script
├── batch_evaluate.py     # Batch evaluation
├── calc_avg.py          # Calculate averages
└── utils.py             # Utility functions
```



## 🎓 Citation

If you find VeriGUI useful in your research, please cite our paper:

```bibtex
@article{verigui2024,
  title={VeriGUI: Verifiable Long-Chain GUI Dataset},
  author={Authors},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## 📞 Contact

For questions, suggestions, or collaborations, please feel free to:

- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/agent-workflow-devkit/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/agent-workflow-devkit/discussions)

## 👥 Contributors

We thank all contributors who have helped make VeriGUI possible. Special thanks to the research team and community members who provided valuable feedback and improvements.


## 📄 License

This project is licensed under the Apache 2.0 License.


---

<div align="center">
  <p><strong>🌟 Star us on GitHub if you find VeriGUI helpful! 🌟</strong></p>
</div>
