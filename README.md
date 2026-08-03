# Deep Learning -- CC3092 @ UVG

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C)
![LaTeX](https://img.shields.io/badge/LaTeX-Reports-008080?logo=latex&logoColor=white)

## About This Repository

This repository hosts the academic assignments and projects for the **Deep Learning (CC3092)** course at Universidad del Valle de Guatemala. Each subfolder corresponds to a different assignment, detailed in the table of contents below. Inside each subfolder you'll find a dedicated `README.md` with assignment details, alongside written reports and project-specific documentation authored in LaTeX.

Most labs are starter notebooks provided by the course (corpus/data, initial weights, and verification cells already in place) where the assignment is to fill in the missing forward/backward-pass code blocks; each block is self-checked against a hash or tolerance, and a final cell reports the automatic grade for the code section. The per-lab READMEs note which parts were given vs. filled in.

## Structure

### Labs

| Assignment | Topic | Link | Description |
| :--------- | :---- | :--- | :---------- |
| **Lab-01** | CNN Forward & Backward Pass | [View Folder](./Lab-01/) | Filled in a from-scratch NumPy CNN pipeline (Conv -> ReLU -> MaxPool -> Dense -> Sigmoid -> BCE) on a provided starter notebook: forward pass, manual backpropagation through every layer, and a single verified gradient descent update. |
| **Lab-02** | LSTM from Scratch (PyTorch) | [View Folder](./Lab-02/) | Filled in a from-scratch PyTorch LSTM (many-to-one sequence classifier) on a provided starter notebook: forward pass, manual BPTT backpropagation through every gate, and a verified 5-iteration convergence check. |
| **Lab-03** | Seq2Seq (Encoder-Decoder LSTM) | [View Folder](./Lab-03/) | Filled in a from-scratch PyTorch Encoder-Decoder LSTM on a provided starter notebook (EN->ES translation with intentionally mismatched sentence lengths): forward pass, manual BPTT through both encoder and decoder, sparse vs. dense embedding-gradient comparison, and a 5-iteration training loop with greedy decoding. |

### Hojas de Trabajo

| Assignment | Topic | Link | Description |
| :--------- | :---- | :--- | :---------- |
| **HT-01** | Neural Network Foundations | [View Folder](./HT-01/) | Written derivations (linear-layer collapse, binary cross-entropy from maximum likelihood, chain-rule backpropagation for logistic regression) plus a from-scratch NumPy logistic regression trained via gradient descent. |
