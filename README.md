# pbshm-damage-detection

This repository houses a comprehensive, automated pipeline for Structural Health Monitoring (SHM) and damage detection. Built for scalability toward Population-Based SHM (PBSHM), the framework integrates three core modules:

1. Sensor Network Placement Optimization (SNPO): Utilizes genetic algorithms to determine highly practical, sparse sensor layouts that maximize modal observability.

2. Structural Simulation & Data Generation: Automates ETABS/SAP2000 via the CSI OAPI to synthetically generate thousands of unique structural damage scenarios and extract dynamic responses.

3. Deep Learning Diagnostics: Employs autoencoder bottlenecks and neural networks to process sparse sensor data, map it to a shared latent space, and accurately predict damage severity across structural zones.