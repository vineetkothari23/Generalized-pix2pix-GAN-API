# Designing a scalable deep learning pipeline and keeping track of multiple training.

A deep learning task involves 
- complex deep pipelines with many parameters
- tweaking architectures
- tuning hyperparameters by trying all varied experiments.

It is often difficult to track all the hit and trial versions of the training

Here is where designing a **directed acyclic graph (DAG) workflow** comes into picture.
This is migrating a fixed task flow into a **scalable deep learning pipeline**

