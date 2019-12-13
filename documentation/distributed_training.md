# Distributed training in pytorch

![network image]()
> Distributed computing means using different components of the network to perform a particular task in synchronization

The individual components of the network are called nodes and a bunch of nodes are called cluster.

##### Characteristics of a distributed computed task:
- **Distribute computation among multiple nodes** for coherent processing.
- Execute an **effective synchronization** amongst the nodes for consistency.

The nodes communicated amongst each other over different protocols like MPI (Message passing Interface) TCP, GLoo, NCCL.
Communication is important in this scenario to keep the variables and state of the process in sync amongst the nodes.

## Types of parallelism

![Types of parallelism](https://xiandong79.github.io/downloads/ddl1.png)

### 1. Model parallelism

The deep network of model is distributed across nodes respect with different layers of the network of nodes. 
This makes a deep complex network with many parameters to train **memory efficient**, but **not computationally efficient in terms of execution time.**

### 2. Data parallelism

- This is a popular specification to distribute large scale data amongst batches amongst different nodes of the network.

- These batches are trained on multiple replicas of the same network. So each replica can be a whole network of its own.

- It is a production quality execution of training of deep learning algorithm.

![Data parallelism](https://cwiki.apache.org/confluence/download/attachments/75977306/PS-based%20distributed%20training.png?version=1&modificationDate=1522110941000&api=v2)

## torch.distributed API
 Pytorch offers a very effecrive API over the Message passing interface.
 
 Here's a simple program to explain the key concepts involved
 
 ```
import torch
import torch.distributed as dist
def main(rank, world):
    if rank == 0:
        x = torch.tensor([1., -1.]) # Tensor of interest
        dist.send(x, dst=1) # x is the sent tensor
        print('Rank-0 has sent the following tensor to Rank-1')
        print(x)
    else:
        z = torch.tensor([0., 0.]) # A holder for recieving the tensor
        dist.recv(z, src=0) #z is the variable where the received variable is stored
        print('Rank-1 has recieved the following tensor from Rank-0')
        print(z)

if __name__ == '__main__':
    dist.init_process_group(backend='mpi')
    main(dist.get_rank(), dist.get_world_size())
 ```
 
 
 1. The variable **rank** is like the id assigned to each node by the MPI. This is helpful to reference nodes while communicating and sharing parameters amongst.
 The other variable **world** refers to the cluster/collection of nodes specified in the context.
 
 2. Here node 0 is trying to send a tensor to node 1, using dist.send() and dist.recv().
 
 3. **'backend = 'mpi''** specifies the channel/ interface over which the nodes are going to communicate.
 
 ### all-reduce 
 
 ![all-reduce image](https://pytorch.org/tutorials/_images/all_reduce.png)
 
 ## Deep learning using torch.distributed.
 
The basic usage template:

```
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```
 For training the model, using **synchronized stochastic gradient descent** we need to synchronize the gradients amongst the nodes.
 I.e. we need to sum all the gradients across all data batch points. 
 To do this we use the all_reduce function to sum across multiple nodes 
 
 ```
 def sync_gradients(model, rank, world_size):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM) #SUmming the gradients accross nodes.
 ```
 
 If we wanted to synchronize the hyperparameter, following would be the way:
 
 ```
 def sync_hyperparameters(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            for sibling in range(1, world_size):
                dist.send(param.data, dst=sibling)
        else:
            # Nodes must recieve the parameters
            dist.recv(param.data, src=0)
 ```
## References
- [WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [DISTRIBUTED COMMUNICATION PACKAGE - TORCH.DISTRIBUTED](https://pytorch.org/docs/stable/distributed.html)
- [Distributed Data parallel tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Distributed training in deep neural network](https://medium.com/intel-student-ambassadors/distributed-training-of-deep-learning-models-with-pytorch-1123fa538848)
 
