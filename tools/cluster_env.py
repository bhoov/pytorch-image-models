import logging
import os
import re
from typing import Optional

from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.environments.slurm_environment import SLURMEnvironment


log = logging.getLogger(__name__)

class AIMOSEnvironment(SLURMEnvironment):
    """Cluster environment for training on a cluster managed by SLURM. Modified for AIMOS SLURM weirdness
    """
    def world_size(self) -> int:
        return int(self.num_nodes() * self.gpus_per_node())
    
    def gpus_per_node(self) -> int:
        return int(os.environ["SLURM_GPUS_PER_NODE"])
    
    def num_nodes(self) -> int:
        return int(os.environ["SLURM_NNODES"])
    
    

if __name__ == "__main__":
    env = AIMOSEnvironment(True)
    print("slurm detected:", env.detect())

    print("main address:", env.main_address)
    print("main port:", env.main_port)

    print("world size/number of GPUs:", env.world_size())
    print("local rank:", env.local_rank()) # The issue is that THIS NEVER INCREMENTS

    print("global rank (rank of process GPU):", env.global_rank())
    print("node rank:", env.node_rank())
    # print("\tproc rank (should match above):", os.environ["SLURM_PROCID"])
    
#     print("job name:", env.job_name())
#     print("job id:", env.job_id())
#     print("slurm detected:", env.detect())
    
#     print("main address:", env.main_address)
#     # print("\troot_node_address (should match above):", env.resolve_root_node_address(os.environ["SLURM_JOB_NODELIST"]))
#     print("main port:", env.main_port)

#     print("GPUS per node: ", env.gpus_per_node())

#     print("world size/number of nodes:", env.world_size())
#     print("local rank:", env.local_rank()) # The issue is that THIS NEVER INCREMENTS

#     print("global rank (rank of GPU in total machines):", env.global_rank())
#     print("node rank:", env.node_rank())
#     print("\tproc rank (should match above):", os.environ["SLURM_PROCID"])