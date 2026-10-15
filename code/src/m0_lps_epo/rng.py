from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#数据类，包含11 个随机数生成器，每个生成器负责模型中的一个随机过程
@dataclass
class RNGStreams:
    network: np.random.Generator
    private_opinion: np.random.Generator
    profile: np.random.Generator
    cost: np.random.Generator
    activity: np.random.Generator
    participation: np.random.Generator
    expression_noise: np.random.Generator
    posts: np.random.Generator
    random_visibility: np.random.Generator
    algorithm_visibility: np.random.Generator
    prb_observers: np.random.Generator

#base_seed↓root SeedSequence↓某一次 replication 的种子↓11 个独立子种子↓
#11 个随机数生成器↓分别供网络、观点、参与、发帖、曝光等模块使用

def make_rng_streams(base_seed: int, replication_id: int) -> RNGStreams:
    root = np.random.SeedSequence(base_seed)
    rep_seq = root.spawn(replication_id + 1)[replication_id]
    child_seqs = rep_seq.spawn(11)
    return RNGStreams(
        network=np.random.default_rng(child_seqs[0]),
        private_opinion=np.random.default_rng(child_seqs[1]),
        profile=np.random.default_rng(child_seqs[2]),
        cost=np.random.default_rng(child_seqs[3]),
        activity=np.random.default_rng(child_seqs[4]),
        participation=np.random.default_rng(child_seqs[5]),
        expression_noise=np.random.default_rng(child_seqs[6]),
        posts=np.random.default_rng(child_seqs[7]),
        random_visibility=np.random.default_rng(child_seqs[8]),
        algorithm_visibility=np.random.default_rng(child_seqs[9]),
        prb_observers=np.random.default_rng(child_seqs[10]),
    )

