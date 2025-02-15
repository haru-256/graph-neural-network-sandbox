import pathlib
import sys
from datetime import datetime

import torch
from dateutil import tz
from libs import build_benchmark_func
from line_profiler import LineProfiler
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict
from torch_geometric.data import Data
from torch_geometric.sampler import EdgeSamplerInput, NegativeSampling

if ".." not in sys.path:
    sys.path.append("..")

from sampler import sampler
from sampler.sampler import DAEMONNeighborSampler


class Settings(BaseSettings):
    """Settings for the job-runner"""

    model_config = SettingsConfigDict(cli_parse_args=True, cli_prog_name="benchmark")

    # from args
    num_calls: int = Field(description="Number of calls", default=10)
    output_dir: pathlib.Path = Field(
        description="Machine type to use for the job",
        default=pathlib.Path(
            f"./output/{datetime.now(tz=tz.gettz('Asia/Tokyo')).strftime('%Y%m%d%H%M%S')}"
        ),
    )

    def cli_cmd(self) -> None:
        logger.info("Running the bench-mark cli command")
        logger.info(self.model_dump())

        main(self.num_calls, self.output_dir)


def _sample_data():
    # graph follows the below image.
    # https://github.com/pyg-team/pytorch_geometric/discussions/9816#discussion-7583341
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    edge_index = torch.tensor(
        [[0, 4], [0, 5], [1, 0], [2, 0], [3, 0], [1, 6]]
    ).T.contiguous()
    edge_index_type = torch.tensor([0, 1, 0, 1, 0, 0])
    edge_label_index = torch.tensor([[0, 4], [1, 6]]).T.contiguous()
    return Data(
        x=x,
        edge_index=edge_index,
        edge_index_type=edge_index_type,
        edge_label_index=edge_label_index,
    )


def main(num_calls: int, output_dir: pathlib.Path):
    data = _sample_data()
    daemon_neighbor_sampler = DAEMONNeighborSampler(data=data, num_neighbors=[2, 2])
    neg_sampling = NegativeSampling(mode="triplet", amount=1)
    src = torch.tensor([0])
    dst_pos = torch.tensor([4])
    inputs = EdgeSamplerInput(input_id=None, row=src, col=dst_pos)

    profiler = LineProfiler()

    # add functions to profile
    profiler.add_function(DAEMONNeighborSampler.sample_from_edges)
    profiler.add_function(DAEMONNeighborSampler._sample)
    profiler.add_function(sampler.sample_one_hop_neighbors)

    # run the function
    _func = build_benchmark_func(
        func=daemon_neighbor_sampler.sample_from_edges,
        num_calls=num_calls,
        inputs=inputs,
        neg_sampling=neg_sampling,
    )
    profiler_wrapper = profiler(_func)
    profiler_wrapper()

    # save the profile stats
    logger.info(f"Profile stats saved to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(output_dir / "raw.lprof")
    stats_path = output_dir / "stats.txt"
    with open(stats_path, "w") as f:
        profiler.print_stats(stream=f)
    print(stats_path.read_text())


if __name__ == "__main__":
    s = CliApp.run(Settings)
