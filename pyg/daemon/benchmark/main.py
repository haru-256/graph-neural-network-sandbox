import pathlib
from datetime import datetime

import _sample
from dateutil import tz
from libs import build_benchmark_func
from line_profiler import LineProfiler
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the job-runner"""

    model_config = SettingsConfigDict(cli_parse_args=True, cli_prog_name="benchmark")

    # from args
    num_calls: int = Field(description="Number of calls", default=10)
    output_path: pathlib.Path = Field(
        description="Machine type to use for the job",
        default=pathlib.Path(
            f"./output/{datetime.now(tz=tz.gettz('Asia/Tokyo')).strftime('%Y%m%d%H%M%S')}.lprof"
        ),
    )

    def cli_cmd(self) -> None:
        logger.info("Running the bench-mark cli command")
        logger.info(self.model_dump())

        main(self.num_calls, self.output_path)


def main(num_calls: int, output_path: pathlib.Path):
    profiler = LineProfiler()

    # add functions to profile
    profiler.add_module(_sample)

    # run the function
    _func = build_benchmark_func(func=_sample.fn, num_calls=num_calls, length=1000)
    profiler_wrapper = profiler(_func)
    profiler_wrapper()

    profiler.print_stats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(output_path)
    logger.info(f"Profile stats saved to {output_path}")


if __name__ == "__main__":
    s = CliApp.run(Settings)
