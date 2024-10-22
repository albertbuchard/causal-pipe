from causal_pipe.causal_pipe import CausalPipeConfig, CausalPipe


def compare_pipelines(
    data,
    config: CausalPipeConfig,
):

    toolkit = CausalPipe(config)
    toolkit.run_pipeline(data)
