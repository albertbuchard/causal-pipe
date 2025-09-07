from causal_pipe.causal_pipe import CausalPipeConfig, CausalPipe


def compare_pipelines(
    data,
    config: CausalPipeConfig,
):
    config.show_plots = False
    toolkit = CausalPipe(config)
    toolkit.run_pipeline(data)
