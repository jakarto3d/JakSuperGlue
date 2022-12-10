import torch
import typer
from pathlib import Path

from jaksuperglue.engine.inference_engine import InferenceEngine
from jaksuperglue.engine.postprocessing_engine import PostProcessingEngine
from jaksuperglue.engine.preprocessing_engine import PreProcessingEngine

app = typer.Typer()

@app.command()
def inference(working_dir: str,
              root_output: str,
              device: str='cuda',
              fisheye_mask_path: str = None):


    my_preprocessing_engine = PreProcessingEngine(working_dir=working_dir,
                                                  root_output=root_output,
                                                  fisheye_mask_path=fisheye_mask_path)

    my_preprocessing_engine.process()

    my_inference_engine = InferenceEngine(working_dir=root_output,
                                          device=device)
    my_inference_engine.infer()

    my_postprocessing_engine = PostProcessingEngine(working_dir=root_output)

    my_postprocessing_engine.process()



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    app()