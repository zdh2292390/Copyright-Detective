# src/cka_analyzer/feature_analysis.py

from typing import List

from .cka_analysis import run_cka_analysis
from .fisher_analysis import run_fim_analysis
from .pca_shift_analysis import run_pca_shift
from .pca_sim_analysis import run_pca_similarity
from .types import FeatureAnalysisResult

def run_feature_analysis(
    feature: str,
    model_reference_path: str,
    model_path: str,
    query: List[str],
    output_path: str,
    device: str = "cuda",
    batch_size: int = 4,
    num_batches: int = 10,
    max_length: int = 128
)-> FeatureAnalysisResult:
    """
    Unified API to run different representation analyses.

    Args:
        feature: one of "fim", "pca_shift", "pca_sim", "cka"
        model_reference_path: path or HF ID for reference model
        model_path:           path or HF ID for updated model
        query:                list of input strings
        output_path:          for 'fim' this is an output directory;
                              for others, a PDF file path
        batch_size:           (fim & cka) DataLoader batch size
        num_batches:          (fim & cka) how many batches to process
        max_length:           tokenizer max_length (for all methods)
    """
    feature = feature.lower()
    if feature == "fim":
        return run_fim_analysis(
            model_reference_path=model_reference_path,
            model_path=model_path,
            query=query,
            device=device,
            batch_size=batch_size,
            num_batches=num_batches,
            max_length=max_length
        )

    elif feature == "pca_shift":
        return run_pca_shift(
            model_reference_path=model_reference_path,
            model_path=model_path,
            query=query,
            device=device,
            max_length=max_length
        )

    elif feature == "pca_sim":
        return run_pca_similarity(
            model_reference_path=model_reference_path,
            model_path=model_path,
            query=query,
            device=device,
            max_length=max_length
        )

    elif feature == "cka":
        return run_cka_analysis(
            model_reference_path=model_reference_path,
            model_path=model_path,
            query=query,
            device=device,
            batch_size=batch_size,
            num_batches=num_batches,
            max_length=max_length
        )

    else:
        raise ValueError(
            f"Unknown feature: {feature!r}. "
            "Choose from 'fim', 'pca_shift', 'pca_sim', 'cka'."
        )
