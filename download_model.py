import os
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def download_and_save_model_snapshot():
    model_repo_id = os.getenv("MODEL_REPO_ID")
    model_path = os.getenv("MODEL_PATH")

    if not all([model_repo_id, model_path]):
        error_msg = "MODEL_REPO_ID and MODEL_PATH must be set."
        logger.error(error_msg)
        raise ValueError(error_msg)

    ignore_patterns = ["*.h5", "*.ot", "*.msgpack"]

    logger.info(f"Starting download for model: {model_repo_id}")
    logger.info(f"Model will be saved to: {model_path}")

    try:
        snapshot_download(
            repo_id=model_repo_id,
            local_dir=model_path,
            ignore_patterns=ignore_patterns,
        )
        logger.info(f"Successfully downloaded {model_repo_id}")

    except Exception as e:
        logger.error(f"Failed to download {model_repo_id}. Error: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_and_save_model_snapshot()
