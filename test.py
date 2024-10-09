import logging
import hydra
from omegaconf import DictConfig
import os

# Custom logging setup for CSV metrics
class CSVFormatter(logging.Formatter):
    def format(self, record):
        return f"{record.asctime},{record.epoch},{record.loss},{record.accuracy},{record.regularization}"


@hydra.main(version_base=None, config_path="config", config_name="config")
def train_model(cfg: DictConfig):
    # Get Hydra's output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Set up CSV logger for metrics
    csv_logger = logging.getLogger('csv_logger')

    # Custom log record for each epoch
    for epoch in range(cfg.train.epochs):
        # Simulate metrics for the example
        loss = 0.1 * (10 - epoch)
        accuracy = 0.1 * epoch
        regularization = 0.01 * epoch

        # Create a log record with custom fields for metrics
        log_record = logging.LogRecord(
            name="csv_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=None,
            exc_info=None
        )

        # Add custom attributes for logging
        log_record.epoch = epoch
        log_record.loss = loss
        log_record.accuracy = accuracy
        log_record.regularization = regularization

        # Log the metrics in CSV-like format
        csv_logger.handle(log_record)

    print(f"Metrics saved to: {output_dir}/metrics.csv")


if __name__ == "__main__":
    train_model()
