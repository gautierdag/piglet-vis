import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from config import HogConfig
from train import train

cs = ConfigStore.instance()
cs.store(name="base_config", node=HogConfig)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: HogConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    best_model_path = train(
        cfg, job_type="pretrain", resume_from_checkpoint=cfg.pretrain.checkpoint_path
    )
    best_model_path = "/home/gautier/Desktop/hog/output/checkpoints/images_fuse_256_42/42_33f99oxq.ckpt"
    print(f"Best pretrain model saved at {best_model_path}")
    best_model_path_nlu = train(
        cfg,
        job_type="nlu",
        best_model_path=best_model_path,
        resume_from_checkpoint=cfg.pretrain.checkpoint_path,
    )
    print(f"Best NLU model saved at {best_model_path_nlu}")


if __name__ == "__main__":
    main()
