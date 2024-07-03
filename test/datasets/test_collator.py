import time
import pickle
from src.scripts.batch_samplers import gpt_batch_sampler, mlm_batch_sampler, GPTDataCollator, MLMDataCollator

def timeit_gpt_collator():
    from src.models.vae import VQVAE
    from src.common.configs import ModelConfig, TotalConfigs
    from src.scripts.vae_prepare import prepare_config_dataset
    from pathlib import Path
    
    save_path = Path.home() / "PycharmProjects/BERTAP_v2/save/BERTAP_VAE-0625-1317"
    configs: ModelConfig = ModelConfig.load_from_txt(save_path)
    params = pickle.load((save_path / "model_params.pkl").open("rb"))
    loader, configs = prepare_config_dataset('antmaze-large-play-v2',
                                             seq_len=configs.seq_len,
                                             latent_step=configs.latent_step,
                                             batch_size=4,
                                             n_epochs=10)
    
    collator = GPTDataCollator(tokenizer=VQVAE(configs, training=False), params=params)
    sampler_fn, (normalizer, splits) = gpt_batch_sampler(loader, batch_size=4, data_collator=collator, normalize=True)
    
    batch = sampler_fn()
    for i in range(10):
        st  = time.time()
        batch = sampler_fn()
        print(f'Time: {time.time() - st:.4f}s')
        
    print("Done")
    
    
def timeit_mlm_collator():
    from src.models.vae import VQVAE
    from src.common.configs import ModelConfig
    from src.scripts.prior_prepare import prepare_config_dataset
    from pathlib import Path
    
    save_path = Path.home() / "PycharmProjects/BERTAP_v2/save/BERTAP_VAE-0625-1317"
    loader, prior_configs, vae_params, vae_configs = prepare_config_dataset(save_path, batch_size=4, n_epochs=10)
    collator = MLMDataCollator(VQVAE(vae_configs.model_config, training=False), vae_params, prior_configs.model_config)
    sampler_fn, (normalizer, splits) = mlm_batch_sampler(loader, batch_size=4, data_collator=collator, normalize=True)
    
    batch = sampler_fn()
    for i in range(10):
        st = time.time()
        batch = sampler_fn()
        print(f"Time: {time.time() - st:.4f}s")
        
    print("Done")