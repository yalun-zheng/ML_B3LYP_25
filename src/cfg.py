import yaml, ml_collections

def get_cfg(yaml_file='cfg.yaml'):
    with open(yaml_file) as cfg_f:
        cfg = yaml.full_load(cfg_f)
    for key in cfg.get('basic', []):
        for domain in cfg:
            if key not in cfg[domain]:
                cfg[domain][key] = cfg['basic'][key]
    return ml_collections.ConfigDict(cfg)