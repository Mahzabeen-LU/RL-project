from gym.envs.registration import register

register(
    id='vnf-v0',
    entry_point='vnf_env.envs:Vnf_Env',
)

#run pip install -e . on parent folder