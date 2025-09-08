import pygenn
dual_exp_model = pygenn.create_postsynaptic_model(
    "dual_exp_post",
    params=[("taur", "scalar"), ("taud", "scalar")],
    vars=[("g", "scalar")],
    sim_code= """
        injectCurrent(g);
        g += (-g/taud + inSyn)*dt;
        inSyn += -inSyn/taur*dt; 
    """
)