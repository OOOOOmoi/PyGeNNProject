import pygenn
OUnoise = pygenn.create_current_source_model(
    "OUnoise",
    params=["ave", "sd", "theta"],
    vars=[("OU", "scalar", pygenn.VarAccess.READ_WRITE)],
    injection_code=
    """
    injectCurrent(OU);
    OU = OU + theta * (ave - OU) * dt + sd * curand_normal();
    """
)