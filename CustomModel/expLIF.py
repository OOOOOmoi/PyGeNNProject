import pygenn
expLIF_model = pygenn.create_neuron_model(
    "expLIF",
    params=["Vthresh", "TauM", "TauRefrac", "C", "Vrest", "Vreset", "Ioffset", "DeltaT", "VT"],
    vars=[("V", "scalar", pygenn.VarAccess.READ_WRITE),
          ("RefracTime", "scalar", pygenn.VarAccess.READ_WRITE)],
    sim_code=
        """
        if (RefracTime <= 0.0) {
            scalar dV = (-(V - Vrest) + DeltaT * exp((V - VT) / DeltaT) + Rmembrane * (Ioffset + Isyn)) / TauM;
            V += dV * dt;
        }else {
            RefracTime -= dt;
        }
        """,
    threshold_condition_code="(RefracTime <= 0.0) && (V >= Vthresh)",
    reset_code=
        """
        V = Vreset;
        RefracTime = TauRefrac;
        """,
    derived_params=[("Rmembrane", lambda pars,dt:pars["TauM"]/pars["C"])],
)