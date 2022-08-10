function [fid, sfo] = jres_sim(field, field_unit, isotopes, shifts, couplings, ...
        tau_c, offset, sweep, npoints, channel)

    [magnet, sfo] = get_magnet_and_sfo(field, field_unit, channel);
    sys.magnet = magnet;
    sys.isotopes = isotopes;

    % Interations
    nspins = length(shifts);
    inter.zeeman.scalar = shifts;
    inter.coupling.scalar = get_couplings(nspins, couplings);

    % Basis set
    bas.formalism = 'sphten-liouv';
    bas.approximation = 'IK-2';
    bas.connectivity = 'scalar_couplings';
    bas.space_level = 1;

    % Relaxation theory parameters
    inter.relaxation = {'redfield'};
    inter.equilibrium = 'zero';
    inter.rlx_keep = 'kite';
    inter.tau_c = {tau_c};

    % Sequence parameters
    parameters.offset = offset;
    parameters.sweep = sweep;
    parameters.npoints = npoints;
    parameters.spins = {channel};

    % Spinach housekeeping
    spin_system = create(sys, inter);
    spin_system = basis(spin_system, bas);

    % Simulation
    fid = liquid(spin_system, @jres_seq, parameters, 'nmr').';
end
