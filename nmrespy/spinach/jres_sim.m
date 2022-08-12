function [fid, sfo] = jres_sim(field, isotopes, shifts, couplings, offset, ...
        sweep, npoints, channel)

    sfo = get_sfo(field, channel);
    sys.magnet = field;
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
