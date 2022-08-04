function [fid, sfo] = onedim_sim(field, field_unit, isotopes, shifts, couplings, ...
        tau_c, offset, sweep, npoints, channel)

    % System
    gamma = spin(channel);
    if field_unit == "MHz"
        sys.magnet = (2e6 * pi * field) / gamma;
        sfo = field;
    else
        sys.magnet = field;
        sfo = (field * gamma) / (2e6 * pi);
    end
    sys.isotopes = isotopes;

    % Interations
    nspins = length(shifts);
    inter.zeeman.scalar = shifts;
    inter.coupling.scalar = cell(nspins, nspins);
    for elem = couplings
        inter.coupling.scalar{elem{1}{1}, elem{1}{2}} = elem{1}{3};
    end

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

    % Spinach housekeeping
    spin_system = create(sys, inter);
    spin_system = basis(spin_system, bas);

    % Sequence parameters
    parameters.offset = offset;
    parameters.sweep = sweep;
    parameters.npoints = npoints;
    parameters.spins = {channel};
    parameters.coil=state(spin_system,'L+','1H');
    parameters.rho0=state(spin_system,'L+','1H');

    % Simulation
    fid = liquid(spin_system, @acquire, parameters, 'nmr');
end
