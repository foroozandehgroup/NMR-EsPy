function fid = invrec_sim(shifts, couplings, n_delays, max_delay, t1s, t2s, pts, sw, offset, sfo, nucleus)
    field = get_field(sfo, nucleus, offset);
    sys.magnet = field;
    nspins = length(shifts);
    isotopes = cell(nspins, 1);
    for i = 1:nspins
        isotopes{i} = nucleus;
    end
    sys.isotopes = isotopes;

    % Interations
    inter.zeeman.scalar = shifts;
    inter.coupling.scalar = get_couplings(nspins, couplings);
    inter.relaxation = {'t1_t2'};
    inter.r1_rates = t1s;
    inter.r2_rates = t2s;
    inter.equilibrium ='dibari';
    inter.rlx_keep = 'secular';
    inter.temperature = 298;

    % Basis set
    bas.formalism = 'sphten-liouv';
    bas.approximation = 'IK-2';
    bas.connectivity = 'scalar_couplings';
    bas.space_level = 1;

    % Spinach housekeeping
    spin_system = create(sys, inter);
    spin_system = basis(spin_system, bas);

    % Sequence parameters
    parameters.offset = offset;
    parameters.sweep = sw;
    parameters.npoints = pts;
    parameters.spins = {nucleus};
    parameters.n_delays = n_delays - 1;
    parameters.max_delay = max_delay;

    disp(parameters.n_delays);
    disp(parameters.max_delay);
    % Simulation
    fid = liquid(spin_system, @inv_rec, parameters, 'nmr');
end
